"""
Experiment Runner for orchestrating the full optimization loop.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from ..data_generation.parameter_sampler import ParameterSampler
from ..data_generation.void_generator import VoidGenerator
from ..embedding.dinov2_embedder import DinoV2Embedder
from ..embedding.pca_projector import PCAProjector
from ..optimization.metrics import compute_all_metrics
from ..optimization.optimizer_placeholder import suggest_next_parameters
from .iteration_manager import IterationManager


class ExperimentRunner:
    """Orchestrates the full optimization experiment"""

    def __init__(self, config_path: Path):
        """
        Initialize experiment runner.

        Args:
            config_path: Path to experiment configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.sampler = ParameterSampler()
        self.generator = VoidGenerator(Path(self.config['base_image_dir']))
        self.embedder = DinoV2Embedder(model_name=self.config['dino_model'])

        # PCA projectors (will be fitted in iteration 0)
        self.pca_embedding = None  # 768 -> 400D for optimization
        self.pca_viz = None  # 400D -> 2D for visualization

        # Iteration manager
        self.iteration_manager = IterationManager(Path(self.config['experiment_dir']))

        # Store real embeddings (fixed reference)
        self.real_embeddings_768d = None
        self.real_embeddings_400d = None

        print(f"Initialized ExperimentRunner")
        print(f"Experiment directory: {self.config['experiment_dir']}")

    def run_iteration_0(self) -> Dict:
        """
        Run iteration 0: Generate real distribution and initial synthetic distribution.

        Returns:
            Dictionary with iteration 0 results
        """
        print("\n" + "=" * 60)
        print("ITERATION 0: Initialization")
        print("=" * 60)

        seed = self.config['random_seed']
        initial_condition = self.config['initial_condition']

        print(f"\nInitial condition: {initial_condition}")

        # Generate real distribution
        print("\n[1/5] Generating real distribution...")
        real_params = self.sampler.sample_parameter_sets(
            'real',
            n_sets=self.config['real']['param_sets'],
            seed=seed
        )
        real_images, real_metadata = self.generator.generate_batch(
            real_params,
            replications=self.config['real']['replications'],
            seed_offset=0
        )
        print(f"Generated {len(real_images)} real images")

        # Generate initial synthetic distribution based on initial_condition
        print(f"\n[2/5] Generating synthetic distribution from '{initial_condition}'...")
        synthetic_params = self.sampler.sample_parameter_sets(
            initial_condition,
            n_sets=self.config[initial_condition]['param_sets'],
            seed=seed + 100
        )
        synthetic_images, synthetic_metadata = self.generator.generate_batch(
            synthetic_params,
            replications=self.config[initial_condition]['replications'],
            seed_offset=1000
        )
        print(f"Generated {len(synthetic_images)} synthetic images")

        # Extract embeddings
        print("\n[3/5] Extracting embeddings with DiNOv2...")
        real_embeddings = self.embedder.embed_batch(real_images)
        synthetic_embeddings = self.embedder.embed_batch(synthetic_images)

        # Fit PCA on combined data
        print("\n[4/5] Fitting PCA models...")
        all_embeddings = np.vstack([real_embeddings, synthetic_embeddings])

        # Stage 1: Fit PCA for optimization (768 -> 400D)
        print("  - Stage 1: 768 -> 400D for optimization")
        self.pca_embedding = PCAProjector(n_components=self.config['pca_embedding_dim'])
        all_embeddings_400d = self.pca_embedding.fit_transform(all_embeddings, verbose=False)

        # Stage 2: Fit PCA for visualization (400D -> 2D)
        print("  - Stage 2: 400D -> 2D for visualization")
        self.pca_viz = PCAProjector(n_components=self.config['pca_visualization_dim'])
        self.pca_viz.fit_transform(all_embeddings_400d, verbose=False)

        # Save PCA models
        pca_embedding_path = self.iteration_manager.models_dir / "pca_embedding_400d.pkl"
        pca_viz_path = self.iteration_manager.models_dir / "pca_viz_2d.pkl"
        self.pca_embedding.save(pca_embedding_path)
        self.pca_viz.save(pca_viz_path)

        # Transform to 400D embeddings
        real_embeddings_400d = self.pca_embedding.transform(real_embeddings)
        synthetic_embeddings_400d = self.pca_embedding.transform(synthetic_embeddings)

        # Store real embeddings as reference
        self.real_embeddings_768d = real_embeddings
        self.real_embeddings_400d = real_embeddings_400d

        # Calculate metrics
        print("\n[5/5] Computing baseline metrics...")
        metrics = compute_all_metrics(synthetic_embeddings_400d, real_embeddings_400d)

        print(f"\n  Synthetic ({initial_condition}) vs Real:")
        print(f"    MMD (RBF): {metrics['mmd_rbf']:.4f}")
        print(f"    Wasserstein: {metrics['wasserstein']:.4f}")
        print(f"    Mean NN distance: {metrics['mean_nn_distance']:.4f}")
        print(f"    Coverage: {metrics['coverage']:.4f}")

        # Save iteration 0 data
        # Save both synthetic and real embeddings as separate files
        iter_0_embeddings = {
            'synthetic': synthetic_embeddings_400d,
            'real': real_embeddings_400d
        }

        self.iteration_manager.save_iteration(
            iteration=0,
            params=synthetic_params,
            embeddings=iter_0_embeddings,
            metrics=metrics,
            metadata={
                'initial_condition': initial_condition,
                'n_real': len(real_images),
                'n_synthetic': len(synthetic_images),
                'pca_explained_variance_400d': float(self.pca_embedding.explained_variance_ratio_.sum()),
                'pca_explained_variance_2d': float(self.pca_viz.explained_variance_ratio_.sum())
            }
        )

        print("\nIteration 0 complete!")

        return {
            'metrics': metrics,
            'initial_condition': initial_condition
        }

    def run_iteration(self, iteration: int) -> Tuple[Dict, bool]:
        """
        Run a single optimization iteration.

        Args:
            iteration: Iteration number (1, 2, 3, ...)

        Returns:
            Tuple of (metrics_dict, converged)
        """
        print("\n" + "=" * 60)
        print(f"ITERATION {iteration}")
        print("=" * 60)

        # Load previous iteration data
        prev_data = self.iteration_manager.load_iteration(iteration - 1)
        prev_embeddings = prev_data['embeddings']
        prev_params = prev_data['params']
        prev_metrics = prev_data['metrics']

        # Get next parameter sets from optimizer
        print("\n[1/4] Getting next parameter sets from optimizer...")
        next_params, converged = suggest_next_parameters(
            synthetic_embeddings=prev_embeddings,
            synthetic_params=prev_params,
            real_embeddings=self.real_embeddings_400d,
            metrics=prev_metrics,
            iteration=iteration,
            config=self.config
        )
        print(f"Optimizer suggested {len(next_params)} parameter sets")
        if converged:
            print("Optimizer reports convergence!")

        # Generate synthetic images with suggested parameters
        print("\n[2/4] Generating synthetic images...")
        synthetic_images, synthetic_metadata = self.generator.generate_batch(
            next_params,
            replications=self.config['replications_per_iteration'],
            seed_offset=iteration * 10000
        )
        print(f"Generated {len(synthetic_images)} synthetic images")

        # Extract and transform embeddings
        print("\n[3/4] Extracting embeddings...")
        synthetic_embeddings_768d = self.embedder.embed_batch(synthetic_images)
        synthetic_embeddings_400d = self.pca_embedding.transform(synthetic_embeddings_768d)

        # Calculate metrics
        print("\n[4/4] Computing metrics...")
        metrics = compute_all_metrics(synthetic_embeddings_400d, self.real_embeddings_400d)

        print(f"\n  Synthetic vs Real:")
        print(f"    MMD (RBF): {metrics['mmd_rbf']:.4f}")
        print(f"    Wasserstein: {metrics['wasserstein']:.4f}")
        print(f"    Mean NN distance: {metrics['mean_nn_distance']:.4f}")
        print(f"    Coverage: {metrics['coverage']:.4f}")

        # Save iteration data
        self.iteration_manager.save_iteration(
            iteration=iteration,
            params=next_params,
            embeddings=synthetic_embeddings_400d,
            metrics=metrics,
            metadata={
                'n_images': len(synthetic_images),
                'converged': converged
            }
        )

        print(f"\nIteration {iteration} complete!")

        return metrics, converged

    def run(self) -> Dict:
        """
        Run the full experiment from iteration 0 to convergence.

        Returns:
            Dictionary with experiment summary
        """
        print("\n" + "=" * 60)
        print("STARTING EXPERIMENT")
        print("=" * 60)
        print(f"Experiment: {self.config['experiment_name']}")
        print(f"Max iterations: {self.config['max_iterations']}")

        # Run iteration 0 (initialization)
        iter0_results = self.run_iteration_0()

        # Collect results
        all_metrics = [iter0_results]
        converged = False

        # Run optimization iterations
        for iteration in range(1, self.config['max_iterations'] + 1):
            metrics, converged = self.run_iteration(iteration)
            all_metrics.append(metrics)

            if converged:
                print(f"\nConverged at iteration {iteration}!")
                break

        # Create summary
        summary = {
            'experiment_name': self.config['experiment_name'],
            'total_iterations': iteration,
            'converged': converged,
            'all_metrics': all_metrics
        }

        self.iteration_manager.save_summary(summary)

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"Total iterations: {iteration}")
        print(f"Converged: {converged}")
        print(f"Results saved to: {self.config['experiment_dir']}")

        return summary
