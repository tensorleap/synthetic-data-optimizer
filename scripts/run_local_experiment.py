"""
Local experiment runner with data generation.

This script runs the complete optimization loop using generated data.
For testing the framework end-to-end.

Usage:
    python -m scripts.run_local_experiment --config configs/experiment_config.yaml
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

from src.orchestration.experiment_runner import ExperimentRunner
from src.data_generation.parameter_sampler import ParameterSampler
from src.data_generation.void_generator import VoidGenerator
from src.embedding.dinov2_embedder import DinoV2Embedder
from src.embedding.pca_projector import PCAProjector


def generate_initial_distributions(
    config: Dict,
    sampler: ParameterSampler,
    n_distributions: int
) -> List[Tuple[str, Dict]]:
    """
    Generate initial distributions from config.

    Samples from the 'initial_condition' distribution type and converts
    to the grouped format expected by the optimizer.

    Returns:
        List of (group_name, params_dict) tuples
    """
    initial_condition = config.get('initial_condition', 'close')
    seed = config.get('random_seed', 42)

    # Sample parameter sets
    param_sets = sampler.sample_parameter_sets(
        initial_condition,
        n_sets=n_distributions,
        seed=seed
    )

    # Convert to grouped format: (group_name, distribution_params)
    distributions = []
    for params in param_sets:
        group_name = params['void_shape']
        # Extract distribution parameters (mean/std for each param)
        dist_params = {}
        for key, value in params.items():
            if key != 'void_shape':
                # For initial data, use the sampled value as mean with small std
                dist_params[f'{key}_mean'] = value
                dist_params[f'{key}_std'] = 0.1 * abs(value) if value != 0 else 0.1
        distributions.append((group_name, dist_params))

    return distributions


def generate_synthetic_data(
    distributions: List[Tuple[str, Dict]],
    sampler: ParameterSampler,
    generator: VoidGenerator,
    embedder: DinoV2Embedder,
    pca: PCAProjector,
    replications: int,
    seed_offset: int
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Generate synthetic data for given distributions.

    Args:
        distributions: List of (group_name, params_dict) tuples
        sampler: Parameter sampler
        generator: Void generator
        embedder: DinoV2 embedder
        pca: Fitted PCA projector
        replications: Number of samples per distribution
        seed_offset: Seed offset for reproducibility

    Returns:
        embeddings_400d: Generated embeddings (N, 400)
        metadata: List of metadata dicts with param_set_id
    """
    all_params = []

    for dist_idx, (group_name, dist_params) in enumerate(distributions):
        # Convert grouped format to nested spec
        nested_spec = sampler.grouped_to_nested_dist_spec(group_name, dist_params)

        # Sample concrete parameters
        params = sampler.sample_from_distribution_spec(
            nested_spec,
            n_samples=replications,
            seed=seed_offset + dist_idx
        )

        # Tag with distribution ID
        for p in params:
            p['param_set_id'] = dist_idx
        all_params.extend(params)

    # Generate images
    images, metadata = generator.generate_batch(
        all_params,
        replications=1,
        seed_offset=seed_offset
    )

    # Extract embeddings
    embeddings_768d = embedder.embed_batch(images)
    embeddings_400d = pca.transform(embeddings_768d)

    return embeddings_400d, metadata


def run_local_experiment(config_path: Path):
    """
    Run complete optimization loop with generated data.

    Args:
        config_path: Path to experiment config YAML
    """
    print("\n" + "=" * 60)
    print("LOCAL EXPERIMENT (with data generation)")
    print("=" * 60)

    # Initialize experiment runner
    runner = ExperimentRunner(config_path)
    config = runner.config

    # Initialize data generation components
    sampler = ParameterSampler()
    generator = VoidGenerator(Path(config['base_image_dir']))
    embedder = DinoV2Embedder(model_name=config['dino_model'])

    # Setup: Generate real embeddings
    print("\n[Setup] Generating real distribution...")
    seed = config.get('random_seed', 42)
    real_params = sampler.sample_parameter_sets(
        'real',
        n_sets=config['real']['param_sets'],
        seed=seed
    )
    real_images, _ = generator.generate_batch(
        real_params,
        replications=config['real']['replications'],
        seed_offset=0
    )
    print(f"  Generated {len(real_images)} real images")

    # Extract embeddings and fit PCA
    print("  Extracting embeddings and fitting PCA...")
    real_embeddings_768d = embedder.embed_batch(real_images)
    pca = PCAProjector(n_components=config['pca_embedding_dim'])
    real_embeddings_400d = pca.fit_transform(real_embeddings_768d, verbose=False)

    # Set real embeddings in runner
    runner.set_real_embeddings(real_embeddings_400d)

    # Generate initial distributions
    n_distributions = config.get('iteration_batch_size', 8)
    current_distributions = generate_initial_distributions(config, sampler, n_distributions)
    print(f"\n[Setup] Generated {len(current_distributions)} initial distributions")

    # Run optimization loop
    max_iterations = config.get('max_iterations', 10)
    replications = config.get('replications_per_iteration', 10)

    print(f"\nStarting optimization loop (max {max_iterations} iterations)...")

    for iteration in range(max_iterations):
        # Generate synthetic data for current distributions
        embeddings_400d, metadata = generate_synthetic_data(
            current_distributions,
            sampler,
            generator,
            embedder,
            pca,
            replications=replications,
            seed_offset=(iteration + 1) * 10000
        )

        # Run iteration
        next_suggestions = runner.run_iteration(
            iteration=iteration,
            current_distributions=current_distributions,
            synthetic_embeddings_400d=embeddings_400d,
            synthetic_metadata=metadata
        )

        # Use suggestions for next iteration
        current_distributions = next_suggestions

    print("\n" + "=" * 60)
    print("LOCAL EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Total iterations: {max_iterations}")
    print(f"Results saved to: {config['experiment_dir']}")


def main():
    parser = argparse.ArgumentParser(description='Run local experiment with data generation')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment_config.yaml',
        help='Path to experiment config YAML'
    )
    args = parser.parse_args()

    run_local_experiment(Path(args.config))


if __name__ == '__main__':
    main()
