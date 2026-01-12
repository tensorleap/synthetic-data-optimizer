"""
End-to-end integration test for OptunaOptimizer with full pipeline.

Tests the optimizer with actual:
- VoidGenerator
- DinoV2Embedder
- PCA projection
- Metrics computation
- ExperimentRunner integration
"""

import pytest
import optuna
import tempfile
import shutil
from pathlib import Path

from src.optimization.optuna_optimizer import OptunaOptimizer
from src.data_generation.void_generator import VoidGenerator
from src.data_generation.parameter_sampler import ParameterSampler
from src.embedding.dinov2_embedder import DinoV2Embedder
from src.embedding.pca_projector import PCAProjector
from src.optimization.metrics import compute_all_metrics, compute_per_param_set_metrics
from src.visualization.experiment_reporter import ExperimentReporter
from src.utils.bounds_inference import get_param_bounds


@pytest.fixture
def temp_test_dir():
    """Create temporary test directory"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def param_bounds_and_groups():
    """Get param_bounds and group_names from the real pipeline"""
    return get_param_bounds()


@pytest.fixture
def mini_config(temp_test_dir):
    """Minimal config for fast end-to-end test"""
    return {
        'experiment_name': 'test_e2e_optuna',
        'experiment_dir': str(temp_test_dir / 'experiment'),
        'base_image_dir': 'data/base_chips',
        'dino_model': 'dinov2_vits14',  # Smallest DinoV2 model for speed
        'pca_embedding_dim': 50,  # Reduced for speed
        'random_seed': 42,
        'iteration_batch_size': 2,  # Small batch for speed
        'replications_per_iteration': 1,  # Single replication for speed
        'max_iterations': 3,
        'optimization_metrics': ['mmd_rbf', 'mean_nn_distance'],
        'param_precision': {
            'base_size': 1,
            'rotation': 1,
            'center_x': 2,
            'center_y': 2,
            'position_spread': 2
        },
        'optimizer': {
            'n_startup_trials': 2,
            'multivariate': True
        },
    }


class TestOptunaE2EIntegration:
    """End-to-end integration tests with full pipeline"""

    def _create_metrics_list(self, n_param_sets: int, base_metrics: dict) -> list:
        """Helper to create metrics_list with distribution_id for each distribution"""
        return [
            {**base_metrics, 'distribution_id': i}
            for i in range(n_param_sets)
        ]

    def _create_distributions(self, n: int, group_names: list, param_bounds: dict) -> list:
        """Create mock distributions within bounds for testing."""
        distributions = []
        for i in range(n):
            group = group_names[i % len(group_names)]
            bounds = param_bounds[group]

            # Create params within bounds using midpoint values
            params = {}
            for param_name, bound in bounds.items():
                if isinstance(bound, list) and len(bound) == 2:
                    params[param_name] = (bound[0] + bound[1]) / 2

            distributions.append((group, params))
        return distributions

    def test_optimizer_with_real_components(self, mini_config, temp_test_dir, param_bounds_and_groups):
        """Test OptunaOptimizer with real generator, embedder, and metrics"""
        param_bounds, group_names = param_bounds_and_groups

        # Initialize components
        experiment_dir = Path(mini_config['experiment_dir'])
        experiment_dir.mkdir(parents=True, exist_ok=True)

        optimizer = OptunaOptimizer(
            experiment_dir,
            mini_config,
            param_bounds=param_bounds,
            group_names=group_names
        )
        sampler = ParameterSampler()
        generator = VoidGenerator(Path(mini_config['base_image_dir']))
        embedder = DinoV2Embedder(model_name=mini_config['dino_model'])

        # Generate "real" distribution (small sample)
        print("\n[Test] Generating real distribution...")
        real_params = sampler.sample_parameter_sets('real', n_sets=2, seed=42)
        real_images, _ = generator.generate_batch(real_params, replications=1, seed_offset=0)

        # Extract embeddings and project to lower dimension
        print("[Test] Extracting embeddings...")
        real_embeddings_full = embedder.embed_batch(real_images)

        # Fit PCA
        pca = PCAProjector(n_components=mini_config['pca_embedding_dim'])
        real_embeddings = pca.fit_transform(real_embeddings_full)

        print(f"[Test] Real embeddings shape: {real_embeddings.shape}")

        # Run 2 optimization iterations
        iteration_results = []

        # Initialize with starting distributions for iteration 0
        n_distributions = mini_config['iteration_batch_size']
        current_distributions = self._create_distributions(n_distributions, group_names, param_bounds)

        for iteration in range(2):
            print(f"\n[Test] === Iteration {iteration} ===")

            # Get metrics for current distributions
            if iteration == 0:
                # Create initial metrics_list
                metrics_list = self._create_metrics_list(
                    n_distributions,
                    compute_all_metrics(real_embeddings, real_embeddings)
                )
            else:
                metrics_list = iteration_results[-1]['metrics_list']

            # Ask optimizer for next distributions (unified interface)
            next_distributions = optimizer.suggest_next_distributions(
                current_distributions=current_distributions,
                metrics_list=metrics_list,
                config=mini_config
            )

            # Sample parameters from distributions (using new grouped format)
            next_params = []
            for dist_idx, (group_name, dist_params) in enumerate(next_distributions):
                nested_spec = sampler.grouped_to_nested_dist_spec(group_name, dist_params)
                params = sampler.sample_from_distribution_spec(
                    nested_spec,
                    n_samples=mini_config['replications_per_iteration'],
                    seed=iteration * 1000 + dist_idx
                )
                # Tag with distribution_id
                for p in params:
                    p['distribution_id'] = dist_idx
                next_params.extend(params)

            # Generate synthetic images with sampled parameters
            print(f"[Test] Generating {len(next_params)} synthetic samples...")
            synthetic_images, synthetic_metadata = generator.generate_batch(
                next_params,
                replications=1,  # Already sampled
                seed_offset=iteration * 100000
            )

            # Extract and project embeddings
            synthetic_embeddings_full = embedder.embed_batch(synthetic_images)
            synthetic_embeddings = pca.transform(synthetic_embeddings_full)

            # Compute per-distribution metrics
            n_distributions = mini_config['iteration_batch_size']
            metrics_list = compute_per_param_set_metrics(
                synthetic_embeddings,
                synthetic_metadata,
                real_embeddings,
                n_param_sets=n_distributions
            )

            # Also compute aggregate metrics for logging
            metrics = compute_all_metrics(synthetic_embeddings, real_embeddings)

            print(f"[Test] Metrics: mmd_rbf={metrics['mmd_rbf']:.4f}, "
                  f"wasserstein={metrics['wasserstein']:.4f}, "
                  f"mean_nn_distance={metrics['mean_nn_distance']:.4f}")

            # Store results
            iteration_results.append({
                'iteration': iteration,
                'params': next_params,
                'embeddings': synthetic_embeddings,
                'metrics': metrics,
                'metrics_list': metrics_list
            })

            # Update current_distributions for next iteration
            current_distributions = next_distributions

        # Verify basic integration
        assert len(iteration_results) == 2, "Should complete 2 iterations"

        # Verify optimizer produced valid parameters
        expected_params = mini_config['iteration_batch_size'] * mini_config['replications_per_iteration']
        for result in iteration_results:
            assert len(result['params']) == expected_params
            for params in result['params']:
                assert 'void_shape' in params
                assert 'void_count' in params
                assert params['void_count'] >= 1 and params['void_count'] <= 10
                assert 'distribution_id' in params  # Should be tagged with distribution ID

        # Verify metrics were computed
        import numpy as np
        for result in iteration_results:
            metrics = result['metrics']
            assert 'mmd_rbf' in metrics
            assert 'wasserstein' in metrics
            assert 'mean_nn_distance' in metrics
            # Check core metrics are numeric
            assert isinstance(metrics['mmd_rbf'], (float, np.floating, np.integer))
            assert isinstance(metrics['wasserstein'], (float, np.floating, np.integer))
            assert isinstance(metrics['mean_nn_distance'], (float, np.floating, np.integer))

        # Verify Pareto front exists
        pareto_front = optimizer.get_pareto_front()
        assert len(pareto_front) > 0, "Pareto front should have solutions"
        print(f"\n[Test] Final Pareto front size: {len(pareto_front)}")

        # Verify each trial has correct number of objectives from config
        n_metrics = len(mini_config['optimization_metrics'])
        for trial in pareto_front:
            assert len(trial.values) == n_metrics, f"Each trial should have {n_metrics} objective values"

        # Verify SQLite persistence
        assert optimizer.study_path.exists(), "SQLite study file should exist"

        print("[Test] ✓ End-to-end integration test passed!")

    def test_add_trial_pattern_e2e(self, mini_config, temp_test_dir, param_bounds_and_groups):
        """
        Test add_trial pattern in end-to-end workflow.
        Verifies that trials are properly registered via add_trial() and
        the optimizer correctly suggests new distributions.
        """
        import numpy as np
        param_bounds, group_names = param_bounds_and_groups

        # Initialize components
        experiment_dir = Path(mini_config['experiment_dir'])
        experiment_dir.mkdir(parents=True, exist_ok=True)

        optimizer = OptunaOptimizer(
            experiment_dir,
            mini_config,
            param_bounds=param_bounds,
            group_names=group_names
        )
        sampler = ParameterSampler()
        generator = VoidGenerator(Path(mini_config['base_image_dir']))
        embedder = DinoV2Embedder(model_name=mini_config['dino_model'])

        # Generate real distribution
        print("\n[E2E Test] Generating real distribution...")
        real_params = sampler.sample_parameter_sets('real', n_sets=2, seed=42)
        real_images, _ = generator.generate_batch(real_params, replications=1, seed_offset=0)
        real_embeddings_full = embedder.embed_batch(real_images)

        pca = PCAProjector(n_components=mini_config['pca_embedding_dim'])
        real_embeddings = pca.fit_transform(real_embeddings_full)

        # Track trial counts throughout experiment
        trial_count_history = []

        # Initialize with starting distributions
        n_distributions = mini_config['iteration_batch_size']
        current_distributions = self._create_distributions(n_distributions, group_names, param_bounds)

        # Run 3 iterations and verify add_trial pattern
        for iteration in range(3):
            print(f"\n[E2E Test] === Iteration {iteration} ===")

            # Compute metrics for current distributions
            # (In a real pipeline, this would involve generating images and computing metrics)
            metrics_list = self._create_metrics_list(
                n_distributions,
                {'mmd_rbf': 0.5 - iteration * 0.1, 'mean_nn_distance': 1.0 - iteration * 0.1}
            )

            # Count trials before this iteration
            completed_before = len([t for t in optimizer.study.trials
                                   if t.state == optuna.trial.TrialState.COMPLETE])

            # Get next distributions (registers current results via add_trial)
            next_distributions = optimizer.suggest_next_distributions(
                current_distributions=current_distributions,
                metrics_list=metrics_list,
                config=mini_config
            )

            # Count trials after this iteration
            completed_after = len([t for t in optimizer.study.trials
                                  if t.state == optuna.trial.TrialState.COMPLETE])

            # Verify trials were added
            trials_added = completed_after - completed_before
            print(f"[E2E Test] Trials added: {trials_added}, total completed: {completed_after}")

            trial_count_history.append({
                'iteration': iteration,
                'trials_added': trials_added,
                'total_completed': completed_after,
                'pending_count': len(optimizer.pending_trials)
            })

            # Verify n_distributions trials were added
            assert trials_added == n_distributions, \
                f"Expected {n_distributions} trials added, got {trials_added}"

            # Verify pending trials list has n_distributions entries
            assert len(optimizer.pending_trials) == n_distributions, \
                f"Expected {n_distributions} pending trials, got {len(optimizer.pending_trials)}"

            # Update for next iteration
            current_distributions = next_distributions

        # Verify final state
        print("\n[E2E Test] Verifying final state...")

        # Verify completed trials count (3 iterations * n_distributions)
        completed_trials = [t for t in optimizer.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        expected_completed = 3 * n_distributions
        assert len(completed_trials) == expected_completed, \
            f"Expected {expected_completed} completed trials, got {len(completed_trials)}"

        # Verify each completed trial has correct number of objectives from config
        n_metrics = len(mini_config['optimization_metrics'])
        for trial in completed_trials:
            assert len(trial.values) == n_metrics, f"Each trial should have {n_metrics} objective values"
            assert all(isinstance(v, (float, int)) for v in trial.values), "All values should be numeric"

        # Verify Pareto front exists
        pareto_front = optimizer.get_pareto_front()
        assert len(pareto_front) > 0, "Pareto front should have solutions"

        print(f"[E2E Test] Completed trials: {len(completed_trials)}")
        print(f"[E2E Test] Pareto front size: {len(pareto_front)}")
        print(f"[E2E Test] Trial count history:")
        for state in trial_count_history:
            print(f"  Iteration {state['iteration']}: added={state['trials_added']}, "
                  f"total={state['total_completed']}, pending={state['pending_count']}")

        print("[E2E Test] ✓ Add trial pattern E2E test passed!")

    def test_metrics_per_param_set(self, mini_config, temp_test_dir, param_bounds_and_groups):
        """
        Test that metrics are computed per parameter set correctly.
        Each param set = multiple samples from same distribution = same shape.
        """
        import numpy as np
        param_bounds, group_names = param_bounds_and_groups

        # Initialize components
        experiment_dir = Path(mini_config['experiment_dir'])
        experiment_dir.mkdir(parents=True, exist_ok=True)

        optimizer = OptunaOptimizer(
            experiment_dir,
            mini_config,
            param_bounds=param_bounds,
            group_names=group_names
        )
        sampler = ParameterSampler()
        generator = VoidGenerator(Path(mini_config['base_image_dir']))
        embedder = DinoV2Embedder(model_name=mini_config['dino_model'])

        # Generate real distribution
        real_params = sampler.sample_parameter_sets('real', n_sets=2, seed=42)
        real_images, _ = generator.generate_batch(real_params, replications=1, seed_offset=0)
        real_embeddings_full = embedder.embed_batch(real_images)

        pca = PCAProjector(n_components=mini_config['pca_embedding_dim'])
        real_embeddings = pca.fit_transform(real_embeddings_full)

        # Get distributions from optimizer
        n_distributions = mini_config['iteration_batch_size']
        initial_distributions = self._create_distributions(n_distributions, group_names, param_bounds)
        initial_metrics_list = self._create_metrics_list(n_distributions, {'mmd_rbf': 0.5, 'mean_nn_distance': 1.0})
        next_distributions = optimizer.suggest_next_distributions(
            current_distributions=initial_distributions,
            metrics_list=initial_metrics_list,
            config=mini_config
        )

        # Sample with multiple replications per distribution
        replications_per_dist = 3
        next_params = []
        for dist_idx, (group_name, dist_params) in enumerate(next_distributions):
            nested_spec = sampler.grouped_to_nested_dist_spec(group_name, dist_params)
            params = sampler.sample_from_distribution_spec(
                nested_spec,
                n_samples=replications_per_dist,
                seed=dist_idx * 1000
            )
            for p in params:
                p['distribution_id'] = dist_idx
            next_params.extend(params)

        # Verify all samples within a param set have same shape
        print("\n[Test] Verifying samples per param set:")
        for dist_idx in range(len(next_distributions)):
            dist_params = [p for p in next_params if p['distribution_id'] == dist_idx]
            shapes = [p['void_shape'] for p in dist_params]
            print(f"  Distribution {dist_idx}: {len(dist_params)} samples, shapes={shapes}")
            # All samples should have same shape (from same group)
            assert len(set(shapes)) == 1, f"All samples in dist {dist_idx} should have same shape"

        # Generate and compute metrics
        synthetic_images, synthetic_metadata = generator.generate_batch(
            next_params,
            replications=1,
            seed_offset=0
        )

        synthetic_embeddings_full = embedder.embed_batch(synthetic_images)
        synthetic_embeddings = pca.transform(synthetic_embeddings_full)

        # Compute per-distribution metrics
        n_distributions = len(next_distributions)
        metrics_list = compute_per_param_set_metrics(
            synthetic_embeddings,
            synthetic_metadata,
            real_embeddings,
            n_param_sets=n_distributions
        )

        # Verify metrics_list structure
        assert len(metrics_list) == n_distributions
        for i, metrics in enumerate(metrics_list):
            assert 'distribution_id' in metrics
            assert 'mmd_rbf' in metrics
            assert 'mean_nn_distance' in metrics
            print(f"  Distribution {metrics['distribution_id']}: mmd_rbf={metrics['mmd_rbf']:.4f}")

        print("[Test] ✓ Metrics per distribution test passed!")
