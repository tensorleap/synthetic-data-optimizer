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
        """Helper to create metrics_list with param_set_id for each distribution"""
        return [
            {**base_metrics, 'param_set_id': f'dist_{i:03d}'}
            for i in range(n_param_sets)
        ]

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

        for iteration in range(1, 3):
            print(f"\n[Test] === Iteration {iteration} ===")

            # Get previous iteration data (or empty for iteration 1)
            if iteration == 1:
                # Create dummy metrics_list for iteration 0 (won't be used)
                prev_metrics_list = self._create_metrics_list(
                    len(real_params),
                    compute_all_metrics(real_embeddings, real_embeddings)
                )
            else:
                prev_metrics_list = iteration_results[-1]['metrics_list']

            # Ask optimizer for next distributions
            next_distributions, converged = optimizer.suggest_next_distributions(
                metrics_list=prev_metrics_list,
                iteration=iteration,
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
                'metrics_list': metrics_list,
                'converged': converged
            })

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

    def test_ask_tell_pattern_e2e(self, mini_config, temp_test_dir, param_bounds_and_groups):
        """
        Test proper ask/tell pattern in end-to-end workflow.
        Verifies that pending trials are properly tracked and completed.
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

        # Track pending trials state throughout experiment
        pending_trials_history = []

        # Run 3 iterations and verify ask/tell pattern
        prev_metrics_list = self._create_metrics_list(
            len(real_params),
            compute_all_metrics(real_embeddings, real_embeddings)
        )

        for iteration in range(3):
            print(f"\n[E2E Test] === Iteration {iteration} ===")

            # Get next distributions (triggers ask/tell)
            next_distributions, converged = optimizer.suggest_next_distributions(
                metrics_list=prev_metrics_list,
                iteration=iteration,
                config=mini_config
            )

            # Verify pending trials state
            print(f"[E2E Test] Pending trials after iteration {iteration}: {list(optimizer.pending_trials.keys())}")
            pending_trials_history.append({
                'iteration': iteration,
                'pending_iterations': list(optimizer.pending_trials.keys()),
                'n_pending_trials': {k: len(v) for k, v in optimizer.pending_trials.items()}
            })

            # Verify current iteration has pending trials
            assert iteration in optimizer.pending_trials, f"Iteration {iteration} should have pending trials"
            assert len(optimizer.pending_trials[iteration]) == mini_config['iteration_batch_size']

            # Verify previous iteration was completed (if exists)
            if iteration > 0:
                assert (iteration - 1) not in optimizer.pending_trials, \
                    f"Iteration {iteration-1} should have been completed"

            # Sample parameters from distributions (using new grouped format)
            next_params = []
            for dist_idx, (group_name, dist_params) in enumerate(next_distributions):
                nested_spec = sampler.grouped_to_nested_dist_spec(group_name, dist_params)
                params = sampler.sample_from_distribution_spec(
                    nested_spec,
                    n_samples=mini_config['replications_per_iteration'],
                    seed=iteration * 1000 + dist_idx
                )
                for p in params:
                    p['distribution_id'] = dist_idx
                next_params.extend(params)

            # Generate synthetic images with sampled parameters
            print(f"[E2E Test] Generating {len(next_params)} synthetic samples...")
            synthetic_images, synthetic_metadata = generator.generate_batch(
                next_params,
                replications=1,
                seed_offset=iteration * 100000
            )

            # Extract embeddings
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

            print(f"[E2E Test] Iteration {iteration} metrics:")
            print(f"  mmd_rbf: {metrics['mmd_rbf']:.4f}")
            print(f"  wasserstein: {metrics['wasserstein']:.4f}")
            print(f"  mean_nn_distance: {metrics['mean_nn_distance']:.4f}")

            # Update for next iteration
            prev_metrics_list = metrics_list

        # Verify final state
        print("\n[E2E Test] Verifying final state...")

        # Only last iteration should have pending trials
        assert len(optimizer.pending_trials) == 1, "Only last iteration should have pending trials"
        assert 2 in optimizer.pending_trials, "Iteration 2 should have pending trials"

        # Verify completed trials count
        completed_trials = [t for t in optimizer.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        expected_completed = 2 * mini_config['iteration_batch_size']  # Iterations 0 and 1
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
        print(f"[E2E Test] Pending trials state history:")
        for state in pending_trials_history:
            print(f"  Iteration {state['iteration']}: pending={state['pending_iterations']}, counts={state['n_pending_trials']}")

        print("[E2E Test] ✓ Ask/tell pattern E2E test passed!")

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
        prev_metrics_list = self._create_metrics_list(2, {'mmd_rbf': 0.5, 'mean_nn_distance': 1.0})
        next_distributions, _ = optimizer.suggest_next_distributions(
            metrics_list=prev_metrics_list,
            iteration=0,
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
            assert 'param_set_id' in metrics
            assert 'mmd_rbf' in metrics
            assert 'mean_nn_distance' in metrics
            print(f"  Param set {metrics['param_set_id']}: mmd_rbf={metrics['mmd_rbf']:.4f}")

        print("[Test] ✓ Metrics per param set test passed!")
