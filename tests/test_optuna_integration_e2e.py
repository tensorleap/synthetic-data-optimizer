"""
End-to-end integration test for OptunaOptimizer with full pipeline.

Tests the optimizer with actual:
- VoidGenerator
- DinoV2Embedder
- PCA projection
- Metrics computation
- ExperimentRunner integration

NOTE: These tests use joint optimization format where each distribution contains:
- shape_logit_* keys (converted to probabilities via softmax)
- {shape}__{param} keys for all parameters of all shapes
"""

import pytest
import optuna
import tempfile
import shutil
from pathlib import Path

from src.optimization.optuna_optimizer import OptunaOptimizer
from src.data_generation.void_generator import VoidGenerator
from src.embedding.dinov2_embedder import DinoV2Embedder
from src.embedding.pca_projector import PCAProjector
from src.optimization.metrics import compute_all_metrics, compute_per_param_set_metrics
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

    def _create_joint_distributions(
        self,
        n: int,
        group_names: list,
        param_bounds: dict,
        logit_bounds: tuple = (-5.0, 5.0)
    ) -> list:
        """
        Create mock joint distributions for testing.

        Each distribution contains:
        - shape_logit_* keys for all groups
        - {group}__{param} keys for all params of all groups

        Returns:
            List of (dist_id, params_dict) tuples
        """
        import math

        distributions = []
        for i in range(n):
            params = {}

            # Add shape logits (equal probabilities by default)
            for group_name in group_names:
                # Slight variation per distribution to differentiate them
                base_logit = 0.0 + (i * 0.1)
                params[f'shape_logit_{group_name}'] = base_logit

            # Add all params for all groups using midpoint values
            for group_name in group_names:
                group_bounds = param_bounds.get(group_name, {})
                for param_name, bound in group_bounds.items():
                    optuna_key = f'{group_name}__{param_name}'
                    if isinstance(bound, list) and len(bound) == 2:
                        # Add slight variation per distribution
                        mid = (bound[0] + bound[1]) / 2
                        variation = (bound[1] - bound[0]) * 0.1 * i
                        params[optuna_key] = mid + variation
                    elif isinstance(bound, list):
                        # Categorical - use first value
                        params[optuna_key] = bound[0]

            distributions.append((f"dist_{i}", params))

        return distributions

    def _joint_params_to_concrete_samples(
        self,
        joint_params: dict,
        group_names: list,
        n_samples: int,
        seed: int = 42
    ) -> list:
        """
        Convert joint distribution params to concrete samples.

        Uses softmax on logits to get probabilities, then samples shapes
        according to those probabilities and uses shape-specific params.

        Args:
            joint_params: Dict with shape_logit_* and {shape}__{param} keys
            group_names: List of group names
            n_samples: Number of concrete samples to generate
            seed: Random seed

        Returns:
            List of concrete parameter dicts with 'void_shape' and other params
        """
        import numpy as np

        rng = np.random.default_rng(seed)

        # Get probabilities from logits via softmax
        probs = OptunaOptimizer.logits_to_probabilities(joint_params)

        # Sample shapes according to probabilities
        prob_values = [probs.get(g, 1.0 / len(group_names)) for g in group_names]
        prob_values = np.array(prob_values) / sum(prob_values)  # Normalize

        samples = []
        for _ in range(n_samples):
            # Sample shape
            shape = rng.choice(group_names, p=prob_values)

            # Extract shape-specific params
            sample = {'void_shape': shape}
            prefix = f'{shape}__'
            for key, value in joint_params.items():
                if key.startswith(prefix):
                    param_name = key[len(prefix):]
                    # Remove _mean/_std suffix for concrete params
                    if param_name.endswith('_mean'):
                        base_param = param_name[:-5]
                        # Use mean directly (in real pipeline, would sample from distribution)
                        # Cast void_count to int
                        if base_param == 'void_count':
                            sample[base_param] = int(round(value))
                        else:
                            sample[base_param] = value
                    elif param_name.endswith('_std'):
                        pass  # Skip std for now (would be used for sampling)
                    else:
                        sample[param_name] = value

            # Ensure rotation is always present (default 0.0 for non-ellipse)
            if 'rotation' not in sample:
                sample['rotation'] = 0.0

            samples.append(sample)

        return samples

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
        generator = VoidGenerator(Path(mini_config['base_image_dir']))
        embedder = DinoV2Embedder(model_name=mini_config['dino_model'])

        # Generate "real" distribution using simple params
        print("\n[Test] Generating real distribution...")
        real_params = [
            {'void_shape': 'circle', 'void_count': 3, 'base_size': 10.0,
             'center_x': 0.5, 'center_y': 0.5, 'position_spread': 0.1, 'rotation': 0.0},
            {'void_shape': 'ellipse', 'void_count': 2, 'base_size': 12.0,
             'center_x': 0.5, 'center_y': 0.5, 'position_spread': 0.1, 'rotation': 45.0}
        ]
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

        # Initialize with starting distributions (joint format)
        n_distributions = mini_config['iteration_batch_size']
        current_distributions = self._create_joint_distributions(
            n_distributions, group_names, param_bounds
        )

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

            # Convert joint distributions to concrete samples
            next_params = []
            for dist_idx, (dist_id, joint_params) in enumerate(next_distributions):
                samples = self._joint_params_to_concrete_samples(
                    joint_params,
                    group_names,
                    n_samples=mini_config['replications_per_iteration'],
                    seed=iteration * 1000 + dist_idx
                )
                # Tag with distribution_id
                for p in samples:
                    p['distribution_id'] = dist_idx
                next_params.extend(samples)

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
        generator = VoidGenerator(Path(mini_config['base_image_dir']))
        embedder = DinoV2Embedder(model_name=mini_config['dino_model'])

        # Generate real distribution
        print("\n[E2E Test] Generating real distribution...")
        real_params = [
            {'void_shape': 'circle', 'void_count': 3, 'base_size': 10.0,
             'center_x': 0.5, 'center_y': 0.5, 'position_spread': 0.1, 'rotation': 0.0}
        ]
        real_images, _ = generator.generate_batch(real_params, replications=1, seed_offset=0)
        real_embeddings_full = embedder.embed_batch(real_images)

        pca = PCAProjector(n_components=mini_config['pca_embedding_dim'])
        real_embeddings = pca.fit_transform(real_embeddings_full)

        # Track trial counts throughout experiment
        trial_count_history = []

        # Initialize with starting distributions (joint format)
        n_distributions = mini_config['iteration_batch_size']
        current_distributions = self._create_joint_distributions(
            n_distributions, group_names, param_bounds
        )

        # Run 3 iterations and verify add_trial pattern
        for iteration in range(3):
            print(f"\n[E2E Test] === Iteration {iteration} ===")

            # Compute metrics for current distributions
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

            # Verify output format is joint (dist_id, full_params)
            for dist_id, params in next_distributions:
                assert isinstance(dist_id, str), "dist_id should be string"
                assert isinstance(params, dict), "params should be dict"
                # Check for logit keys
                for g in group_names:
                    assert f'shape_logit_{g}' in params, f"Missing shape_logit_{g}"
                # Check for param keys
                for g in group_names:
                    for param_name in param_bounds.get(g, {}).keys():
                        optuna_key = f'{g}__{param_name}'
                        assert optuna_key in params, f"Missing {optuna_key}"

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
        Each distribution generates samples that are tagged with distribution_id.
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
        generator = VoidGenerator(Path(mini_config['base_image_dir']))
        embedder = DinoV2Embedder(model_name=mini_config['dino_model'])

        # Generate real distribution
        real_params = [
            {'void_shape': 'circle', 'void_count': 3, 'base_size': 10.0,
             'center_x': 0.5, 'center_y': 0.5, 'position_spread': 0.1, 'rotation': 0.0}
        ]
        real_images, _ = generator.generate_batch(real_params, replications=1, seed_offset=0)
        real_embeddings_full = embedder.embed_batch(real_images)

        pca = PCAProjector(n_components=mini_config['pca_embedding_dim'])
        real_embeddings = pca.fit_transform(real_embeddings_full)

        # Create initial joint distributions
        n_distributions = mini_config['iteration_batch_size']
        initial_distributions = self._create_joint_distributions(
            n_distributions, group_names, param_bounds
        )
        initial_metrics_list = self._create_metrics_list(
            n_distributions, {'mmd_rbf': 0.5, 'mean_nn_distance': 1.0}
        )

        # Get next distributions from optimizer
        next_distributions = optimizer.suggest_next_distributions(
            current_distributions=initial_distributions,
            metrics_list=initial_metrics_list,
            config=mini_config
        )

        # Sample with multiple replications per distribution
        replications_per_dist = 3
        next_params = []
        for dist_idx, (dist_id, joint_params) in enumerate(next_distributions):
            samples = self._joint_params_to_concrete_samples(
                joint_params,
                group_names,
                n_samples=replications_per_dist,
                seed=dist_idx * 1000
            )
            for p in samples:
                p['distribution_id'] = dist_idx
            next_params.extend(samples)

        # Verify distribution of samples
        print("\n[Test] Verifying samples per distribution:")
        for dist_idx in range(len(next_distributions)):
            dist_params = [p for p in next_params if p['distribution_id'] == dist_idx]
            shapes = [p['void_shape'] for p in dist_params]
            print(f"  Distribution {dist_idx}: {len(dist_params)} samples, shapes={shapes}")
            assert len(dist_params) == replications_per_dist

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

    def test_joint_distribution_format(self, mini_config, temp_test_dir, param_bounds_and_groups):
        """
        Test that optimizer output follows joint distribution format.

        Each distribution should contain:
        - shape_logit_* keys for all groups
        - {group}__{param} keys for all params
        """
        param_bounds, group_names = param_bounds_and_groups

        # Initialize optimizer
        experiment_dir = Path(mini_config['experiment_dir'])
        experiment_dir.mkdir(parents=True, exist_ok=True)

        optimizer = OptunaOptimizer(
            experiment_dir,
            mini_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        # Create initial distributions and get suggestions
        n_distributions = mini_config['iteration_batch_size']
        initial_distributions = self._create_joint_distributions(
            n_distributions, group_names, param_bounds
        )
        initial_metrics_list = self._create_metrics_list(
            n_distributions, {'mmd_rbf': 0.5, 'mean_nn_distance': 1.0}
        )

        next_distributions = optimizer.suggest_next_distributions(
            current_distributions=initial_distributions,
            metrics_list=initial_metrics_list,
            config=mini_config
        )

        # Verify format
        print("\n[Test] Verifying joint distribution format...")

        for dist_id, params in next_distributions:
            print(f"\n  Distribution: {dist_id}")

            # Check logit keys exist
            logit_keys = [f'shape_logit_{g}' for g in group_names]
            for key in logit_keys:
                assert key in params, f"Missing {key}"

            # Convert logits to probabilities
            probs = OptunaOptimizer.logits_to_probabilities(params)
            total_prob = sum(probs.values())
            assert abs(total_prob - 1.0) < 1e-6, f"Probabilities should sum to 1, got {total_prob}"
            print(f"    Probabilities: {probs}")

            # Check all group params exist
            for group_name in group_names:
                group_bounds = param_bounds.get(group_name, {})
                for param_name in group_bounds.keys():
                    optuna_key = f'{group_name}__{param_name}'
                    assert optuna_key in params, f"Missing {optuna_key}"

            # Count total params
            n_logits = len(group_names)
            n_params = sum(len(param_bounds.get(g, {})) for g in group_names)
            expected_total = n_logits + n_params
            assert len(params) == expected_total, \
                f"Expected {expected_total} params, got {len(params)}"

            print(f"    Total params: {len(params)} ({n_logits} logits + {n_params} shape params)")

        print("\n[Test] ✓ Joint distribution format test passed!")
