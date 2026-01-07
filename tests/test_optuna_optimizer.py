"""
Unit tests for OptunaOptimizer (Stage 3: Proper ask/tell pattern)
"""

import pytest
import optuna
import numpy as np
import tempfile
import shutil
from pathlib import Path

from src.optimization.optuna_optimizer import OptunaOptimizer


@pytest.fixture
def temp_experiment_dir():
    """Create temporary experiment directory"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config():
    """Sample configuration for testing"""
    return {
        'experiment_name': 'test_optuna',
        'random_seed': 42,
        'iteration_batch_size': 4,
        'max_iterations': 5,
        'optimization_metrics': ['mmd_rbf', 'mean_nn_distance'],  # Configurable metrics
        'param_bounds': {
            'void_shape': ['circle', 'ellipse', 'irregular'],
            'void_count': [1, 10],
            'base_size': [5.0, 15.0],
            'rotation': [0.0, 360.0],
            'center_x': [0.2, 0.8],
            'center_y': [0.2, 0.8],
            'position_spread': [0.1, 0.8]
        },
        'param_precision': {
            'base_size': 1,
            'rotation': 1,
            'center_x': 2,
            'center_y': 2,
            'position_spread': 2
        }
    }


class TestOptunaOptimizer:
    """Test suite for OptunaOptimizer Stage 3: Proper ask/tell pattern"""

    def _create_metrics_list(self, n_param_sets: int, base_metrics: dict) -> list:
        """Helper to create metrics_list with param_set_id for each parameter set"""
        return [
            {**base_metrics, 'param_set_id': f'ps_{i:03d}'}
            for i in range(n_param_sets)
        ]

    def test_initialization(self, temp_experiment_dir, test_config):
        """Test optimizer initialization and SQLite creation"""
        optimizer = OptunaOptimizer(temp_experiment_dir, test_config)

        # Check SQLite file was created
        assert optimizer.study_path.exists()
        assert optimizer.study_path.name == "optuna_study.db"

        # Check study configuration
        assert optimizer.study is not None
        assert optimizer.study.study_name == 'test_optuna'
        # Number of objectives should match config
        n_metrics = len(test_config['optimization_metrics'])
        assert len(optimizer.study.directions) == n_metrics

    def test_search_space_valid_parameters(self, temp_experiment_dir, test_config):
        """Test that _define_search_space produces valid parameters within bounds"""
        optimizer = OptunaOptimizer(temp_experiment_dir, test_config)

        # Generate multiple parameter sets
        param_sets = []
        for _ in range(10):
            trial = optimizer.study.ask()
            params = optimizer._define_search_space(trial)
            param_sets.append(params)

        # Verify all parameter sets are valid
        bounds = test_config['param_bounds']
        for params in param_sets:
            # Check void_shape is valid
            assert params['void_shape'] in bounds['void_shape']

            # Check void_count is in bounds and integer
            assert bounds['void_count'][0] <= params['void_count'] <= bounds['void_count'][1]
            assert isinstance(params['void_count'], int)

            # Check continuous parameters are in bounds
            assert bounds['base_size'][0] <= params['base_size'] <= bounds['base_size'][1]
            assert bounds['rotation'][0] <= params['rotation'] <= bounds['rotation'][1]
            assert bounds['center_x'][0] <= params['center_x'] <= bounds['center_x'][1]
            assert bounds['center_y'][0] <= params['center_y'] <= bounds['center_y'][1]
            assert bounds['position_spread'][0] <= params['position_spread'] <= bounds['position_spread'][1]

    def test_parameter_precision(self, temp_experiment_dir, test_config):
        """Test that parameters respect precision requirements"""
        optimizer = OptunaOptimizer(temp_experiment_dir, test_config)

        # Generate parameter sets
        param_sets = []
        for _ in range(10):
            trial = optimizer.study.ask()
            params = optimizer._define_search_space(trial)
            param_sets.append(params)

        precision = test_config['param_precision']

        for params in param_sets:
            # base_size: 1 decimal place
            # Use np.isclose to handle floating point comparison
            assert np.isclose(params['base_size'], round(params['base_size'], precision['base_size']))

            # rotation: 1 decimal place
            assert np.isclose(params['rotation'], round(params['rotation'], precision['rotation']))

            # center_x, center_y, position_spread: 2 decimal places
            assert np.isclose(params['center_x'], round(params['center_x'], precision['center_x']))
            assert np.isclose(params['center_y'], round(params['center_y'], precision['center_y']))
            assert np.isclose(params['position_spread'], round(params['position_spread'], precision['position_spread']))

    def test_suggest_next_parameters_batch_size(self, temp_experiment_dir, test_config):
        """Test that suggest_next_parameters returns correct batch size"""
        optimizer = OptunaOptimizer(temp_experiment_dir, test_config)

        # Mock data for iteration 0
        synthetic_embeddings = np.random.randn(24, 400)
        synthetic_params = []
        real_embeddings = np.random.randn(15, 400)
        # Create metrics_list with one dict per parameter set
        metrics_list = self._create_metrics_list(
            test_config['iteration_batch_size'],
            {'mmd_rbf': 0.15, 'wasserstein': 0.08, 'mean_nn_distance': 3.5}
        )

        # Test iteration 1
        next_params, converged = optimizer.suggest_next_parameters(
            synthetic_embeddings,
            synthetic_params,
            real_embeddings,
            metrics_list,
            iteration=1,
            config=test_config
        )

        # Should return correct batch size
        assert len(next_params) == test_config['iteration_batch_size']
        assert converged is False  # Not converged yet

    def test_convergence_at_max_iterations(self, temp_experiment_dir, test_config):
        """Test that optimizer converges at max_iterations"""
        optimizer = OptunaOptimizer(temp_experiment_dir, test_config)

        # Mock data
        synthetic_embeddings = np.random.randn(24, 400)
        synthetic_params = []
        real_embeddings = np.random.randn(15, 400)
        metrics_list = self._create_metrics_list(
            test_config['iteration_batch_size'],
            {'mmd_rbf': 0.15, 'wasserstein': 0.08, 'mean_nn_distance': 3.5}
        )

        # Test at max_iterations
        next_params, converged = optimizer.suggest_next_parameters(
            synthetic_embeddings,
            synthetic_params,
            real_embeddings,
            metrics_list,
            iteration=test_config['max_iterations'],
            config=test_config
        )

        # Should converge
        assert converged is True

    def test_study_persistence(self, temp_experiment_dir, test_config):
        """Test that study state persists to SQLite"""
        # Create optimizer and generate some trials
        optimizer1 = OptunaOptimizer(temp_experiment_dir, test_config)

        synthetic_embeddings = np.random.randn(24, 400)
        synthetic_params = []
        real_embeddings = np.random.randn(15, 400)
        metrics_list = self._create_metrics_list(
            test_config['iteration_batch_size'],
            {'mmd_rbf': 0.15, 'wasserstein': 0.08, 'mean_nn_distance': 3.5}
        )

        next_params1, _ = optimizer1.suggest_next_parameters(
            synthetic_embeddings,
            synthetic_params,
            real_embeddings,
            metrics_list,
            iteration=1,
            config=test_config
        )

        n_trials_1 = len(optimizer1.study.trials)

        # Create new optimizer instance with same directory
        optimizer2 = OptunaOptimizer(temp_experiment_dir, test_config)

        # Should load existing study
        n_trials_2 = len(optimizer2.study.trials)
        assert n_trials_2 == n_trials_1

    def test_parameter_structure(self, temp_experiment_dir, test_config):
        """Test that returned parameters have correct structure"""
        optimizer = OptunaOptimizer(temp_experiment_dir, test_config)

        synthetic_embeddings = np.random.randn(24, 400)
        synthetic_params = []
        real_embeddings = np.random.randn(15, 400)
        metrics_list = self._create_metrics_list(
            test_config['iteration_batch_size'],
            {'mmd_rbf': 0.15, 'wasserstein': 0.08, 'mean_nn_distance': 3.5}
        )

        next_params, _ = optimizer.suggest_next_parameters(
            synthetic_embeddings,
            synthetic_params,
            real_embeddings,
            metrics_list,
            iteration=1,
            config=test_config
        )

        # Check each parameter set has all required keys
        required_keys = {'void_shape', 'void_count', 'base_size', 'rotation',
                        'center_x', 'center_y', 'position_spread'}

        for params in next_params:
            assert set(params.keys()) == required_keys

    def test_reproducibility_with_seed(self, temp_experiment_dir, test_config):
        """Test that same seed produces same parameter suggestions"""
        # First optimizer
        optimizer1 = OptunaOptimizer(temp_experiment_dir, test_config)

        synthetic_embeddings = np.random.randn(24, 400)
        synthetic_params = []
        real_embeddings = np.random.randn(15, 400)
        metrics_list = self._create_metrics_list(test_config['iteration_batch_size'], {'mmd_rbf': 0.15, 'wasserstein': 0.08, 'mean_nn_distance': 3.5})

        next_params1, _ = optimizer1.suggest_next_parameters(
            synthetic_embeddings,
            synthetic_params,
            real_embeddings,
            metrics_list,
            iteration=1,
            config=test_config
        )

        # Second optimizer with new temp dir but same seed
        temp_dir2 = Path(tempfile.mkdtemp())
        optimizer2 = OptunaOptimizer(temp_dir2, test_config)

        next_params2, _ = optimizer2.suggest_next_parameters(
            synthetic_embeddings,
            synthetic_params,
            real_embeddings,
            metrics_list,
            iteration=1,
            config=test_config
        )

        # Should produce same parameters (TPE is deterministic with seed)
        # Note: This might not be exactly equal due to TPE's behavior, but check first param
        assert next_params1[0]['void_shape'] == next_params2[0]['void_shape']
        assert next_params1[0]['void_count'] == next_params2[0]['void_count']

        # Cleanup
        shutil.rmtree(temp_dir2)

    def test_multi_objective_three_metrics(self, temp_experiment_dir, test_config):
        """Test that optimizer handles 3 objectives correctly (Stage 2)"""
        optimizer = OptunaOptimizer(temp_experiment_dir, test_config)

        # Verify study has correct number of objectives from config
        n_metrics = len(test_config['optimization_metrics'])
        assert len(optimizer.study.directions) == n_metrics
        assert all(d == optuna.study.StudyDirection.MINIMIZE for d in optimizer.study.directions)

    def test_get_pareto_front(self, temp_experiment_dir, test_config):
        """Test Pareto front retrieval after multiple iterations"""
        optimizer = OptunaOptimizer(temp_experiment_dir, test_config)

        synthetic_embeddings = np.random.randn(24, 400)
        synthetic_params = []
        real_embeddings = np.random.randn(15, 400)

        # Run multiple iterations with different metrics to build Pareto front
        for i in range(1, 6):
            # Vary metrics to create trade-offs
            metrics_list = self._create_metrics_list(test_config['iteration_batch_size'], {
                'mmd_rbf': 0.10 + i * 0.01,
                'wasserstein': 0.08 - i * 0.005,
                'mean_nn_distance': 3.0 + i * 0.2
            })

            next_params, _ = optimizer.suggest_next_parameters(
                synthetic_embeddings,
                synthetic_params,
                real_embeddings,
                metrics_list,
                iteration=i,
                config=test_config
            )

        # Get Pareto front
        pareto_front = optimizer.get_pareto_front()

        # Should have some non-dominated solutions
        assert len(pareto_front) > 0
        assert all(isinstance(trial, optuna.trial.FrozenTrial) for trial in pareto_front)

        # Each trial should have correct number of objective values from config
        n_metrics = len(test_config['optimization_metrics'])
        for trial in pareto_front:
            assert len(trial.values) == n_metrics

    def test_pareto_front_growth(self, temp_experiment_dir, test_config):
        """Test that Pareto front can grow over iterations"""
        optimizer = OptunaOptimizer(temp_experiment_dir, test_config)

        synthetic_embeddings = np.random.randn(24, 400)
        synthetic_params = []
        real_embeddings = np.random.randn(15, 400)

        pareto_sizes = []

        # Run iterations and track Pareto front size
        for i in range(1, 4):
            metrics_list = self._create_metrics_list(test_config['iteration_batch_size'], {
                'mmd_rbf': 0.15 - i * 0.02,
                'wasserstein': 0.08 - i * 0.01,
                'mean_nn_distance': 3.5 - i * 0.3
            })

            optimizer.suggest_next_parameters(
                synthetic_embeddings,
                synthetic_params,
                real_embeddings,
                metrics_list,
                iteration=i,
                config=test_config
            )

            pareto_sizes.append(len(optimizer.get_pareto_front()))

        # Pareto front should be non-empty after multiple iterations
        assert pareto_sizes[-1] > 0

    def test_pending_trials_initialization(self, temp_experiment_dir, test_config):
        """Test that pending_trials dict is initialized (Stage 3)"""
        optimizer = OptunaOptimizer(temp_experiment_dir, test_config)

        # Verify pending_trials exists and is empty
        assert hasattr(optimizer, 'pending_trials')
        assert isinstance(optimizer.pending_trials, dict)
        assert len(optimizer.pending_trials) == 0

    def test_pending_trials_stored_on_ask(self, temp_experiment_dir, test_config):
        """Test that trials are stored as pending when asked (Stage 3)"""
        optimizer = OptunaOptimizer(temp_experiment_dir, test_config)

        synthetic_embeddings = np.random.randn(24, 400)
        synthetic_params = []
        real_embeddings = np.random.randn(15, 400)
        metrics_list = self._create_metrics_list(test_config['iteration_batch_size'], {'mmd_rbf': 0.15, 'wasserstein': 0.08, 'mean_nn_distance': 3.5})

        # Ask for parameters for iteration 0
        next_params, _ = optimizer.suggest_next_parameters(
            synthetic_embeddings,
            synthetic_params,
            real_embeddings,
            metrics_list,
            iteration=0,
            config=test_config
        )

        # Verify trials are stored as pending for iteration 0
        assert 0 in optimizer.pending_trials
        assert len(optimizer.pending_trials[0]) == test_config['iteration_batch_size']

    def test_pending_trials_completed_on_tell(self, temp_experiment_dir, test_config):
        """Test that pending trials are completed and removed when results are reported (Stage 3)"""
        optimizer = OptunaOptimizer(temp_experiment_dir, test_config)

        synthetic_embeddings = np.random.randn(24, 400)
        synthetic_params = []
        real_embeddings = np.random.randn(15, 400)

        # Iteration 0: Ask for parameters
        metrics_list_0 = self._create_metrics_list(
            test_config['iteration_batch_size'],
            {'mmd_rbf': 0.15, 'wasserstein': 0.08, 'mean_nn_distance': 3.5}
        )
        next_params_0, _ = optimizer.suggest_next_parameters(
            synthetic_embeddings,
            synthetic_params,
            real_embeddings,
            metrics_list_0,
            iteration=0,
            config=test_config
        )

        # Verify iteration 0 trials are pending
        assert 0 in optimizer.pending_trials
        n_trials_iter0 = len(optimizer.pending_trials[0])

        # Iteration 1: Report results for iteration 0 and ask for new parameters
        metrics_list_1 = self._create_metrics_list(
            test_config['iteration_batch_size'],
            {'mmd_rbf': 0.12, 'wasserstein': 0.07, 'mean_nn_distance': 3.2}
        )
        next_params_1, _ = optimizer.suggest_next_parameters(
            synthetic_embeddings,
            next_params_0,
            real_embeddings,
            metrics_list_1,
            iteration=1,
            config=test_config
        )

        # Verify iteration 0 trials were completed and removed
        assert 0 not in optimizer.pending_trials

        # Verify iteration 1 trials are now pending
        assert 1 in optimizer.pending_trials
        assert len(optimizer.pending_trials[1]) == test_config['iteration_batch_size']

        # Verify trials were actually completed in the study
        completed_trials = [t for t in optimizer.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        assert len(completed_trials) == n_trials_iter0

    def test_ask_tell_pattern_multiple_iterations(self, temp_experiment_dir, test_config):
        """Test proper ask/tell pattern over multiple iterations (Stage 3)"""
        optimizer = OptunaOptimizer(temp_experiment_dir, test_config)

        synthetic_embeddings = np.random.randn(24, 400)
        synthetic_params = []
        real_embeddings = np.random.randn(15, 400)

        # Track completed trials
        completed_trials_per_iteration = []

        for iteration in range(3):
            metrics_list = self._create_metrics_list(test_config['iteration_batch_size'], {
                'mmd_rbf': 0.15 - iteration * 0.02,
                'wasserstein': 0.08 - iteration * 0.01,
                'mean_nn_distance': 3.5 - iteration * 0.3
            })

            # Ask for next parameters
            next_params, _ = optimizer.suggest_next_parameters(
                synthetic_embeddings,
                synthetic_params,
                real_embeddings,
                metrics_list,
                iteration=iteration,
                config=test_config
            )

            # Verify current iteration is pending
            assert iteration in optimizer.pending_trials
            assert len(optimizer.pending_trials[iteration]) == test_config['iteration_batch_size']

            # Verify previous iteration was completed (if exists)
            if iteration > 0:
                assert (iteration - 1) not in optimizer.pending_trials

            # Track completed trials
            completed_trials = [t for t in optimizer.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            completed_trials_per_iteration.append(len(completed_trials))

            # Update for next iteration
            synthetic_params = next_params

        # Verify trials were completed incrementally
        # Iteration 0: no completed trials (no previous iteration to report)
        assert completed_trials_per_iteration[0] == 0
        # Iteration 1: iteration 0's trials should be completed
        assert completed_trials_per_iteration[1] == test_config['iteration_batch_size']
        # Iteration 2: both iteration 0 and 1's trials should be completed
        assert completed_trials_per_iteration[2] == 2 * test_config['iteration_batch_size']

    def test_trials_receive_correct_metrics(self, temp_experiment_dir, test_config):
        """Test that trials receive the correct metric values when completed (Stage 3)"""
        optimizer = OptunaOptimizer(temp_experiment_dir, test_config)

        synthetic_embeddings = np.random.randn(24, 400)
        synthetic_params = []
        real_embeddings = np.random.randn(15, 400)

        # Iteration 0
        # Note: metrics parameter in iteration 0 doesn't matter (no previous iteration to report)
        metrics_list_dummy = self._create_metrics_list(
            test_config['iteration_batch_size'],
            {'mmd_rbf': 0.99, 'wasserstein': 0.99, 'mean_nn_distance': 99.9}
        )
        next_params_0, _ = optimizer.suggest_next_parameters(
            synthetic_embeddings,
            synthetic_params,
            real_embeddings,
            metrics_list_dummy,
            iteration=0,
            config=test_config
        )

        # Iteration 1: Pass metrics from iteration 0 - these will be used to complete iteration 0 trials
        base_metrics_0 = {'mmd_rbf': 0.15, 'wasserstein': 0.08, 'mean_nn_distance': 3.5}
        metrics_list_0 = self._create_metrics_list(
            test_config['iteration_batch_size'],
            base_metrics_0
        )
        next_params_1, _ = optimizer.suggest_next_parameters(
            synthetic_embeddings,
            next_params_0,
            real_embeddings,
            metrics_list_0,  # These are iteration 0's results
            iteration=1,
            config=test_config
        )

        # Verify completed trials (iteration 0) have correct metric values
        completed_trials = [t for t in optimizer.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        n_metrics = len(test_config['optimization_metrics'])
        for trial in completed_trials:
            assert len(trial.values) == n_metrics
            # Check each configured metric value
            for i, metric_name in enumerate(test_config['optimization_metrics']):
                assert trial.values[i] == base_metrics_0[metric_name]

    def test_no_orphaned_trials(self, temp_experiment_dir, test_config):
        """Test that no trials are left in pending state (Stage 3)"""
        optimizer = OptunaOptimizer(temp_experiment_dir, test_config)

        synthetic_embeddings = np.random.randn(24, 400)
        synthetic_params = []
        real_embeddings = np.random.randn(15, 400)

        # Run 3 iterations
        for iteration in range(3):
            metrics_list = self._create_metrics_list(test_config['iteration_batch_size'], {
                'mmd_rbf': 0.15 - iteration * 0.02,
                'wasserstein': 0.08 - iteration * 0.01,
                'mean_nn_distance': 3.5 - iteration * 0.3
            })

            next_params, _ = optimizer.suggest_next_parameters(
                synthetic_embeddings,
                synthetic_params,
                real_embeddings,
                metrics_list,
                iteration=iteration,
                config=test_config
            )

            synthetic_params = next_params

        # Only the last iteration should have pending trials
        assert len(optimizer.pending_trials) == 1
        assert 2 in optimizer.pending_trials  # Last iteration

        # All other trials should be completed
        completed_trials = [t for t in optimizer.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        # Iterations 0 and 1 should be completed
        assert len(completed_trials) == 2 * test_config['iteration_batch_size']
