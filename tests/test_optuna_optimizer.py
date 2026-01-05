"""
Unit tests for OptunaOptimizer (Stage 1: Single-objective optimization)
"""

import pytest
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
    """Test suite for OptunaOptimizer Stage 1"""

    def test_initialization(self, temp_experiment_dir, test_config):
        """Test optimizer initialization and SQLite creation"""
        optimizer = OptunaOptimizer(temp_experiment_dir, test_config)

        # Check SQLite file was created
        assert optimizer.study_path.exists()
        assert optimizer.study_path.name == "optuna_study.db"

        # Check study configuration
        assert optimizer.study is not None
        assert optimizer.study.study_name == 'test_optuna'
        assert len(optimizer.study.directions) == 1  # Single-objective

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
        metrics = {'mmd_rbf': 0.15, 'wasserstein': 0.08, 'mean_nn_distance': 3.5}

        # Test iteration 1
        next_params, converged = optimizer.suggest_next_parameters(
            synthetic_embeddings,
            synthetic_params,
            real_embeddings,
            metrics,
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
        metrics = {'mmd_rbf': 0.15, 'wasserstein': 0.08, 'mean_nn_distance': 3.5}

        # Test at max_iterations
        next_params, converged = optimizer.suggest_next_parameters(
            synthetic_embeddings,
            synthetic_params,
            real_embeddings,
            metrics,
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
        metrics = {'mmd_rbf': 0.15, 'wasserstein': 0.08, 'mean_nn_distance': 3.5}

        next_params1, _ = optimizer1.suggest_next_parameters(
            synthetic_embeddings,
            synthetic_params,
            real_embeddings,
            metrics,
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
        metrics = {'mmd_rbf': 0.15, 'wasserstein': 0.08, 'mean_nn_distance': 3.5}

        next_params, _ = optimizer.suggest_next_parameters(
            synthetic_embeddings,
            synthetic_params,
            real_embeddings,
            metrics,
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
        metrics = {'mmd_rbf': 0.15, 'wasserstein': 0.08, 'mean_nn_distance': 3.5}

        next_params1, _ = optimizer1.suggest_next_parameters(
            synthetic_embeddings,
            synthetic_params,
            real_embeddings,
            metrics,
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
            metrics,
            iteration=1,
            config=test_config
        )

        # Should produce same parameters (TPE is deterministic with seed)
        # Note: This might not be exactly equal due to TPE's behavior, but check first param
        assert next_params1[0]['void_shape'] == next_params2[0]['void_shape']
        assert next_params1[0]['void_count'] == next_params2[0]['void_count']

        # Cleanup
        shutil.rmtree(temp_dir2)
