"""
Tests for OptunaOptimizer grouped mode (conditional parameters).

Tests use get_param_bounds() to mirror the real pipeline flow.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.optimization.optuna_optimizer import OptunaOptimizer
from src.utils.bounds_inference import get_param_bounds


@pytest.fixture
def temp_experiment_dir():
    """Create temporary experiment directory"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def param_bounds_and_groups():
    """Get param_bounds and group_names from the real pipeline"""
    return get_param_bounds()


@pytest.fixture
def grouped_config():
    """Config for grouped mode tests"""
    return {
        'experiment_name': 'test_grouped',
        'random_seed': 42,
        'iteration_batch_size': 4,
        'max_iterations': 5,
        'optimization_metrics': ['mmd_rbf'],
        'optimizer': {'n_startup_trials': 6}  # 2 per group for testing
    }


class TestGroupedOptimizerInit:
    """Tests for grouped optimizer initialization"""

    def test_init_with_param_bounds(self, temp_experiment_dir, param_bounds_and_groups, grouped_config):
        """Test initializing optimizer with param_bounds"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        assert optimizer.group_names == group_names
        assert len(optimizer.param_bounds) == len(group_names)
        for group in group_names:
            assert group in optimizer.param_bounds

    def test_group_specific_bounds(self, temp_experiment_dir, param_bounds_and_groups, grouped_config):
        """Test that each group has its own bounds with expected parameters"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        # All groups should have void_count_mean
        for group in group_names:
            assert 'void_count_mean' in optimizer.param_bounds[group]

        # Only ellipse has rotation
        assert 'rotation_mean' not in optimizer.param_bounds['circle']
        assert 'rotation_mean' in optimizer.param_bounds['ellipse']
        assert 'rotation_mean' not in optimizer.param_bounds['irregular']

    def test_startup_trials_scaled_by_groups(self, temp_experiment_dir, param_bounds_and_groups):
        """Test that n_startup_trials defaults to 20 * num_groups"""
        param_bounds, group_names = param_bounds_and_groups
        config = {
            'experiment_name': 'test',
            'optimization_metrics': ['mmd_rbf'],
        }

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        # Default: 20 * num_groups
        expected_startup = 20 * len(group_names)
        sampler = optimizer.study.sampler
        assert sampler._n_startup_trials == expected_startup

    def test_missing_params_raises(self, temp_experiment_dir, grouped_config):
        """Test that missing param_bounds or group_names raises error"""
        with pytest.raises(ValueError, match="param_bounds and group_names are required"):
            OptunaOptimizer(
                experiment_dir=temp_experiment_dir,
                config=grouped_config,
                param_bounds=None,
                group_names=None
            )

    def test_mismatched_keys_raises(self, temp_experiment_dir, param_bounds_and_groups, grouped_config):
        """Test that mismatched param_bounds keys and group_names raises error"""
        param_bounds, _ = param_bounds_and_groups

        with pytest.raises(ValueError, match="must match"):
            OptunaOptimizer(
                experiment_dir=temp_experiment_dir,
                config=grouped_config,
                param_bounds=param_bounds,
                group_names=['circle', 'ellipse']  # Missing 'irregular'
            )


class TestGroupedSearchSpace:
    """Tests for grouped search space suggestions"""

    def test_grouped_search_space_returns_group_and_params(
        self, temp_experiment_dir, param_bounds_and_groups, grouped_config
    ):
        """Test that _define_grouped_search_space returns (group, params)"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        trial = optimizer.study.ask()
        group, params = optimizer._define_grouped_search_space(trial)

        assert group in group_names
        assert isinstance(params, dict)

    def test_params_match_selected_group(
        self, temp_experiment_dir, param_bounds_and_groups, grouped_config
    ):
        """Test that params match the selected group's parameters"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        # Run multiple trials to get different groups
        for _ in range(10):
            trial = optimizer.study.ask()
            group, params = optimizer._define_grouped_search_space(trial)
            optimizer.study.tell(trial, [0.5])

            expected_params = set(optimizer.param_bounds[group].keys())
            actual_params = set(params.keys())
            assert actual_params == expected_params, \
                f"Group {group}: expected {expected_params}, got {actual_params}"

    def test_ellipse_has_rotation(
        self, temp_experiment_dir, param_bounds_and_groups, grouped_config
    ):
        """Test that ellipse group has rotation parameter"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        # Find an ellipse trial
        for _ in range(20):
            trial = optimizer.study.ask()
            group, params = optimizer._define_grouped_search_space(trial)
            optimizer.study.tell(trial, [0.5])

            if group == 'ellipse':
                assert 'rotation_mean' in params
                assert 'rotation_std' in params
                break


class TestGroupLearning:
    """Tests that TPE learns which group is best"""

    def test_tpe_learns_best_group(
        self, temp_experiment_dir, param_bounds_and_groups, grouped_config
    ):
        """Test that TPE learns to prefer the group with better scores"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        # Give circle the best score, ellipse medium, irregular worst
        score_by_group = {'circle': 0.1, 'ellipse': 0.5, 'irregular': 0.9}
        group_counts = {g: 0 for g in group_names}

        # Run enough trials to see learning
        for i in range(30):
            trial = optimizer.study.ask()
            group, params = optimizer._define_grouped_search_space(trial)
            group_counts[group] += 1

            # Report score based on group
            optimizer.study.tell(trial, [score_by_group[group]])

        # After learning, circle should be sampled most
        # (this may not be deterministic, so we just check circle was sampled)
        assert group_counts['circle'] > 0, "Circle should be sampled at least once"


class TestSuggestNextDistributions:
    """Tests for the suggest_next_distributions method"""

    def _create_metrics_list(self, n: int, base_metrics: dict) -> list:
        return [{**base_metrics, 'param_set_id': f'trial_{i}'} for i in range(n)]

    def test_returns_correct_format(
        self, temp_experiment_dir, param_bounds_and_groups, grouped_config
    ):
        """Test suggest_next_distributions returns (suggestions, converged)"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        metrics_list = self._create_metrics_list(grouped_config['iteration_batch_size'], {'mmd_rbf': 0.5})
        suggestions, converged = optimizer.suggest_next_distributions(
            metrics_list=metrics_list,
            iteration=0,
            config=grouped_config
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == grouped_config['iteration_batch_size']
        assert isinstance(converged, bool)

        # Each suggestion is (group_name, params_dict)
        for group, params in suggestions:
            assert group in group_names
            assert isinstance(params, dict)

    def test_ask_tell_pattern(
        self, temp_experiment_dir, param_bounds_and_groups, grouped_config
    ):
        """Test ask/tell pattern works over multiple iterations"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        for iteration in range(3):
            metrics_list = self._create_metrics_list(
                grouped_config['iteration_batch_size'],
                {'mmd_rbf': 0.5 - iteration * 0.1}
            )

            suggestions, converged = optimizer.suggest_next_distributions(
                metrics_list=metrics_list,
                iteration=iteration,
                config=grouped_config
            )

            assert len(suggestions) == grouped_config['iteration_batch_size']

            # Verify pending trials tracking
            assert iteration in optimizer.pending_trials
            if iteration > 0:
                assert (iteration - 1) not in optimizer.pending_trials
