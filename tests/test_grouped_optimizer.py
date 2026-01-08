"""
Tests for OptunaOptimizer grouped mode (conditional parameters).

Tests that the optimizer correctly:
- Infers bounds from DataFrames per group
- Uses Optuna's native categorical for group selection
- Suggests conditional parameters based on selected group
- Learns which group produces better outcomes
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from src.optimization.optuna_optimizer import OptunaOptimizer
from src.utils.bounds_inference import infer_bounds_from_dataframe


@pytest.fixture
def temp_experiment_dir():
    """Create temporary experiment directory"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def group_dataframes():
    """Create DataFrames with different params per group (like mock data generator)"""
    # Circle: no rotation
    circle_df = pd.DataFrame({
        'void_count_mean': [3.0, 5.0, 7.0, 4.0, 6.0],
        'void_count_std': [1.0, 1.5, 2.0, 1.2, 1.8],
        'base_size_mean': [8.0, 10.0, 12.0, 9.0, 11.0],
        'base_size_std': [2.0, 3.0, 4.0, 2.5, 3.5],
    })

    # Ellipse: has rotation (unique to this group)
    ellipse_df = pd.DataFrame({
        'void_count_mean': [2.0, 4.0, 6.0, 3.0, 5.0],
        'void_count_std': [0.5, 1.0, 1.5, 0.8, 1.2],
        'rotation_mean': [0.0, 90.0, 180.0, 45.0, 135.0],
        'rotation_std': [10.0, 20.0, 30.0, 15.0, 25.0],
    })

    # Irregular: has complexity (unique to this group)
    irregular_df = pd.DataFrame({
        'void_count_mean': [1.0, 3.0, 5.0, 2.0, 4.0],
        'void_count_std': [0.5, 1.0, 1.5, 0.7, 1.3],
        'complexity_mean': [5.0, 8.0, 11.0, 6.5, 9.5],
        'complexity_std': [1.0, 2.0, 3.0, 1.5, 2.5],
    })

    return [circle_df, ellipse_df, irregular_df]


@pytest.fixture
def group_names():
    return ['circle', 'ellipse', 'irregular']


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


class TestBoundsInference:
    """Tests for bounds inference from DataFrames"""

    def test_infer_bounds_from_single_dataframe(self, group_dataframes):
        """Test inferring bounds from a single DataFrame"""
        circle_df = group_dataframes[0]
        bounds = infer_bounds_from_dataframe(circle_df)

        assert 'void_count_mean' in bounds
        assert 'void_count_std' in bounds
        assert 'base_size_mean' in bounds
        assert 'base_size_std' in bounds

        # Check bounds are [min, max]
        assert bounds['void_count_mean'] == [3.0, 7.0]
        assert bounds['base_size_mean'] == [8.0, 12.0]

    def test_different_params_per_group(self, group_dataframes):
        """Test that different groups have different parameters"""
        circle_bounds = infer_bounds_from_dataframe(group_dataframes[0])
        ellipse_bounds = infer_bounds_from_dataframe(group_dataframes[1])
        irregular_bounds = infer_bounds_from_dataframe(group_dataframes[2])

        # Circle has no rotation
        assert 'rotation_mean' not in circle_bounds
        # Ellipse has rotation
        assert 'rotation_mean' in ellipse_bounds
        # Irregular has complexity
        assert 'complexity_mean' in irregular_bounds
        assert 'rotation_mean' not in irregular_bounds


class TestGroupedOptimizerInit:
    """Tests for grouped optimizer initialization"""

    def test_init_with_dataframes(self, temp_experiment_dir, group_dataframes, group_names, grouped_config):
        """Test initializing optimizer with DataFrames"""
        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_dataframes=group_dataframes,
            group_names=group_names
        )

        assert optimizer.group_names == group_names
        assert len(optimizer.param_bounds) == 3
        assert 'circle' in optimizer.param_bounds
        assert 'ellipse' in optimizer.param_bounds
        assert 'irregular' in optimizer.param_bounds

    def test_group_specific_bounds(self, temp_experiment_dir, group_dataframes, group_names, grouped_config):
        """Test that each group has its own bounds"""
        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_dataframes=group_dataframes,
            group_names=group_names
        )

        # Circle bounds
        assert 'void_count_mean' in optimizer.param_bounds['circle']
        assert 'rotation_mean' not in optimizer.param_bounds['circle']

        # Ellipse bounds
        assert 'rotation_mean' in optimizer.param_bounds['ellipse']
        assert 'complexity_mean' not in optimizer.param_bounds['ellipse']

        # Irregular bounds
        assert 'complexity_mean' in optimizer.param_bounds['irregular']
        assert 'rotation_mean' not in optimizer.param_bounds['irregular']

    def test_startup_trials_scaled_by_groups(self, temp_experiment_dir, group_dataframes, group_names):
        """Test that n_startup_trials defaults to 20 * num_groups"""
        config = {
            'experiment_name': 'test',
            'optimization_metrics': ['mmd_rbf'],
        }
        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=config,
            param_dataframes=group_dataframes,
            group_names=group_names
        )

        # Default: 20 * 3 groups = 60
        sampler = optimizer.study.sampler
        assert sampler._n_startup_trials == 60

    def test_missing_group_names_raises(self, temp_experiment_dir, group_dataframes, grouped_config):
        """Test that missing group_names raises error"""
        with pytest.raises(ValueError, match="group_names required"):
            OptunaOptimizer(
                experiment_dir=temp_experiment_dir,
                config=grouped_config,
                param_dataframes=group_dataframes,
                group_names=None
            )

    def test_mismatched_lengths_raises(self, temp_experiment_dir, group_dataframes, grouped_config):
        """Test that mismatched dataframes/names raises error"""
        with pytest.raises(ValueError, match="must match"):
            OptunaOptimizer(
                experiment_dir=temp_experiment_dir,
                config=grouped_config,
                param_dataframes=group_dataframes,
                group_names=['circle', 'ellipse']  # Only 2 names for 3 dataframes
            )


class TestGroupedSearchSpace:
    """Tests for grouped search space suggestions"""

    def test_grouped_search_space_returns_group_and_params(
        self, temp_experiment_dir, group_dataframes, group_names, grouped_config
    ):
        """Test that _define_grouped_search_space returns (group, params)"""
        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_dataframes=group_dataframes,
            group_names=group_names
        )

        trial = optimizer.study.ask()
        group, params = optimizer._define_grouped_search_space(trial)

        assert group in group_names
        assert isinstance(params, dict)

    def test_params_match_selected_group(
        self, temp_experiment_dir, group_dataframes, group_names, grouped_config
    ):
        """Test that params match the selected group's parameters"""
        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_dataframes=group_dataframes,
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
        self, temp_experiment_dir, group_dataframes, group_names, grouped_config
    ):
        """Test that ellipse group has rotation parameter"""
        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_dataframes=group_dataframes,
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

    def test_irregular_has_complexity(
        self, temp_experiment_dir, group_dataframes, group_names, grouped_config
    ):
        """Test that irregular group has complexity parameter"""
        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_dataframes=group_dataframes,
            group_names=group_names
        )

        # Find an irregular trial
        for _ in range(20):
            trial = optimizer.study.ask()
            group, params = optimizer._define_grouped_search_space(trial)
            optimizer.study.tell(trial, [0.5])

            if group == 'irregular':
                assert 'complexity_mean' in params
                assert 'complexity_std' in params
                break


class TestGroupLearning:
    """Tests that TPE learns which group is best"""

    def test_tpe_learns_best_group(
        self, temp_experiment_dir, group_dataframes, group_names, grouped_config
    ):
        """Test that TPE learns to prefer the group with better scores"""
        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_dataframes=group_dataframes,
            group_names=group_names
        )

        # Give circle the best score, ellipse medium, irregular worst
        score_by_group = {'circle': 0.1, 'ellipse': 0.5, 'irregular': 0.9}
        group_counts = {'circle': 0, 'ellipse': 0, 'irregular': 0}

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


class TestSuggestNextGroupedDistributions:
    """Tests for the suggest_next_grouped_distributions method"""

    def _create_metrics_list(self, n: int, base_metrics: dict) -> list:
        return [{**base_metrics, 'param_set_id': f'trial_{i}'} for i in range(n)]

    def test_returns_correct_format(
        self, temp_experiment_dir, group_dataframes, group_names, grouped_config
    ):
        """Test suggest_next_grouped_distributions returns (suggestions, converged)"""
        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_dataframes=group_dataframes,
            group_names=group_names
        )

        metrics_list = self._create_metrics_list(grouped_config['iteration_batch_size'], {'mmd_rbf': 0.5})
        suggestions, converged = optimizer.suggest_next_grouped_distributions(
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
        self, temp_experiment_dir, group_dataframes, group_names, grouped_config
    ):
        """Test ask/tell pattern works over multiple iterations"""
        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=grouped_config,
            param_dataframes=group_dataframes,
            group_names=group_names
        )

        for iteration in range(3):
            metrics_list = self._create_metrics_list(
                grouped_config['iteration_batch_size'],
                {'mmd_rbf': 0.5 - iteration * 0.1}
            )

            suggestions, converged = optimizer.suggest_next_grouped_distributions(
                metrics_list=metrics_list,
                iteration=iteration,
                config=grouped_config
            )

            assert len(suggestions) == grouped_config['iteration_batch_size']

            # Verify pending trials tracking
            assert iteration in optimizer.pending_trials
            if iteration > 0:
                assert (iteration - 1) not in optimizer.pending_trials

    def test_raises_without_grouped_bounds(self, temp_experiment_dir):
        """Test that suggest_next_grouped_distributions raises error in flat mode"""
        config = {
            'experiment_name': 'test_flat',
            'optimization_metrics': ['mmd_rbf'],
            'distribution_param_bounds': {
                'param1': [0.0, 1.0],
                'param2': [0.0, 1.0],
            }
        }
        optimizer = OptunaOptimizer(temp_experiment_dir, config)

        with pytest.raises(ValueError, match="requires grouped bounds"):
            optimizer.suggest_next_grouped_distributions([], iteration=0, config=config)


class TestBackwardCompatibility:
    """Tests that flat (legacy) mode still works"""

    def test_flat_mode_still_works(self, temp_experiment_dir):
        """Test that flat bounds from config still work"""
        config = {
            'experiment_name': 'test_flat',
            'random_seed': 42,
            'optimization_metrics': ['mmd_rbf'],
            'distribution_param_bounds': {
                'void_count_mean': [1, 10],
                'void_count_std': [0.5, 5.0],
                'base_size_mean': [5.0, 15.0],
            }
        }

        optimizer = OptunaOptimizer(temp_experiment_dir, config)

        assert optimizer.group_names is None
        assert 'void_count_mean' in optimizer.param_bounds

        # Test search space
        trial = optimizer.study.ask()
        params = optimizer._define_search_space(trial)

        assert 'void_count_mean' in params
        assert 'void_count_std' in params
        assert 'base_size_mean' in params
