"""
Tests for OptunaOptimizer joint mode (shape probabilities + all params).

Tests use get_param_bounds() to mirror the real pipeline flow.
"""

import math
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
def joint_config():
    """Config for joint mode tests"""
    return {
        'experiment_name': 'test_joint',
        'random_seed': 42,
        'iteration_batch_size': 4,
        'max_iterations': 5,
        'optimization_metrics': ['mmd_rbf'],
        'optimizer': {'n_startup_trials': 10}
    }


class TestJointOptimizerInit:
    """Tests for joint optimizer initialization"""

    def test_init_with_param_bounds(self, temp_experiment_dir, param_bounds_and_groups, joint_config):
        """Test initializing optimizer with param_bounds"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=joint_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        assert optimizer.group_names == group_names
        assert len(optimizer.param_bounds) == len(group_names)
        for group in group_names:
            assert group in optimizer.param_bounds

    def test_default_logit_bounds(self, temp_experiment_dir, param_bounds_and_groups, joint_config):
        """Test default logit bounds are (-5.0, 5.0)"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=joint_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        assert optimizer.logit_bounds == (-5.0, 5.0)

    def test_custom_logit_bounds(self, temp_experiment_dir, param_bounds_and_groups, joint_config):
        """Test custom logit bounds"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=joint_config,
            param_bounds=param_bounds,
            group_names=group_names,
            logit_bounds=(-3.0, 3.0)
        )

        assert optimizer.logit_bounds == (-3.0, 3.0)

    def test_startup_trials_scales_with_params(self, temp_experiment_dir, param_bounds_and_groups):
        """Test that n_startup_trials scales with total param count"""
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

        # Count total params: logits + all shape params
        total_params = len(group_names)
        for group_bounds in param_bounds.values():
            total_params += len(group_bounds)

        expected_startup = max(50, 3 * total_params)
        sampler = optimizer.study.sampler
        assert sampler._n_startup_trials == expected_startup

    def test_missing_params_raises(self, temp_experiment_dir, joint_config):
        """Test that missing param_bounds or group_names raises error"""
        with pytest.raises(ValueError, match="param_bounds and group_names are required"):
            OptunaOptimizer(
                experiment_dir=temp_experiment_dir,
                config=joint_config,
                param_bounds=None,
                group_names=None
            )

    def test_mismatched_keys_raises(self, temp_experiment_dir, param_bounds_and_groups, joint_config):
        """Test that mismatched param_bounds keys and group_names raises error"""
        param_bounds, _ = param_bounds_and_groups

        with pytest.raises(ValueError, match="must match"):
            OptunaOptimizer(
                experiment_dir=temp_experiment_dir,
                config=joint_config,
                param_bounds=param_bounds,
                group_names=['circle', 'ellipse']  # Missing 'irregular'
            )


class TestJointSearchSpace:
    """Tests for joint search space suggestions"""

    def test_joint_search_space_contains_logits(
        self, temp_experiment_dir, param_bounds_and_groups, joint_config
    ):
        """Test that _define_joint_search_space includes shape logits"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=joint_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        trial = optimizer.study.ask()
        params = optimizer._define_joint_search_space(trial)

        # Check logits for all groups
        for group in group_names:
            logit_key = f'shape_logit_{group}'
            assert logit_key in params, f"Missing {logit_key}"
            assert isinstance(params[logit_key], float)

    def test_joint_search_space_contains_all_params(
        self, temp_experiment_dir, param_bounds_and_groups, joint_config
    ):
        """Test that _define_joint_search_space includes all shape params"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=joint_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        trial = optimizer.study.ask()
        params = optimizer._define_joint_search_space(trial)

        # Check all params for all groups
        for group in group_names:
            for param_name in param_bounds[group].keys():
                full_key = f'{group}__{param_name}'
                assert full_key in params, f"Missing {full_key}"

    def test_params_within_bounds(
        self, temp_experiment_dir, param_bounds_and_groups, joint_config
    ):
        """Test that suggested params are within their bounds"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=joint_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        for _ in range(5):
            trial = optimizer.study.ask()
            params = optimizer._define_joint_search_space(trial)
            optimizer.study.tell(trial, [0.5])

            # Check logits within bounds
            for group in group_names:
                logit = params[f'shape_logit_{group}']
                assert optimizer.logit_bounds[0] <= logit <= optimizer.logit_bounds[1]

            # Check params within bounds
            for group in group_names:
                for param_name, bounds in param_bounds[group].items():
                    value = params[f'{group}__{param_name}']
                    assert bounds[0] <= value <= bounds[1], \
                        f"{group}__{param_name}={value} not in {bounds}"


class TestLogitUtilities:
    """Tests for logit conversion utilities"""

    def test_sample_counts_to_logits_uniform(self):
        """Test sample_counts_to_logits with uniform distribution"""
        counts = {'circle': 100, 'ellipse': 100, 'irregular': 100}
        logits = OptunaOptimizer.sample_counts_to_logits(counts)

        assert 'shape_logit_circle' in logits
        assert 'shape_logit_ellipse' in logits
        assert 'shape_logit_irregular' in logits

        # Uniform distribution should give equal logits
        values = list(logits.values())
        assert all(abs(v - values[0]) < 0.01 for v in values)

    def test_sample_counts_to_logits_skewed(self):
        """Test sample_counts_to_logits with skewed distribution"""
        counts = {'circle': 100, 'ellipse': 50, 'irregular': 25}
        logits = OptunaOptimizer.sample_counts_to_logits(counts)

        # Circle should have highest logit (most samples)
        assert logits['shape_logit_circle'] > logits['shape_logit_ellipse']
        assert logits['shape_logit_ellipse'] > logits['shape_logit_irregular']

    def test_sample_counts_to_logits_zero_raises(self):
        """Test that zero total samples raises error"""
        counts = {'circle': 0, 'ellipse': 0}
        with pytest.raises(ValueError, match="Total sample count cannot be zero"):
            OptunaOptimizer.sample_counts_to_logits(counts)

    def test_sample_counts_with_zero_shape(self):
        """Test that zero samples for one shape yields very low logit"""
        counts = {'circle': 100, 'ellipse': 0, 'irregular': 50}
        logits = OptunaOptimizer.sample_counts_to_logits(counts)

        # Ellipse should have very low logit (clamped at 1e-6)
        assert logits['shape_logit_ellipse'] < -10  # log(1e-6) ≈ -13.8

    def test_logits_to_probabilities_sum_to_one(self):
        """Test that logits_to_probabilities sums to 1.0"""
        params = {
            'shape_logit_circle': 0.5,
            'shape_logit_ellipse': -0.3,
            'shape_logit_irregular': -0.2,
            'circle__radius_mean': 10.0  # should be ignored
        }
        probs = OptunaOptimizer.logits_to_probabilities(params)

        assert abs(sum(probs.values()) - 1.0) < 1e-9

    def test_logits_to_probabilities_ordering(self):
        """Test that higher logit gives higher probability"""
        params = {
            'shape_logit_circle': 2.0,
            'shape_logit_ellipse': 0.0,
            'shape_logit_irregular': -2.0,
        }
        probs = OptunaOptimizer.logits_to_probabilities(params)

        assert probs['circle'] > probs['ellipse'] > probs['irregular']

    def test_logits_to_probabilities_equal_logits(self):
        """Test that equal logits give equal probabilities"""
        params = {
            'shape_logit_circle': 1.0,
            'shape_logit_ellipse': 1.0,
            'shape_logit_irregular': 1.0,
        }
        probs = OptunaOptimizer.logits_to_probabilities(params)

        assert abs(probs['circle'] - 1/3) < 1e-9
        assert abs(probs['ellipse'] - 1/3) < 1e-9
        assert abs(probs['irregular'] - 1/3) < 1e-9

    def test_roundtrip_counts_to_logits_to_probs(self):
        """Test that counts -> logits -> probs preserves ratios"""
        counts = {'circle': 100, 'ellipse': 60, 'irregular': 40}
        total = sum(counts.values())
        expected_probs = {k: v / total for k, v in counts.items()}

        logits = OptunaOptimizer.sample_counts_to_logits(counts)
        probs = OptunaOptimizer.logits_to_probabilities(logits)

        for shape in counts.keys():
            assert abs(probs[shape] - expected_probs[shape]) < 1e-6


class TestBuildFullDistributions:
    """Tests for _build_full_distributions"""

    def test_contains_all_logit_distributions(
        self, temp_experiment_dir, param_bounds_and_groups, joint_config
    ):
        """Test that _build_full_distributions includes logit distributions"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=joint_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        distributions = optimizer._build_full_distributions()

        for group in group_names:
            key = f'shape_logit_{group}'
            assert key in distributions
            import optuna
            assert isinstance(distributions[key], optuna.distributions.FloatDistribution)

    def test_contains_all_param_distributions(
        self, temp_experiment_dir, param_bounds_and_groups, joint_config
    ):
        """Test that _build_full_distributions includes all param distributions"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=joint_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        distributions = optimizer._build_full_distributions()

        for group in group_names:
            for param_name in param_bounds[group].keys():
                key = f'{group}__{param_name}'
                assert key in distributions


class TestSuggestNextDistributions:
    """Tests for the suggest_next_distributions method"""

    def _create_joint_distribution(self, group_names: list, param_bounds: dict) -> dict:
        """Create a mock joint distribution with all params."""
        params = {}

        # Add logits (uniform distribution)
        for group in group_names:
            params[f'shape_logit_{group}'] = 0.0  # Equal probability

        # Add all shape params at midpoint
        for group in group_names:
            for param_name, bounds in param_bounds[group].items():
                params[f'{group}__{param_name}'] = (bounds[0] + bounds[1]) / 2

        return params

    def _create_distributions(self, n: int, group_names: list, param_bounds: dict) -> list:
        """Create mock distributions for testing."""
        distributions = []
        for i in range(n):
            dist_id = f"dist_{i}"
            params = self._create_joint_distribution(group_names, param_bounds)
            distributions.append((dist_id, params))
        return distributions

    def _create_metrics_list(self, n: int, base_metrics: dict) -> list:
        return [{**base_metrics} for _ in range(n)]

    def test_returns_correct_format(
        self, temp_experiment_dir, param_bounds_and_groups, joint_config
    ):
        """Test suggest_next_distributions returns (dist_id, params) tuples"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=joint_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        n = joint_config['iteration_batch_size']
        current_distributions = self._create_distributions(n, group_names, param_bounds)
        metrics_list = self._create_metrics_list(n, {'mmd_rbf': 0.5})

        suggestions = optimizer.suggest_next_distributions(
            current_distributions=current_distributions,
            metrics_list=metrics_list,
            config=joint_config
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == joint_config['iteration_batch_size']

        # Each suggestion is (dist_id, params_dict)
        for dist_id, params in suggestions:
            assert isinstance(dist_id, str)
            assert dist_id.startswith('dist_')
            assert isinstance(params, dict)

            # Params should contain logits
            for group in group_names:
                assert f'shape_logit_{group}' in params

            # Params should contain all shape params
            for group in group_names:
                for param_name in param_bounds[group].keys():
                    assert f'{group}__{param_name}' in params

    def test_add_trial_pattern(
        self, temp_experiment_dir, param_bounds_and_groups, joint_config
    ):
        """Test add_trial pattern works over multiple iterations"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=joint_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        n = joint_config['iteration_batch_size']
        current_distributions = self._create_distributions(n, group_names, param_bounds)

        for iteration in range(3):
            metrics_list = self._create_metrics_list(n, {'mmd_rbf': 0.5 - iteration * 0.1})

            suggestions = optimizer.suggest_next_distributions(
                current_distributions=current_distributions,
                metrics_list=metrics_list,
                config=joint_config
            )

            assert len(suggestions) == joint_config['iteration_batch_size']

            # Use suggestions as next distributions
            current_distributions = suggestions

        # Verify trials were registered (3 iterations * n distributions)
        import optuna
        completed = [t for t in optimizer.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        assert len(completed) == 3 * n

    def test_mismatched_distributions_metrics_raises(
        self, temp_experiment_dir, param_bounds_and_groups, joint_config
    ):
        """Test that mismatched distributions and metrics raises error"""
        param_bounds, group_names = param_bounds_and_groups

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=joint_config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        distributions = self._create_distributions(4, group_names, param_bounds)
        metrics_list = self._create_metrics_list(3, {'mmd_rbf': 0.5})  # Wrong count

        with pytest.raises(ValueError, match="Mismatch"):
            optimizer.suggest_next_distributions(
                current_distributions=distributions,
                metrics_list=metrics_list,
                config=joint_config
            )


class TestParetoFront:
    """Tests for Pareto front functionality"""

    def test_pareto_front_tracking(
        self, temp_experiment_dir, param_bounds_and_groups
    ):
        """Test that Pareto front is tracked correctly"""
        param_bounds, group_names = param_bounds_and_groups
        config = {
            'experiment_name': 'test_pareto',
            'random_seed': 42,
            'iteration_batch_size': 2,
            'optimization_metrics': ['mmd_rbf', 'mean_nn_distance'],  # Multi-objective
            'optimizer': {'n_startup_trials': 5}
        }

        optimizer = OptunaOptimizer(
            experiment_dir=temp_experiment_dir,
            config=config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        # Create test helper
        def create_joint_params():
            params = {}
            for group in group_names:
                params[f'shape_logit_{group}'] = 0.0
            for group in group_names:
                for param_name, bounds in param_bounds[group].items():
                    params[f'{group}__{param_name}'] = (bounds[0] + bounds[1]) / 2
            return params

        # Add some trials
        distributions = [(f"dist_{i}", create_joint_params()) for i in range(4)]
        metrics_list = [
            {'mmd_rbf': 0.1, 'mean_nn_distance': 0.9},  # Good on mmd, bad on nn
            {'mmd_rbf': 0.9, 'mean_nn_distance': 0.1},  # Bad on mmd, good on nn
            {'mmd_rbf': 0.5, 'mean_nn_distance': 0.5},  # Balanced
            {'mmd_rbf': 0.8, 'mean_nn_distance': 0.8},  # Dominated
        ]

        optimizer.suggest_next_distributions(
            current_distributions=distributions,
            metrics_list=metrics_list,
            config=config
        )

        pareto_front = optimizer.get_pareto_front()
        # First 3 should be on Pareto front, last one is dominated
        assert len(pareto_front) == 3
