"""
Unit tests for metrics module, including per-parameter-set metrics computation.
"""

import pytest
import numpy as np

from src.optimization.metrics import (
    compute_all_metrics,
    compute_per_param_set_metrics,
    DistributionMetrics,
    SampleMetrics
)


class TestComputePerParamSetMetrics:
    """Tests for compute_per_param_set_metrics function"""

    def test_basic_grouping_and_metrics(self):
        """Test that embeddings are correctly grouped by param_set_id and metrics computed"""
        # Create synthetic data: 2 param sets, each with 3 replications
        # Total: 6 samples
        param_set_0_embeddings = np.array([
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [1.2, 2.2, 3.2]
        ])
        param_set_1_embeddings = np.array([
            [5.0, 6.0, 7.0],
            [5.1, 6.1, 7.1],
            [5.2, 6.2, 7.2]
        ])

        synthetic_embeddings = np.vstack([param_set_0_embeddings, param_set_1_embeddings])

        synthetic_metadata = [
            {'param_set_id': 'ps_000', 'replication_id': 0},
            {'param_set_id': 'ps_000', 'replication_id': 1},
            {'param_set_id': 'ps_000', 'replication_id': 2},
            {'param_set_id': 'ps_001', 'replication_id': 0},
            {'param_set_id': 'ps_001', 'replication_id': 1},
            {'param_set_id': 'ps_001', 'replication_id': 2},
        ]

        # Real distribution
        real_embeddings = np.array([
            [2.0, 3.0, 4.0],
            [2.5, 3.5, 4.5]
        ])

        # Compute per-param-set metrics
        metrics_list = compute_per_param_set_metrics(
            synthetic_embeddings,
            synthetic_metadata,
            real_embeddings,
            n_param_sets=2
        )

        # Verify we get 2 metric dicts (one per param set)
        assert len(metrics_list) == 2

        # Verify each has the expected metrics
        for metrics in metrics_list:
            assert 'mmd_rbf' in metrics
            assert 'wasserstein' in metrics
            assert 'mean_nn_distance' in metrics
            assert 'param_set_id' in metrics

        # Verify param_set_ids are correct
        assert metrics_list[0]['param_set_id'] == 'ps_000'
        assert metrics_list[1]['param_set_id'] == 'ps_001'

        # Verify metrics are different for different param sets
        # (ps_000 is closer to real than ps_001)
        assert metrics_list[0]['mean_nn_distance'] < metrics_list[1]['mean_nn_distance']

    def test_single_param_set(self):
        """Test with single parameter set"""
        synthetic_embeddings = np.random.randn(5, 10)
        synthetic_metadata = [{'param_set_id': 'ps_000'} for _ in range(5)]
        real_embeddings = np.random.randn(3, 10)

        metrics_list = compute_per_param_set_metrics(
            synthetic_embeddings,
            synthetic_metadata,
            real_embeddings,
            n_param_sets=1
        )

        assert len(metrics_list) == 1
        assert metrics_list[0]['param_set_id'] == 'ps_000'

    def test_many_param_sets(self):
        """Test with many parameter sets"""
        n_param_sets = 50
        replications_per_set = 10
        n_samples = n_param_sets * replications_per_set

        synthetic_embeddings = np.random.randn(n_samples, 400)

        # Create metadata with proper param_set_ids
        synthetic_metadata = []
        for ps_idx in range(n_param_sets):
            for rep_idx in range(replications_per_set):
                synthetic_metadata.append({
                    'param_set_id': f'ps_{ps_idx:03d}',
                    'replication_id': rep_idx
                })

        real_embeddings = np.random.randn(15, 400)

        metrics_list = compute_per_param_set_metrics(
            synthetic_embeddings,
            synthetic_metadata,
            real_embeddings,
            n_param_sets=n_param_sets
        )

        # Verify we get one metric dict per param set
        assert len(metrics_list) == n_param_sets

        # Verify all param_set_ids are unique and correctly formatted
        param_set_ids = [m['param_set_id'] for m in metrics_list]
        assert len(set(param_set_ids)) == n_param_sets
        assert param_set_ids == sorted(param_set_ids)

    def test_mismatch_param_sets_raises_error(self):
        """Test that mismatch between expected and actual param sets raises error"""
        synthetic_embeddings = np.random.randn(6, 10)

        # Only provide 2 param sets but claim 3
        synthetic_metadata = [
            {'param_set_id': 'ps_000'},
            {'param_set_id': 'ps_000'},
            {'param_set_id': 'ps_000'},
            {'param_set_id': 'ps_001'},
            {'param_set_id': 'ps_001'},
            {'param_set_id': 'ps_001'},
        ]

        real_embeddings = np.random.randn(3, 10)

        with pytest.raises(ValueError, match="Expected 3 parameter sets, but found 2"):
            compute_per_param_set_metrics(
                synthetic_embeddings,
                synthetic_metadata,
                real_embeddings,
                n_param_sets=3
            )

    def test_unequal_replications_per_set(self):
        """Test that function works even with unequal replications per param set"""
        # Param set 0: 2 replications
        # Param set 1: 4 replications
        synthetic_embeddings = np.random.randn(6, 10)
        synthetic_metadata = [
            {'param_set_id': 'ps_000'},
            {'param_set_id': 'ps_000'},
            {'param_set_id': 'ps_001'},
            {'param_set_id': 'ps_001'},
            {'param_set_id': 'ps_001'},
            {'param_set_id': 'ps_001'},
        ]

        real_embeddings = np.random.randn(3, 10)

        metrics_list = compute_per_param_set_metrics(
            synthetic_embeddings,
            synthetic_metadata,
            real_embeddings,
            n_param_sets=2
        )

        assert len(metrics_list) == 2

    def test_metrics_consistency_with_compute_all_metrics(self):
        """Test that per-set metrics match compute_all_metrics when given same data"""
        # Create embeddings for a single param set
        single_set_embeddings = np.random.randn(5, 20)
        real_embeddings = np.random.randn(3, 20)

        # Compute metrics directly
        direct_metrics = compute_all_metrics(single_set_embeddings, real_embeddings)

        # Compute metrics via per-param-set function
        synthetic_metadata = [{'param_set_id': 'ps_000'} for _ in range(5)]
        metrics_list = compute_per_param_set_metrics(
            single_set_embeddings,
            synthetic_metadata,
            real_embeddings,
            n_param_sets=1
        )

        # Compare key metrics (excluding param_set_id which is added by per-set function)
        assert np.isclose(metrics_list[0]['mmd_rbf'], direct_metrics['mmd_rbf'])
        assert np.isclose(metrics_list[0]['wasserstein'], direct_metrics['wasserstein'])
        assert np.isclose(metrics_list[0]['mean_nn_distance'], direct_metrics['mean_nn_distance'])
