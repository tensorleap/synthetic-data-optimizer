"""
Unit tests for metrics module, including per-distribution metrics computation.
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
        """Test that embeddings are correctly grouped by distribution_id and metrics computed"""
        # Create synthetic data: 2 distributions, each with 3 replications
        # Total: 6 samples
        dist_0_embeddings = np.array([
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [1.2, 2.2, 3.2]
        ])
        dist_1_embeddings = np.array([
            [5.0, 6.0, 7.0],
            [5.1, 6.1, 7.1],
            [5.2, 6.2, 7.2]
        ])

        synthetic_embeddings = np.vstack([dist_0_embeddings, dist_1_embeddings])

        synthetic_metadata = [
            {'distribution_id': 0, 'replication_id': 0},
            {'distribution_id': 0, 'replication_id': 1},
            {'distribution_id': 0, 'replication_id': 2},
            {'distribution_id': 1, 'replication_id': 0},
            {'distribution_id': 1, 'replication_id': 1},
            {'distribution_id': 1, 'replication_id': 2},
        ]

        # Real distribution
        real_embeddings = np.array([
            [2.0, 3.0, 4.0],
            [2.5, 3.5, 4.5]
        ])

        # Compute per-distribution metrics
        metrics_list = compute_per_param_set_metrics(
            synthetic_embeddings,
            synthetic_metadata,
            real_embeddings,
            n_param_sets=2
        )

        # Verify we get 2 metric dicts (one per distribution)
        assert len(metrics_list) == 2

        # Verify each has the expected metrics
        for metrics in metrics_list:
            assert 'mmd_rbf' in metrics
            assert 'wasserstein' in metrics
            assert 'mean_nn_distance' in metrics
            assert 'distribution_id' in metrics

        # Verify distribution_ids are correct
        assert metrics_list[0]['distribution_id'] == 0
        assert metrics_list[1]['distribution_id'] == 1

        # Verify metrics are different for different distributions
        # (dist 0 is closer to real than dist 1)
        assert metrics_list[0]['mean_nn_distance'] < metrics_list[1]['mean_nn_distance']

    def test_single_distribution(self):
        """Test with single distribution"""
        synthetic_embeddings = np.random.randn(5, 10)
        synthetic_metadata = [{'distribution_id': 0} for _ in range(5)]
        real_embeddings = np.random.randn(3, 10)

        metrics_list = compute_per_param_set_metrics(
            synthetic_embeddings,
            synthetic_metadata,
            real_embeddings,
            n_param_sets=1
        )

        assert len(metrics_list) == 1
        assert metrics_list[0]['distribution_id'] == 0

    def test_many_distributions(self):
        """Test with many distributions"""
        n_distributions = 50
        replications_per_dist = 10
        n_samples = n_distributions * replications_per_dist

        synthetic_embeddings = np.random.randn(n_samples, 400)

        # Create metadata with proper distribution_ids
        synthetic_metadata = []
        for dist_idx in range(n_distributions):
            for rep_idx in range(replications_per_dist):
                synthetic_metadata.append({
                    'distribution_id': dist_idx,
                    'replication_id': rep_idx
                })

        real_embeddings = np.random.randn(15, 400)

        metrics_list = compute_per_param_set_metrics(
            synthetic_embeddings,
            synthetic_metadata,
            real_embeddings,
            n_param_sets=n_distributions
        )

        # Verify we get one metric dict per distribution
        assert len(metrics_list) == n_distributions

        # Verify all distribution_ids are unique and correctly ordered
        distribution_ids = [m['distribution_id'] for m in metrics_list]
        assert len(set(distribution_ids)) == n_distributions
        assert distribution_ids == sorted(distribution_ids)

    def test_mismatch_distributions_raises_error(self):
        """Test that mismatch between expected and actual distributions raises error"""
        synthetic_embeddings = np.random.randn(6, 10)

        # Only provide 2 distributions but claim 3
        synthetic_metadata = [
            {'distribution_id': 0},
            {'distribution_id': 0},
            {'distribution_id': 0},
            {'distribution_id': 1},
            {'distribution_id': 1},
            {'distribution_id': 1},
        ]

        real_embeddings = np.random.randn(3, 10)

        with pytest.raises(ValueError, match="Expected 3 distributions, but found 2"):
            compute_per_param_set_metrics(
                synthetic_embeddings,
                synthetic_metadata,
                real_embeddings,
                n_param_sets=3
            )

    def test_unequal_replications_per_distribution(self):
        """Test that function works even with unequal replications per distribution"""
        # Distribution 0: 2 replications
        # Distribution 1: 4 replications
        synthetic_embeddings = np.random.randn(6, 10)
        synthetic_metadata = [
            {'distribution_id': 0},
            {'distribution_id': 0},
            {'distribution_id': 1},
            {'distribution_id': 1},
            {'distribution_id': 1},
            {'distribution_id': 1},
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
        """Test that per-distribution metrics match compute_all_metrics when given same data"""
        # Create embeddings for a single distribution
        single_dist_embeddings = np.random.randn(5, 20)
        real_embeddings = np.random.randn(3, 20)

        # Compute metrics directly
        direct_metrics = compute_all_metrics(single_dist_embeddings, real_embeddings)

        # Compute metrics via per-distribution function
        synthetic_metadata = [{'distribution_id': 0} for _ in range(5)]
        metrics_list = compute_per_param_set_metrics(
            single_dist_embeddings,
            synthetic_metadata,
            real_embeddings,
            n_param_sets=1
        )

        # Compare key metrics (excluding distribution_id which is added by per-set function)
        assert np.isclose(metrics_list[0]['mmd_rbf'], direct_metrics['mmd_rbf'])
        assert np.isclose(metrics_list[0]['wasserstein'], direct_metrics['wasserstein'])
        assert np.isclose(metrics_list[0]['mean_nn_distance'], direct_metrics['mean_nn_distance'])
