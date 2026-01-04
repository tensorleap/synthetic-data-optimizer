"""
Metrics for evaluating distribution similarity and sample distances.

Includes MMD, Wasserstein distance, and per-sample distance metrics.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from typing import Dict, Tuple


class DistributionMetrics:
    """Calculate distribution similarity metrics"""

    @staticmethod
    def mmd(X: np.ndarray, Y: np.ndarray, kernel: str = 'rbf', gamma: float = 1.0) -> float:
        """
        Calculate Maximum Mean Discrepancy (MMD) between two distributions.

        Args:
            X: (N, D) array of samples from distribution 1
            Y: (M, D) array of samples from distribution 2
            kernel: Kernel type ('rbf' or 'linear')
            gamma: RBF kernel parameter

        Returns:
            mmd_value: MMD distance (lower is better)
        """
        if kernel == 'rbf':
            # RBF kernel: k(x,y) = exp(-gamma * ||x-y||^2)
            XX = DistributionMetrics._rbf_kernel(X, X, gamma)
            YY = DistributionMetrics._rbf_kernel(Y, Y, gamma)
            XY = DistributionMetrics._rbf_kernel(X, Y, gamma)
        elif kernel == 'linear':
            # Linear kernel: k(x,y) = x^T y
            XX = X @ X.T
            YY = Y @ Y.T
            XY = X @ Y.T
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        mmd = XX.mean() + YY.mean() - 2 * XY.mean()
        return float(np.sqrt(max(mmd, 0)))  # Ensure non-negative due to numerical errors

    @staticmethod
    def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
        """Compute RBF (Gaussian) kernel between X and Y"""
        # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
        distances_sq = X_norm + Y_norm - 2 * X @ Y.T
        return np.exp(-gamma * distances_sq)

    @staticmethod
    def wasserstein_1d(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate 1D Wasserstein distance (Earth Mover's Distance).

        For high-dimensional data, compute average over all dimensions.

        Args:
            X: (N, D) array of samples from distribution 1
            Y: (M, D) array of samples from distribution 2

        Returns:
            wasserstein: Average 1D Wasserstein distance across dimensions
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        distances = []
        for dim in range(X.shape[1]):
            dist = wasserstein_distance(X[:, dim], Y[:, dim])
            distances.append(dist)

        return float(np.mean(distances))


class SampleMetrics:
    """Calculate per-sample distance metrics"""

    @staticmethod
    def nearest_neighbor_distances(
        synthetic: np.ndarray,
        real: np.ndarray,
        metric: str = 'euclidean'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate distance from each synthetic sample to nearest real sample.

        Args:
            synthetic: (N, D) array of synthetic embeddings
            real: (M, D) array of real embeddings
            metric: Distance metric ('euclidean', 'cosine', etc.)

        Returns:
            distances: (N,) array of nearest neighbor distances
            indices: (N,) array of nearest neighbor indices in real
        """
        # Compute pairwise distances
        dist_matrix = cdist(synthetic, real, metric=metric)

        # Find nearest neighbor
        nearest_distances = dist_matrix.min(axis=1)
        nearest_indices = dist_matrix.argmin(axis=1)

        return nearest_distances, nearest_indices

    @staticmethod
    def coverage(
        synthetic: np.ndarray,
        real: np.ndarray,
        threshold: float = None,
        metric: str = 'euclidean'
    ) -> Dict[str, float]:
        """
        Calculate coverage: fraction of synthetic samples within threshold of real.

        Args:
            synthetic: (N, D) array of synthetic embeddings
            real: (M, D) array of real embeddings
            threshold: Distance threshold (if None, uses median real-real distance)
            metric: Distance metric

        Returns:
            Dictionary with coverage metrics
        """
        # Calculate nearest neighbor distances
        nn_distances, _ = SampleMetrics.nearest_neighbor_distances(synthetic, real, metric)

        # If no threshold provided, use median of real-real distances
        if threshold is None:
            real_real_distances = cdist(real, real, metric=metric)
            # Exclude diagonal (self-distances)
            np.fill_diagonal(real_real_distances, np.inf)
            threshold = np.median(real_real_distances[np.isfinite(real_real_distances)])

        # Calculate coverage
        within_threshold = (nn_distances <= threshold).sum()
        coverage_ratio = within_threshold / len(synthetic)

        return {
            'coverage': float(coverage_ratio),
            'threshold': float(threshold),
            'mean_distance': float(nn_distances.mean()),
            'median_distance': float(np.median(nn_distances)),
            'within_threshold_count': int(within_threshold)
        }


def compute_all_metrics(
    synthetic_embeddings: np.ndarray,
    real_embeddings: np.ndarray
) -> Dict[str, float]:
    """
    Compute all metrics for synthetic vs real comparison.

    Args:
        synthetic_embeddings: (N, D) array
        real_embeddings: (M, D) array

    Returns:
        Dictionary with all metric values
    """
    metrics = {}

    # Distribution-level metrics
    metrics['mmd_rbf'] = DistributionMetrics.mmd(synthetic_embeddings, real_embeddings, kernel='rbf', gamma=1.0)
    metrics['mmd_linear'] = DistributionMetrics.mmd(synthetic_embeddings, real_embeddings, kernel='linear')
    metrics['wasserstein'] = DistributionMetrics.wasserstein_1d(synthetic_embeddings, real_embeddings)

    # Sample-level metrics
    nn_distances, _ = SampleMetrics.nearest_neighbor_distances(synthetic_embeddings, real_embeddings)
    metrics['mean_nn_distance'] = float(nn_distances.mean())
    metrics['median_nn_distance'] = float(np.median(nn_distances))

    # Coverage
    coverage_info = SampleMetrics.coverage(synthetic_embeddings, real_embeddings)
    metrics.update(coverage_info)

    return metrics
