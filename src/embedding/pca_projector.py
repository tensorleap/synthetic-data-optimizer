"""
PCA projector for dimensionality reduction.

Fits PCA on iteration 0, transforms subsequent iterations with fixed PCA.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from typing import Optional, Tuple


class PCAProjector:
    """Project embeddings to lower dimensions using PCA"""

    def __init__(self, n_components: int = 2):
        """
        Initialize PCA projector.

        Args:
            n_components: Number of principal components (default: 2 for visualization)
        """
        self.n_components = n_components
        self.pca = None
        self.fitted = False
        self.explained_variance_ratio_ = None

    def fit_transform(self, embeddings: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Fit PCA on embeddings and transform them.

        This should be called ONCE on iteration 0 data.
        All subsequent iterations use the same PCA via transform().

        Args:
            embeddings: (N, D) array of embeddings
            verbose: Print explained variance

        Returns:
            projections: (N, n_components) array
        """
        if self.fitted:
            raise RuntimeError("PCA already fitted. Use transform() for new data or create new instance.")

        # Automatically adjust n_components if dataset is too small
        n_samples, n_features = embeddings.shape
        max_components = min(n_samples, n_features)
        actual_n_components = min(self.n_components, max_components)

        if actual_n_components < self.n_components:
            if verbose:
                print(f"Warning: Requested {self.n_components} components but dataset only has "
                      f"{n_samples} samples and {n_features} features.")
                print(f"         Reducing to {actual_n_components} components (max possible).")

        if verbose:
            print(f"Fitting PCA with {actual_n_components} components on {n_samples} samples...")

        # Fit PCA
        self.pca = PCA(n_components=actual_n_components)
        projections = self.pca.fit_transform(embeddings)
        self.fitted = True
        self.n_components = actual_n_components  # Update to actual fitted components
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_

        if verbose:
            print(f"PCA fitted successfully")
            print(f"Total explained variance: {self.explained_variance_ratio_.sum():.4f}")

        return projections

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using fitted PCA.

        Args:
            embeddings: (N, D) array of embeddings

        Returns:
            projections: (N, n_components) array
        """
        if not self.fitted:
            raise RuntimeError("PCA not fitted. Call fit_transform() first.")

        return self.pca.transform(embeddings)

    def inverse_transform(self, projections: np.ndarray) -> np.ndarray:
        """
        Reconstruct embeddings from projections.

        Args:
            projections: (N, n_components) array

        Returns:
            embeddings: (N, D) reconstructed embeddings
        """
        if not self.fitted:
            raise RuntimeError("PCA not fitted. Call fit_transform() first.")

        return self.pca.inverse_transform(projections)

    def save(self, path: Path) -> None:
        """
        Save fitted PCA model to disk.

        Args:
            path: Path to save file (e.g., 'pca_model.pkl')
        """
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted PCA model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'pca': self.pca,
            'n_components': self.n_components,
            'fitted': self.fitted,
            'explained_variance_ratio': self.explained_variance_ratio_
        }

        joblib.dump(state, path)
        print(f"PCA model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'PCAProjector':
        """
        Load fitted PCA model from disk.

        Args:
            path: Path to saved model file

        Returns:
            Loaded PCAProjector instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PCA model not found at {path}")

        state = joblib.load(path)

        projector = cls(n_components=state['n_components'])
        projector.pca = state['pca']
        projector.fitted = state['fitted']
        projector.explained_variance_ratio_ = state['explained_variance_ratio']

        print(f"PCA model loaded from {path}")
        print(f"Total explained variance: {projector.explained_variance_ratio_.sum():.4f}")

        return projector

    def get_component_loadings(self, feature_names: Optional[list] = None) -> np.ndarray:
        """
        Get principal component loadings (weights).

        Args:
            feature_names: Optional list of feature names for interpretation

        Returns:
            components: (n_components, n_features) array of loadings
        """
        if not self.fitted:
            raise RuntimeError("PCA not fitted. Call fit_transform() first.")

        return self.pca.components_
