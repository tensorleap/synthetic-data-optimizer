"""
Mock data generator for testing the conditional optimizer.

Generates synthetic embeddings and distribution parameters for k conditional groups.
Each group can have DIFFERENT distribution parameters (e.g., ellipse has rotation_mean/std,
circle doesn't).

The DataFrame columns are the DISTRIBUTION PARAMETERS that Optuna optimizes
(e.g., void_count_mean, void_count_std), not the sampled values.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class MockDataGenerator:
    """
    Generate mock data for testing the conditional parameters optimizer.

    Each shape group has DIFFERENT distribution parameters:
    - circle: No rotation (circles are rotation-invariant)
    - ellipse: Has rotation_mean, rotation_std
    - irregular: No rotation, has complexity_mean, complexity_std instead

    The DataFrame columns are distribution parameters (what Optuna optimizes),
    with _mean and _std suffixes for continuous parameters.
    """

    # Default distribution parameter specs for each shape group
    # Format: {param_base: {mean_bounds: [min, max], std_bounds: [min, max]}}
    DEFAULT_GROUP_SPECS = {
        'circle': {
            # Circles don't have rotation (rotation-invariant)
            'void_count': {'mean_bounds': [1, 10], 'std_bounds': [0.5, 5.0]},
            'base_size': {'mean_bounds': [5.0, 15.0], 'std_bounds': [0.5, 5.0]},
            'center_x': {'mean_bounds': [0.2, 0.8], 'std_bounds': [0.05, 0.3]},
            'center_y': {'mean_bounds': [0.2, 0.8], 'std_bounds': [0.05, 0.3]},
            'position_spread': {'mean_bounds': [0.1, 0.8], 'std_bounds': [0.05, 0.3]},
        },
        'ellipse': {
            # Ellipses HAVE rotation - unique parameter for this shape
            'void_count': {'mean_bounds': [1, 8], 'std_bounds': [0.5, 4.0]},
            'base_size': {'mean_bounds': [6.0, 18.0], 'std_bounds': [1.0, 6.0]},
            'rotation': {'mean_bounds': [0.0, 360.0], 'std_bounds': [10.0, 180.0]},
            'center_x': {'mean_bounds': [0.15, 0.85], 'std_bounds': [0.05, 0.25]},
            'center_y': {'mean_bounds': [0.15, 0.85], 'std_bounds': [0.05, 0.25]},
            'position_spread': {'mean_bounds': [0.1, 0.7], 'std_bounds': [0.05, 0.25]},
        },
        'irregular': {
            # Irregular shapes have complexity instead of rotation
            'void_count': {'mean_bounds': [1, 6], 'std_bounds': [0.5, 3.0]},
            'base_size': {'mean_bounds': [4.0, 12.0], 'std_bounds': [0.5, 4.0]},
            'complexity': {'mean_bounds': [5, 12], 'std_bounds': [1.0, 3.0]},
            'center_x': {'mean_bounds': [0.25, 0.75], 'std_bounds': [0.05, 0.2]},
            'center_y': {'mean_bounds': [0.25, 0.75], 'std_bounds': [0.05, 0.2]},
            'position_spread': {'mean_bounds': [0.05, 0.6], 'std_bounds': [0.05, 0.2]},
        },
    }

    def __init__(
        self,
        base_image_dir: Optional[Path] = None,
        dino_model: str = "dinov2_vitb14",
        use_real_embeddings: bool = False
    ):
        """
        Initialize mock data generator.

        Args:
            base_image_dir: Directory with base chip images (required for real embeddings)
            dino_model: DiNOv2 model name for embedding extraction
            use_real_embeddings: If True, use VoidGenerator + DinoV2Embedder.
                                If False, generate random embeddings (faster for tests).
        """
        self.use_real_embeddings = use_real_embeddings
        self.dino_model = dino_model
        self.base_image_dir = Path(base_image_dir) if base_image_dir else None

        # Lazy load heavy components
        self._generator = None
        self._embedder = None

        if use_real_embeddings and base_image_dir is None:
            raise ValueError("base_image_dir is required when use_real_embeddings=True")

    @property
    def generator(self):
        """Lazy load VoidGenerator."""
        if self._generator is None:
            from .void_generator import VoidGenerator
            self._generator = VoidGenerator(self.base_image_dir)
        return self._generator

    @property
    def embedder(self):
        """Lazy load DinoV2Embedder."""
        if self._embedder is None:
            from ..embedding.dinov2_embedder import DinoV2Embedder
            self._embedder = DinoV2Embedder(model_name=self.dino_model)
        return self._embedder

    def generate_conditional_groups(
        self,
        group_specs: Optional[Dict[str, Dict]] = None,
        n_real_samples: int = 100,
        n_distributions_per_group: int = 10,
        n_samples_per_distribution: int = 10,
        embedding_dim: int = 400,
        seed: int = 42
    ) -> Tuple[np.ndarray, List[np.ndarray], List[pd.DataFrame]]:
        """
        Generate mock real embeddings and k synthetic groups.

        Each group has a DataFrame with DISTRIBUTION PARAMETERS (what Optuna optimizes),
        not sampled values. Multiple rows can have the same parameter values if they
        come from the same distribution.

        Args:
            group_specs: Optional dict mapping group names to parameter specs.
                        If None, uses DEFAULT_GROUP_SPECS (circle, ellipse, irregular).
            n_real_samples: Number of "real" samples to generate
            n_distributions_per_group: Number of unique distributions per group
            n_samples_per_distribution: Number of samples per distribution
            embedding_dim: Dimension of embeddings (default 400 for PCA-reduced)
            seed: Random seed for reproducibility

        Returns:
            real_embeddings: (n_real_samples, embedding_dim) array
            synthetic_embeddings_groups: List of k arrays, each (n_dist * n_samples, embedding_dim)
            synthetic_params_groups: List of k DataFrames with distribution parameters
                                    (columns like void_count_mean, void_count_std, etc.)
        """
        if group_specs is None:
            group_specs = self.DEFAULT_GROUP_SPECS

        np.random.seed(seed)

        if self.use_real_embeddings:
            return self._generate_with_real_embeddings(
                group_specs, n_real_samples, n_distributions_per_group,
                n_samples_per_distribution, seed
            )
        else:
            return self._generate_random_embeddings(
                group_specs, n_real_samples, n_distributions_per_group,
                n_samples_per_distribution, embedding_dim, seed
            )

    def _generate_random_embeddings(
        self,
        group_specs: Dict[str, Dict],
        n_real_samples: int,
        n_distributions_per_group: int,
        n_samples_per_distribution: int,
        embedding_dim: int,
        seed: int
    ) -> Tuple[np.ndarray, List[np.ndarray], List[pd.DataFrame]]:
        """
        Generate random embeddings for fast testing.

        Creates embeddings with different centroids per group to simulate
        shape-dependent embedding distributions.
        """
        np.random.seed(seed)

        group_names = list(group_specs.keys())
        n_groups = len(group_names)
        n_samples_per_group = n_distributions_per_group * n_samples_per_distribution

        # Generate group centroids (spread in embedding space)
        group_centroids = {}
        for i, name in enumerate(group_names):
            centroid = np.zeros(embedding_dim)
            start_idx = (i * embedding_dim) // (n_groups + 1)
            end_idx = start_idx + embedding_dim // (n_groups + 1)
            centroid[start_idx:end_idx] = 1.0
            group_centroids[name] = centroid

        # Generate "real" embeddings (mix of all groups)
        real_embeddings = []
        for i in range(n_real_samples):
            group_idx = i % n_groups
            group_name = group_names[group_idx]
            centroid = group_centroids[group_name]
            embedding = centroid + np.random.randn(embedding_dim) * 0.3
            real_embeddings.append(embedding)

        real_embeddings = np.array(real_embeddings)

        # Generate synthetic groups
        synthetic_embeddings_groups = []
        synthetic_params_groups = []

        for group_idx, (group_name, params_spec) in enumerate(group_specs.items()):
            centroid = group_centroids[group_name]

            # Generate embeddings for this group
            group_embeddings = centroid + np.random.randn(n_samples_per_group, embedding_dim) * 0.3
            synthetic_embeddings_groups.append(group_embeddings)

            # Generate distribution parameter samples for this group
            params_df = self._sample_distribution_params(
                group_name, params_spec, n_distributions_per_group,
                n_samples_per_distribution, seed + group_idx * 1000
            )
            synthetic_params_groups.append(params_df)

        return real_embeddings, synthetic_embeddings_groups, synthetic_params_groups

    def _sample_distribution_params(
        self,
        group_name: str,
        params_spec: Dict,
        n_distributions: int,
        n_samples_per_distribution: int,
        seed: int
    ) -> pd.DataFrame:
        """
        Sample DISTRIBUTION PARAMETERS (what Optuna optimizes).

        Creates a DataFrame where:
        - Columns are like: void_count_mean, void_count_std, base_size_mean, ...
        - Multiple rows can have identical values (samples from same distribution)

        Args:
            group_name: Name of the shape group
            params_spec: Dict of {param_base: {mean_bounds, std_bounds}}
            n_distributions: Number of unique distributions to generate
            n_samples_per_distribution: Number of samples per distribution
            seed: Random seed

        Returns:
            DataFrame with distribution parameters
        """
        np.random.seed(seed)

        # First generate unique distributions
        distributions = []
        for _ in range(n_distributions):
            dist_params = {}
            for param_base, spec in params_spec.items():
                mean_bounds = spec['mean_bounds']
                std_bounds = spec['std_bounds']

                # Sample mean and std uniformly within their bounds
                mean_val = np.random.uniform(mean_bounds[0], mean_bounds[1])
                std_val = np.random.uniform(std_bounds[0], std_bounds[1])

                dist_params[f'{param_base}_mean'] = mean_val
                dist_params[f'{param_base}_std'] = std_val

            distributions.append(dist_params)

        # Replicate each distribution n_samples_per_distribution times
        rows = []
        for dist_params in distributions:
            for _ in range(n_samples_per_distribution):
                rows.append(dist_params.copy())

        return pd.DataFrame(rows)

    def _generate_with_real_embeddings(
        self,
        group_specs: Dict[str, Dict],
        n_real_samples: int,
        n_distributions_per_group: int,
        n_samples_per_distribution: int,
        seed: int
    ) -> Tuple[np.ndarray, List[np.ndarray], List[pd.DataFrame]]:
        """
        Generate real embeddings using VoidGenerator + DinoV2Embedder.

        This is slower but produces realistic embeddings.
        """
        from ..embedding.pca_projector import PCAProjector

        # Generate "real" distribution (mix of shapes)
        real_params = self._generate_mixed_params_for_generator(
            group_specs, n_real_samples, seed
        )
        real_images, _ = self.generator.generate_batch(
            real_params, replications=1, seed_offset=seed
        )
        real_embeddings_768 = self.embedder.embed_batch(real_images)

        # Generate per-shape groups
        synthetic_embeddings_768_groups = []
        synthetic_params_groups = []

        for group_idx, (group_name, params_spec) in enumerate(group_specs.items()):
            # Sample distribution parameters for this group
            params_df = self._sample_distribution_params(
                group_name, params_spec, n_distributions_per_group,
                n_samples_per_distribution, seed + group_idx * 1000
            )
            synthetic_params_groups.append(params_df)

            # Convert distribution params to generator params and generate images
            generator_params = self._distribution_params_to_generator_params(
                params_df, group_name, seed + group_idx * 10000
            )
            images, _ = self.generator.generate_batch(
                generator_params, replications=1, seed_offset=seed + group_idx * 100000
            )
            embeddings_768 = self.embedder.embed_batch(images)
            synthetic_embeddings_768_groups.append(embeddings_768)

        # Fit PCA on all embeddings combined
        all_embeddings = np.vstack([real_embeddings_768] + synthetic_embeddings_768_groups)
        pca = PCAProjector(n_components=400)
        pca.fit(all_embeddings)

        # Transform all embeddings
        real_embeddings = pca.transform(real_embeddings_768)
        synthetic_embeddings_groups = [
            pca.transform(emb) for emb in synthetic_embeddings_768_groups
        ]

        return real_embeddings, synthetic_embeddings_groups, synthetic_params_groups

    def _distribution_params_to_generator_params(
        self,
        params_df: pd.DataFrame,
        group_name: str,
        seed: int
    ) -> List[Dict]:
        """
        Convert distribution parameters to VoidGenerator parameters.

        For each row (distribution params), sample actual values from the distribution
        to feed to the generator.
        """
        np.random.seed(seed)

        generator_params = []
        for _, row in params_df.iterrows():
            params = {'void_shape': group_name}

            # For each distribution parameter, sample a value
            param_bases = set()
            for col in row.index:
                if col.endswith('_mean'):
                    param_bases.add(col[:-5])  # Remove '_mean'

            for param_base in param_bases:
                mean_val = row.get(f'{param_base}_mean', 0)
                std_val = row.get(f'{param_base}_std', 1)

                # Sample from distribution
                value = np.random.normal(mean_val, std_val)

                # Apply constraints
                if param_base == 'void_count':
                    value = int(max(1, value))
                elif param_base == 'rotation':
                    value = value % 360
                elif param_base in ['center_x', 'center_y']:
                    value = np.clip(value, 0.05, 0.95)
                elif param_base == 'position_spread':
                    value = np.clip(value, 0.01, 0.99)
                elif param_base == 'complexity':
                    value = int(max(3, value))

                params[param_base] = value

            # Fill defaults for generator
            params.setdefault('void_count', 3)
            params.setdefault('base_size', 10)
            params.setdefault('rotation', 0)
            params.setdefault('center_x', 0.5)
            params.setdefault('center_y', 0.5)
            params.setdefault('position_spread', 0.4)

            generator_params.append(params)

        return generator_params

    def _generate_mixed_params_for_generator(
        self,
        group_specs: Dict[str, Dict],
        n_samples: int,
        seed: int
    ) -> List[Dict]:
        """
        Generate mixed parameters (all shapes) for "real" distribution.
        """
        np.random.seed(seed)

        group_names = list(group_specs.keys())
        params_list = []

        for i in range(n_samples):
            shape_name = group_names[i % len(group_names)]
            spec = group_specs[shape_name]

            params = {'void_shape': shape_name}

            for param_base, param_spec in spec.items():
                mean_bounds = param_spec['mean_bounds']
                std_bounds = param_spec['std_bounds']

                # Sample distribution params, then sample from distribution
                mean_val = np.random.uniform(mean_bounds[0], mean_bounds[1])
                std_val = np.random.uniform(std_bounds[0], std_bounds[1])
                value = np.random.normal(mean_val, std_val)

                # Apply constraints
                if param_base == 'void_count':
                    value = int(max(1, value))
                elif param_base == 'rotation':
                    value = value % 360
                elif param_base in ['center_x', 'center_y']:
                    value = np.clip(value, 0.05, 0.95)
                elif param_base == 'position_spread':
                    value = np.clip(value, 0.01, 0.99)
                elif param_base == 'complexity':
                    value = int(max(3, value))

                params[param_base] = value

            # Fill defaults
            params.setdefault('rotation', 0)
            params.setdefault('void_count', 3)
            params.setdefault('base_size', 10)
            params.setdefault('center_x', 0.5)
            params.setdefault('center_y', 0.5)
            params.setdefault('position_spread', 0.4)

            params_list.append(params)

        return params_list

    @staticmethod
    def get_group_param_names(group_specs: Optional[Dict] = None) -> Dict[str, List[str]]:
        """
        Get the distribution parameter column names for each group.

        Args:
            group_specs: Optional custom group specs. If None, uses DEFAULT_GROUP_SPECS.

        Returns:
            Dict mapping group name to list of parameter column names
            (e.g., ['void_count_mean', 'void_count_std', 'base_size_mean', ...])
        """
        if group_specs is None:
            group_specs = MockDataGenerator.DEFAULT_GROUP_SPECS

        result = {}
        for group_name, params in group_specs.items():
            cols = []
            for param_base in params.keys():
                cols.append(f'{param_base}_mean')
                cols.append(f'{param_base}_std')
            result[group_name] = cols

        return result


def create_test_data(
    n_real: int = 50,
    n_distributions_per_group: int = 5,
    n_samples_per_distribution: int = 10,
    embedding_dim: int = 400,
    seed: int = 42
) -> Tuple[np.ndarray, List[np.ndarray], List[pd.DataFrame], List[str]]:
    """
    Convenience function to create test data for unit tests.

    Args:
        n_real: Number of real samples
        n_distributions_per_group: Number of unique distributions per group
        n_samples_per_distribution: Number of samples per distribution
        embedding_dim: Embedding dimension
        seed: Random seed

    Returns:
        real_embeddings: (n_real, embedding_dim) array
        synthetic_embeddings_groups: List of k arrays
        synthetic_params_groups: List of k DataFrames with distribution parameters
        group_names: List of group names ['circle', 'ellipse', 'irregular']
    """
    generator = MockDataGenerator(use_real_embeddings=False)
    real_emb, synth_embs, synth_params = generator.generate_conditional_groups(
        n_real_samples=n_real,
        n_distributions_per_group=n_distributions_per_group,
        n_samples_per_distribution=n_samples_per_distribution,
        embedding_dim=embedding_dim,
        seed=seed
    )

    group_names = list(MockDataGenerator.DEFAULT_GROUP_SPECS.keys())

    return real_emb, synth_embs, synth_params, group_names
