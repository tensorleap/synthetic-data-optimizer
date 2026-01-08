"""
Mock data generator for testing the conditional optimizer.

Generates synthetic embeddings and parameters for k conditional groups (e.g., shapes).
Each group can have DIFFERENT parameters - this is the key use case for conditional optimization.

Example:
- circle: void_count, base_size, center_x, center_y, position_spread
- ellipse: void_count, base_size, rotation, center_x, center_y, position_spread
- irregular: void_count, base_size, complexity, center_x, center_y, position_spread

Uses existing VoidGenerator + DinoV2Embedder pipeline or can generate random embeddings
for faster testing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class MockDataGenerator:
    """
    Generate mock data for testing the conditional parameters optimizer.

    Each shape group has DIFFERENT parameters, demonstrating the conditional parameter use case:
    - circle: No rotation (circles are rotation-invariant)
    - ellipse: Has rotation parameter
    - irregular: No rotation, has complexity parameter instead

    Can operate in two modes:
    1. Full pipeline mode: Uses VoidGenerator + DinoV2Embedder (slower, real embeddings)
    2. Random mode: Generates random embeddings (faster for unit tests)
    """

    # Default parameter specs for each shape group
    # NOTE: Different shapes have DIFFERENT parameter sets
    DEFAULT_GROUP_SPECS = {
        'circle': {
            # Circles don't have rotation (rotation-invariant)
            'void_count': {'mean': 5, 'std': 2, 'bounds': [1, 15]},
            'base_size': {'mean': 10, 'std': 3, 'bounds': [3, 20]},
            'center_x': {'mean': 0.5, 'std': 0.15, 'bounds': [0.1, 0.9]},
            'center_y': {'mean': 0.5, 'std': 0.15, 'bounds': [0.1, 0.9]},
            'position_spread': {'mean': 0.4, 'std': 0.1, 'bounds': [0.1, 0.8]},
        },
        'ellipse': {
            # Ellipses HAVE rotation - unique parameter for this shape
            'void_count': {'mean': 4, 'std': 2, 'bounds': [1, 12]},
            'base_size': {'mean': 12, 'std': 4, 'bounds': [4, 25]},
            'rotation': {'mean': 90, 'std': 45, 'bounds': [0, 360]},  # Only ellipse has rotation
            'center_x': {'mean': 0.5, 'std': 0.2, 'bounds': [0.15, 0.85]},
            'center_y': {'mean': 0.5, 'std': 0.2, 'bounds': [0.15, 0.85]},
            'position_spread': {'mean': 0.5, 'std': 0.15, 'bounds': [0.1, 0.7]},
        },
        'irregular': {
            # Irregular shapes have complexity instead of rotation
            'void_count': {'mean': 3, 'std': 1, 'bounds': [1, 8]},
            'base_size': {'mean': 8, 'std': 2, 'bounds': [2, 15]},
            'complexity': {'mean': 8, 'std': 2, 'bounds': [5, 12]},  # Number of polygon points
            'center_x': {'mean': 0.5, 'std': 0.1, 'bounds': [0.2, 0.8]},
            'center_y': {'mean': 0.5, 'std': 0.1, 'bounds': [0.2, 0.8]},
            'position_spread': {'mean': 0.3, 'std': 0.1, 'bounds': [0.05, 0.6]},
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
        n_samples_per_group: int = 100,
        embedding_dim: int = 400,
        seed: int = 42
    ) -> Tuple[np.ndarray, List[np.ndarray], List[pd.DataFrame]]:
        """
        Generate mock real embeddings and k synthetic groups.

        Each group may have DIFFERENT parameters (e.g., ellipse has rotation, circle doesn't).

        Args:
            group_specs: Optional dict mapping group names to parameter specs.
                        If None, uses DEFAULT_GROUP_SPECS (circle, ellipse, irregular).
                        Format: {group_name: {param_name: {mean, std, bounds}, ...}}
            n_real_samples: Number of "real" samples to generate
            n_samples_per_group: Number of samples per conditional group
            embedding_dim: Dimension of embeddings (default 400 for PCA-reduced)
            seed: Random seed for reproducibility

        Returns:
            real_embeddings: (n_real_samples, embedding_dim) array
            synthetic_embeddings_groups: List of k arrays, each (n_samples_per_group, embedding_dim)
            synthetic_params_groups: List of k DataFrames, each with different columns per group!
        """
        if group_specs is None:
            group_specs = self.DEFAULT_GROUP_SPECS

        np.random.seed(seed)

        if self.use_real_embeddings:
            return self._generate_with_real_embeddings(
                group_specs, n_real_samples, n_samples_per_group, seed
            )
        else:
            return self._generate_random_embeddings(
                group_specs, n_real_samples, n_samples_per_group, embedding_dim, seed
            )

    def _generate_random_embeddings(
        self,
        group_specs: Dict[str, Dict],
        n_real_samples: int,
        n_samples_per_group: int,
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

        # Generate group centroids (spread in embedding space)
        group_centroids = {}
        for i, name in enumerate(group_names):
            # Spread centroids along different directions in embedding space
            centroid = np.zeros(embedding_dim)
            # Each group has a unique subspace signature
            start_idx = (i * embedding_dim) // (n_groups + 1)
            end_idx = start_idx + embedding_dim // (n_groups + 1)
            centroid[start_idx:end_idx] = 1.0
            group_centroids[name] = centroid

        # Generate "real" embeddings (mix of all groups)
        real_embeddings = []
        for i in range(n_real_samples):
            # Random group assignment for real data
            group_idx = i % n_groups
            group_name = group_names[group_idx]
            centroid = group_centroids[group_name]

            # Add noise around centroid
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

            # Generate parameter samples for this group
            # NOTE: Each group has DIFFERENT columns based on its params_spec
            params_df = self._sample_params_from_spec(
                group_name, params_spec, n_samples_per_group, seed + group_idx * 1000
            )
            synthetic_params_groups.append(params_df)

        return real_embeddings, synthetic_embeddings_groups, synthetic_params_groups

    def _generate_with_real_embeddings(
        self,
        group_specs: Dict[str, Dict],
        n_real_samples: int,
        n_samples_per_group: int,
        seed: int
    ) -> Tuple[np.ndarray, List[np.ndarray], List[pd.DataFrame]]:
        """
        Generate real embeddings using VoidGenerator + DinoV2Embedder.

        This is slower but produces realistic embeddings.
        """
        from ..embedding.pca_projector import PCAProjector

        # Generate "real" distribution (mix of shapes)
        real_params = self._generate_mixed_params_for_generator(group_specs, n_real_samples, seed)
        real_images, _ = self.generator.generate_batch(real_params, replications=1, seed_offset=seed)
        real_embeddings_768 = self.embedder.embed_batch(real_images)

        # Generate per-shape groups
        synthetic_embeddings_768_groups = []
        synthetic_params_groups = []

        for group_idx, (group_name, params_spec) in enumerate(group_specs.items()):
            # Sample parameters for this group
            params_df = self._sample_params_from_spec(
                group_name, params_spec, n_samples_per_group, seed + group_idx * 1000
            )
            synthetic_params_groups.append(params_df)

            # Convert DataFrame to list of dicts for generator
            # Need to add void_shape and handle missing params for the generator
            param_dicts = self._prepare_params_for_generator(params_df, group_name)

            # Generate images and embeddings
            images, _ = self.generator.generate_batch(
                param_dicts, replications=1, seed_offset=seed + group_idx * 10000
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

    def _sample_params_from_spec(
        self,
        shape_name: str,
        params_spec: Dict,
        n_samples: int,
        seed: int
    ) -> pd.DataFrame:
        """
        Sample parameters from a specification.

        Each shape has its own set of parameters - the returned DataFrame
        will have DIFFERENT columns for different shapes.

        Args:
            shape_name: Name of the shape (stored in void_shape column)
            params_spec: Dict of {param_name: {mean, std, bounds}}
            n_samples: Number of samples to generate
            seed: Random seed

        Returns:
            DataFrame with sampled parameters (columns vary by shape!)
        """
        np.random.seed(seed)

        # void_shape is always present
        data = {'void_shape': [shape_name] * n_samples}

        for param_name, spec in params_spec.items():
            mean = spec.get('mean', 0)
            std = spec.get('std', 1)
            bounds = spec.get('bounds', [float('-inf'), float('inf')])

            values = np.random.normal(mean, std, n_samples)
            values = np.clip(values, bounds[0], bounds[1])

            # Integer parameters
            if param_name in ['void_count', 'complexity']:
                values = values.astype(int)

            data[param_name] = values

        return pd.DataFrame(data)

    def _prepare_params_for_generator(
        self,
        params_df: pd.DataFrame,
        shape_name: str
    ) -> List[Dict]:
        """
        Prepare parameters for VoidGenerator.

        The generator expects specific parameters. This fills in defaults
        for parameters not present in the shape's spec.
        """
        # VoidGenerator expects these params
        required_params = {
            'void_shape': shape_name,
            'void_count': 3,
            'base_size': 10,
            'rotation': 0,  # Default for non-ellipse shapes
            'center_x': 0.5,
            'center_y': 0.5,
            'position_spread': 0.4
        }

        param_dicts = []
        for _, row in params_df.iterrows():
            params = required_params.copy()
            # Override with actual values from DataFrame
            for col in params_df.columns:
                if col in params:
                    params[col] = row[col]
            param_dicts.append(params)

        return param_dicts

    def _generate_mixed_params_for_generator(
        self,
        group_specs: Dict[str, Dict],
        n_samples: int,
        seed: int
    ) -> List[Dict]:
        """
        Generate mixed parameters (all shapes) for "real" distribution.

        Cycles through shapes to create diverse samples.
        """
        np.random.seed(seed)

        group_names = list(group_specs.keys())
        params_list = []

        for i in range(n_samples):
            # Cycle through shapes
            shape_name = group_names[i % len(group_names)]
            spec = group_specs[shape_name]

            # Start with defaults
            params = {
                'void_shape': shape_name,
                'void_count': 3,
                'base_size': 10,
                'rotation': 0,
                'center_x': 0.5,
                'center_y': 0.5,
                'position_spread': 0.4
            }

            # Sample from spec
            for param_name, param_spec in spec.items():
                mean = param_spec.get('mean', 0)
                std = param_spec.get('std', 1)
                bounds = param_spec.get('bounds', [float('-inf'), float('inf')])

                value = np.random.normal(mean, std)
                value = np.clip(value, bounds[0], bounds[1])

                if param_name in ['void_count', 'complexity']:
                    value = int(value)

                params[param_name] = value

            params_list.append(params)

        return params_list

    @staticmethod
    def get_group_param_names(group_specs: Optional[Dict] = None) -> Dict[str, List[str]]:
        """
        Get the parameter names for each group.

        Args:
            group_specs: Optional custom group specs. If None, uses DEFAULT_GROUP_SPECS.

        Returns:
            Dict mapping group name to list of parameter names (excluding void_shape)
        """
        if group_specs is None:
            group_specs = MockDataGenerator.DEFAULT_GROUP_SPECS

        return {
            group_name: list(params.keys())
            for group_name, params in group_specs.items()
        }


def create_test_data(
    n_real: int = 50,
    n_per_group: int = 50,
    embedding_dim: int = 400,
    seed: int = 42
) -> Tuple[np.ndarray, List[np.ndarray], List[pd.DataFrame], List[str]]:
    """
    Convenience function to create test data for unit tests.

    Args:
        n_real: Number of real samples
        n_per_group: Number of samples per conditional group
        embedding_dim: Embedding dimension
        seed: Random seed

    Returns:
        real_embeddings: (n_real, embedding_dim) array
        synthetic_embeddings_groups: List of k arrays
        synthetic_params_groups: List of k DataFrames (with different columns per shape!)
        group_names: List of group names ['circle', 'ellipse', 'irregular']
    """
    generator = MockDataGenerator(use_real_embeddings=False)
    real_emb, synth_embs, synth_params = generator.generate_conditional_groups(
        n_real_samples=n_real,
        n_samples_per_group=n_per_group,
        embedding_dim=embedding_dim,
        seed=seed
    )

    group_names = list(MockDataGenerator.DEFAULT_GROUP_SPECS.keys())

    return real_emb, synth_embs, synth_params, group_names
