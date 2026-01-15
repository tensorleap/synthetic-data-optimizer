import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from .void_generator import VoidGenerator
from .parameter_sampler import ParameterSampler
from ..embedding.unet_embedder import UNetEmbedder
from ..embedding.pca_projector import PCAProjector


class ProductionDataGenerator:

    def __init__(
        self,
        base_image_dir: Path,
        unet_model_path: Optional[Path] = None,
        config_path: Optional[Path] = None
    ):
        self.void_generator = VoidGenerator(base_image_dir)
        self.parameter_sampler = ParameterSampler(config_path)
        self.unet_embedder = UNetEmbedder(unet_model_path)
        self.pca_projector = None

    def generate_data_for_optimizer(
        self,
        n_distributions: int = 2,
        n_samples_per_distribution: int = 10,
        simulation_specs: Optional[Dict[str, Dict]] = None,
        target_embedding_dim: int = 400,
        seed: int = 42
    ) -> Tuple[List[np.ndarray], List[pd.DataFrame]]:
        np.random.seed(seed)

        if simulation_specs is None:
            simulation_specs = {
                'simulation_1': {
                    'void_shape': {'probabilities': {'circle': 1.0}},
                    'void_count': {'mean': 5, 'std': 2},
                    'base_size': {'mean': 10, 'std': 3},
                    'rotation': {'mean': 0, 'std': 0},
                    'center_x': {'mean': 0.5, 'std': 0.1},
                    'center_y': {'mean': 0.5, 'std': 0.1},
                    'position_spread': {'mean': 0.4, 'std': 0.1},
                },
                'simulation_2': {
                    'void_shape': {'probabilities': {'ellipse': 1.0}},
                    'void_count': {'mean': 4, 'std': 2},
                    'base_size': {'mean': 12, 'std': 3},
                    'rotation': {'mean': 45, 'std': 30},
                    'center_x': {'mean': 0.5, 'std': 0.1},
                    'center_y': {'mean': 0.5, 'std': 0.1},
                    'position_spread': {'mean': 0.3, 'std': 0.1},
                },
                'simulation_3': {
                    'void_shape': {'probabilities': {'irregular': 1.0}},
                    'void_count': {'mean': 3, 'std': 1},
                    'base_size': {'mean': 8, 'std': 2},
                    'rotation': {'mean': 0, 'std': 0},
                    'center_x': {'mean': 0.5, 'std': 0.15},
                    'center_y': {'mean': 0.5, 'std': 0.15},
                    'position_spread': {'mean': 0.5, 'std': 0.1},
                }
            }

        print(f"\n{'='*60}")
        print(f"PRODUCTION DATA GENERATOR")
        print(f"{'='*60}")
        print(f"Generating {n_distributions} distributions × {n_samples_per_distribution} samples")
        print(f"Simulations: {list(simulation_specs.keys())}")
        print(f"Target embedding dim: {target_embedding_dim}D (via PCA)")

        embeddings_per_simulation = []
        metadata_per_simulation = []
        all_embeddings_for_pca = []

        for sim_idx, (sim_name, sim_spec) in enumerate(simulation_specs.items()):
            print(f"\n[{sim_name}] Generating data...")

            param_sets = []
            for dist_idx in range(n_distributions):
                dist_seed = seed + sim_idx * 10000 + dist_idx * 100
                params_for_dist = self.parameter_sampler.sample_from_distribution_spec(
                    sim_spec,
                    n_samples=n_samples_per_distribution,
                    seed=dist_seed
                )

                for params in params_for_dist:
                    params['distribution_idx'] = dist_idx

                param_sets.extend(params_for_dist)

            print(f"  Generated {len(param_sets)} parameter sets")

            images, metadata_list = self.void_generator.generate_batch(
                param_sets,
                replications=1,
                seed_offset=seed + sim_idx * 100000
            )

            print(f"  Generated {len(images)} images")

            embeddings_raw = self.unet_embedder.embed_batch(images, verbose=True)
            print(f"  Extracted embeddings: {embeddings_raw.shape}")

            all_embeddings_for_pca.append(embeddings_raw)

            metadata_df = self._create_metadata_dataframe(param_sets)
            print(f"  Created metadata DataFrame: {metadata_df.shape}")

            embeddings_per_simulation.append(embeddings_raw)
            metadata_per_simulation.append(metadata_df)

        print(f"\n{'='*60}")
        print("Applying PCA to reduce embeddings to {target_embedding_dim}D...")
        all_embeddings_combined = np.vstack(all_embeddings_for_pca)
        print(f"  Combined embeddings shape: {all_embeddings_combined.shape}")

        self.pca_projector = PCAProjector(n_components=target_embedding_dim)
        self.pca_projector.fit_transform(all_embeddings_combined, verbose=True)

        embeddings_per_simulation_pca = []
        for emb in embeddings_per_simulation:
            emb_pca = self.pca_projector.transform(emb)
            embeddings_per_simulation_pca.append(emb_pca)

        print(f"\n{'='*60}")
        print("DATA GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Output format:")
        print(f"  - {len(embeddings_per_simulation_pca)} embedding arrays")
        for i, emb in enumerate(embeddings_per_simulation_pca):
            print(f"    [{i}] shape: {emb.shape}")
        print(f"  - {len(metadata_per_simulation)} metadata DataFrames")
        for i, df in enumerate(metadata_per_simulation):
            print(f"    [{i}] shape: {df.shape}, columns: {list(df.columns)}")

        return embeddings_per_simulation_pca, metadata_per_simulation

    def _create_metadata_dataframe(self, param_sets: List[Dict]) -> pd.DataFrame:
        rows = []
        for params in param_sets:
            row = {
                'distribution_idx': params['distribution_idx'],
                'void_count_mean': params['void_count'],
                'base_size_mean': params['base_size'],
                'rotation_mean': params.get('rotation', 0.0),
                'center_x_mean': params['center_x'],
                'center_y_mean': params['center_y'],
                'position_spread_mean': params['position_spread'],
            }
            rows.append(row)

        return pd.DataFrame(rows)
