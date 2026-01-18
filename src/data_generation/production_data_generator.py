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
        real_distribution_type: str = 'real',
        synthetic_distribution_type: str = 'far',
        n_real_samples: int = 100,
        n_distributions: int = 2,
        n_samples_per_distribution: int = 15,
        target_embedding_dim: int = 400,
        seed: int = 42
    ) -> Tuple[np.ndarray, List[np.ndarray], List[pd.DataFrame]]:
        np.random.seed(seed)

        print(f"\n{'='*60}")
        print(f"PRODUCTION DATA GENERATOR")
        print(f"{'='*60}")
        print(f"Real distribution: '{real_distribution_type}' ({n_real_samples} samples)")
        print(f"Synthetic distribution: '{synthetic_distribution_type}' ({n_distributions} dists × {n_samples_per_distribution} samples)")
        print(f"Target embedding dim: {target_embedding_dim}D (via PCA)")

        print(f"\n[REAL DATA] Generating...")
        real_param_sets = self.parameter_sampler.sample_parameter_sets(
            distribution_type=real_distribution_type,
            n_sets=n_real_samples,
            seed=seed
        )

        real_images, _ = self.void_generator.generate_batch(
            real_param_sets,
            replications=1,
            seed_offset=seed
        )
        print(f"  Generated {len(real_images)} real images")

        real_embeddings_raw = self.unet_embedder.embed_batch(real_images, verbose=True)
        print(f"  Extracted embeddings: {real_embeddings_raw.shape}")

        print(f"\n[SYNTHETIC DATA] Generating per simulation type...")
        embeddings_per_simulation = []
        metadata_per_simulation = []
        all_synthetic_embeddings = []

        simulation_names = ['simulation_1', 'simulation_2', 'simulation_3']

        for sim_idx, sim_name in enumerate(simulation_names):
            print(f"\n  [{sim_name}]")

            param_sets = []
            for dist_idx in range(n_distributions):
                dist_seed = seed + sim_idx * 10000 + dist_idx * 100
                params_for_dist = self.parameter_sampler.sample_parameter_sets(
                    distribution_type=synthetic_distribution_type,
                    n_sets=n_samples_per_distribution,
                    seed=dist_seed
                )

                for params in params_for_dist:
                    params['distribution_idx'] = dist_idx

                param_sets.extend(params_for_dist)

            print(f"    Generated {len(param_sets)} parameter sets")

            images, metadata_list = self.void_generator.generate_batch(
                param_sets,
                replications=1,
                seed_offset=seed + sim_idx * 100000
            )
            print(f"    Generated {len(images)} images")

            embeddings_raw = self.unet_embedder.embed_batch(images, verbose=False)
            print(f"    Extracted embeddings: {embeddings_raw.shape}")

            all_synthetic_embeddings.append(embeddings_raw)

            metadata_df = self._create_metadata_dataframe(param_sets)
            print(f"    Created metadata DataFrame: {metadata_df.shape}")

            embeddings_per_simulation.append(embeddings_raw)
            metadata_per_simulation.append(metadata_df)

        print(f"\n{'='*60}")
        print(f"Applying PCA to reduce embeddings to {target_embedding_dim}D...")

        all_embeddings_combined = np.vstack([real_embeddings_raw] + all_synthetic_embeddings)
        print(f"  Combined embeddings shape: {all_embeddings_combined.shape}")

        self.pca_projector = PCAProjector(n_components=target_embedding_dim)
        self.pca_projector.fit_transform(all_embeddings_combined, verbose=True)

        real_embeddings_pca = self.pca_projector.transform(real_embeddings_raw)
        print(f"  Real embeddings (PCA): {real_embeddings_pca.shape}")

        embeddings_per_simulation_pca = []
        for sim_name, emb in zip(simulation_names, embeddings_per_simulation):
            emb_pca = self.pca_projector.transform(emb)
            print(f"  {sim_name} embeddings (PCA): {emb_pca.shape}")
            embeddings_per_simulation_pca.append(emb_pca)

        print(f"\n{'='*60}")
        print("DATA GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Output:")
        print(f"  - Real embeddings: {real_embeddings_pca.shape}")
        print(f"  - {len(embeddings_per_simulation_pca)} synthetic embedding arrays")
        print(f"  - {len(metadata_per_simulation)} metadata DataFrames")

        return real_embeddings_pca, embeddings_per_simulation_pca, metadata_per_simulation

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
