import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from .void_generator import VoidGenerator
from .parameter_sampler import ParameterSampler


class TensorleapDataGenerator:

    def __init__(
        self,
        base_image_dir: Path,
        config_path: Optional[Path] = None
    ):
        self.void_generator = VoidGenerator(base_image_dir)
        self.parameter_sampler = ParameterSampler(config_path)

    def generate_dataset(
        self,
        output_dir: Path,
        real_distribution_type: str = 'real',
        synthetic_shapes: List[str] = None,
        synthetic_distribution_type: str = 'far',
        synthetic_shape_params: Optional[Dict[str, Dict]] = None,
        n_real_samples: int = 100,
        n_samples_per_shape: int = 30,
        n_samples_per_shape_dict: Optional[Dict[str, int]] = None,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42
    ):
        if synthetic_shapes is None:
            synthetic_shapes = ['circle', 'ellipse', 'irregular']

        np.random.seed(seed)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_split, val_split, test_split = train_val_test_split
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"

        if n_samples_per_shape_dict is None:
            n_samples_per_shape_dict = {shape: n_samples_per_shape for shape in synthetic_shapes}

        print(f"\n{'='*60}")
        print(f"TENSORLEAP DATA GENERATOR")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"Real distribution: '{real_distribution_type}' ({n_real_samples} samples)")
        print(f"Synthetic shapes:")
        for shape in synthetic_shapes:
            print(f"  {shape}: {n_samples_per_shape_dict[shape]} samples")
        print(f"Split: train={train_split:.0%}, val={val_split:.0%}, test={test_split:.0%}")

        all_metadata = []

        real_dist_spec = self.parameter_sampler.distributions[real_distribution_type]

        if synthetic_shape_params is None:
            synth_base_spec = self.parameter_sampler.distributions[synthetic_distribution_type]
            synthetic_shape_params = {}
            for shape in synthetic_shapes:
                synthetic_shape_params[shape] = synth_base_spec

        # Create split directories
        for split_name in ['train', 'val', 'test']:
            split_dir = output_dir / split_name
            split_dir.mkdir(exist_ok=True)
            (split_dir / 'real').mkdir(exist_ok=True)
            for shape in synthetic_shapes:
                (split_dir / f'synthetic_{shape}').mkdir(exist_ok=True)

        # Generate real data
        print(f"\n[REAL DATA] Generating {n_real_samples} samples...")
        real_param_sets = self.parameter_sampler.sample_parameter_sets(
            distribution_type=real_distribution_type,
            n_sets=n_real_samples,
            seed=seed
        )

        real_images, real_metadata_list = self.void_generator.generate_batch(
            real_param_sets,
            replications=1,
            seed_offset=seed
        )

        real_dist_params = self._extract_distribution_params(real_dist_spec, shape_filter=None)

        # Split real data
        n_train = int(n_real_samples * train_split)
        n_val = int(n_real_samples * val_split)
        n_test = n_real_samples - n_train - n_val

        splits = {
            'train': (0, n_train),
            'val': (n_train, n_train + n_val),
            'test': (n_train + n_val, n_real_samples)
        }

        global_idx = 0
        for split_name, (start_idx, end_idx) in splits.items():
            split_images = real_images[start_idx:end_idx]
            split_metadata = real_metadata_list[start_idx:end_idx]
            for image, img_metadata in zip(split_images, split_metadata):
                img_name = f"real_{split_name}_{global_idx:04d}.png"
                mask_name = f"real_{split_name}_{global_idx:04d}_mask.png"
                img_path = output_dir / split_name / 'real' / img_name
                mask_path = output_dir / split_name / 'real' / mask_name
                cv2.imwrite(str(img_path), image)
                cv2.imwrite(str(mask_path), img_metadata['mask'])

                metadata_row = {
                    'script_name': 'mixed',
                    'image_name': img_name,
                    'mask_name': mask_name,
                    'package_type': 'test',
                    'dataset_split': split_name,
                    **real_dist_params
                }
                all_metadata.append(metadata_row)
                global_idx += 1

        print(f"  Saved {n_real_samples} images (train={n_train}, val={n_val}, test={n_test})")

        # Generate synthetic data per shape
        print(f"\n[SYNTHETIC DATA] Generating per shape...")

        for shape_idx, shape_name in enumerate(synthetic_shapes):
            dist_name = f"synthetic_{shape_name}"
            n_samples_this_shape = n_samples_per_shape_dict[shape_name]
            print(f"  [{dist_name}] Generating {n_samples_this_shape} samples...")

            shape_base_spec = synthetic_shape_params[shape_name]
            shape_spec = self._create_shape_specific_spec(shape_base_spec, shape_name)

            dist_seed = seed + (shape_idx + 1) * 10000
            param_sets = self.parameter_sampler.sample_from_distribution_spec(
                shape_spec,
                n_samples=n_samples_this_shape,
                seed=dist_seed
            )

            images, metadata_list = self.void_generator.generate_batch(
                param_sets,
                replications=1,
                seed_offset=dist_seed
            )

            synth_dist_params = self._extract_distribution_params(shape_spec, shape_filter=shape_name)

            # Split synthetic data
            n_train_synth = int(n_samples_this_shape * train_split)
            n_val_synth = int(n_samples_this_shape * val_split)
            n_test_synth = n_samples_this_shape - n_train_synth - n_val_synth

            splits_synth = {
                'train': (0, n_train_synth),
                'val': (n_train_synth, n_train_synth + n_val_synth),
                'test': (n_train_synth + n_val_synth, n_samples_this_shape)
            }

            shape_global_idx = 0
            for split_name, (start_idx, end_idx) in splits_synth.items():
                split_images = images[start_idx:end_idx]
                split_metadata_synth = metadata_list[start_idx:end_idx]
                for image, img_metadata in zip(split_images, split_metadata_synth):
                    img_name = f"{shape_name}_{split_name}_{shape_global_idx:04d}.png"
                    mask_name = f"{shape_name}_{split_name}_{shape_global_idx:04d}_mask.png"
                    img_path = output_dir / split_name / dist_name / img_name
                    mask_path = output_dir / split_name / dist_name / mask_name
                    cv2.imwrite(str(img_path), image)
                    cv2.imwrite(str(mask_path), img_metadata['mask'])

                    metadata_row = {
                        'script_name': shape_name,
                        'image_name': img_name,
                        'mask_name': mask_name,
                        'package_type': 'test',
                        'dataset_split': split_name,
                        **synth_dist_params
                    }
                    all_metadata.append(metadata_row)
                    shape_global_idx += 1

            print(f"    Saved {n_samples_this_shape} images (train={n_train_synth}, val={n_val_synth}, test={n_test_synth})")

        metadata_df = pd.DataFrame(all_metadata)

        csv_dir = output_dir / "csv"
        csv_dir.mkdir(exist_ok=True)
        metadata_csv_path = csv_dir / "metadata.csv"
        metadata_df.to_csv(metadata_csv_path, index=False)

        assert len(metadata_df) == metadata_df['image_name'].nunique(), \
            f"Duplicate image_name found! {len(metadata_df)} rows but only {metadata_df['image_name'].nunique()} unique names"

        print(f"\n{'='*60}")
        print("DATA GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Output structure:")
        print(f"  {output_dir}/")
        print(f"    train/")
        print(f"      real/                ({n_train} images)")
        for shape in synthetic_shapes:
            print(f"      synthetic_{shape}/   ({n_train_synth} images)")
        print(f"    val/")
        print(f"      real/                ({n_val} images)")
        for shape in synthetic_shapes:
            print(f"      synthetic_{shape}/   ({n_val_synth} images)")
        print(f"    test/")
        print(f"      real/                ({n_test} images)")
        for shape in synthetic_shapes:
            print(f"      synthetic_{shape}/   ({n_test_synth} images)")
        print(f"    csv/")
        print(f"      metadata.csv       ({len(metadata_df)} rows, {metadata_df['image_name'].nunique()} unique)")
        print(f"\nMetadata columns: {list(metadata_df.columns)}")
        print(f"\nDataset ready for Tensorleap integration!")

        return metadata_df

    def generate_synthetic_only(
        self,
        output_dir: Path,
        synthetic_shapes: List[str],
        synthetic_shape_params: Dict[str, Dict],
        n_samples_per_shape_dict: Dict[str, int],
        seed: int = 42
    ):
        np.random.seed(seed)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"TENSORLEAP SYNTHETIC DATA GENERATOR (EPOCH > 0)")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"Synthetic shapes:")
        for shape in synthetic_shapes:
            print(f"  {shape}: {n_samples_per_shape_dict[shape]} samples")

        all_metadata = []

        for shape in synthetic_shapes:
            (output_dir / f'synthetic_{shape}').mkdir(exist_ok=True)

        for shape_idx, shape_name in enumerate(synthetic_shapes):
            dist_name = f"synthetic_{shape_name}"
            n_samples = n_samples_per_shape_dict[shape_name]
            print(f"\n[{dist_name}] Generating {n_samples} samples...")

            shape_spec = self._create_shape_specific_spec(synthetic_shape_params[shape_name], shape_name)

            dist_seed = seed + (shape_idx + 1) * 10000
            param_sets = self.parameter_sampler.sample_from_distribution_spec(
                shape_spec,
                n_samples=n_samples,
                seed=dist_seed
            )

            images, metadata_list = self.void_generator.generate_batch(
                param_sets,
                replications=1,
                seed_offset=dist_seed
            )

            synth_dist_params = self._extract_distribution_params(shape_spec, shape_filter=shape_name)

            for idx, (image, img_metadata) in enumerate(zip(images, metadata_list)):
                img_name = f"{shape_name}_{idx:04d}.png"
                mask_name = f"{shape_name}_{idx:04d}_mask.png"
                img_path = output_dir / dist_name / img_name
                mask_path = output_dir / dist_name / mask_name
                cv2.imwrite(str(img_path), image)
                cv2.imwrite(str(mask_path), img_metadata['mask'])

                metadata_row = {
                    'script_name': shape_name,
                    'image_name': img_name,
                    'mask_name': mask_name,
                    'package_type': 'test',
                    **synth_dist_params
                }
                all_metadata.append(metadata_row)

            print(f"  Saved {n_samples} images to {output_dir / dist_name}")

        metadata_df = pd.DataFrame(all_metadata)

        csv_dir = output_dir / "csv"
        csv_dir.mkdir(exist_ok=True)
        metadata_csv_path = csv_dir / "metadata.csv"
        metadata_df.to_csv(metadata_csv_path, index=False)

        assert len(metadata_df) == metadata_df['image_name'].nunique(), \
            f"Duplicate image_name found! {len(metadata_df)} rows but only {metadata_df['image_name'].nunique()} unique names"

        print(f"\n{'='*60}")
        print("DATA GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Output structure:")
        print(f"  {output_dir}/")
        for shape in synthetic_shapes:
            print(f"    synthetic_{shape}/   ({n_samples_per_shape_dict[shape]} images)")
        print(f"    csv/")
        print(f"      metadata.csv       ({len(metadata_df)} rows, {metadata_df['image_name'].nunique()} unique)")
        print(f"\nMetadata columns: {list(metadata_df.columns)}")
        print(f"\nDataset ready for Tensorleap integration!")

        return metadata_df

    def _create_shape_specific_spec(self, base_spec: Dict, shape: str) -> Dict:
        spec = base_spec.copy()
        spec['void_shape'] = {'probabilities': {shape: 1.0}}
        return spec

    def _extract_distribution_params(self, dist_spec: Dict, shape_filter: Optional[str]) -> Dict:
        params = {}

        for param_name in ['void_count', 'base_size', 'rotation', 'center_x', 'center_y', 'position_spread']:
            if param_name in dist_spec:
                params[f'{param_name}_mean'] = dist_spec[param_name]['mean']
                params[f'{param_name}_std'] = dist_spec[param_name]['std']

        return params
