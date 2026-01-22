import numpy as np
import cv2
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Import the generation functions from converted notebooks
sys.path.insert(0, str(Path(__file__).parent / 'simulations'))
from circular_shadow import create_outer_shadow_splatter, get_rect_coords as get_rect_coords_circular
from complex_splatter import create_splatter_void as create_complex_splatter_void, get_rect_coords as get_rect_coords_complex
from dark_bezel import create_void_effect as create_dark_bezel_void, get_rect_coords as get_rect_coords_dark_bezel
from hole_like import create_splatter_void as create_hole_like_void, get_rect_coords as get_rect_coords_hole_like
from main_splatter import create_splatter_void as create_main_splatter_void, get_rect_coords as get_rect_coords_main_splatter
from structured_void import create_void_effect_stretch_rotate as create_structured_void, get_rect_coords as get_rect_coords_structured


class InfineonVoidGenerator:

    def __init__(self, base_image_dir: Path):
        self.base_image_dir = Path(base_image_dir)
        self.base_images = self._load_base_images()

    def _load_base_images(self) -> List[Tuple[str, np.ndarray]]:
        base_images = []
        for img_path in sorted(self.base_image_dir.glob("*_result.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                base_images.append((img_path.name, img))

        if not base_images:
            raise ValueError(f"No base images found in {self.base_image_dir}")

        print(f"Loaded {len(base_images)} base chip images")
        return base_images

    def generate_single(
        self,
        params: Dict,
        seed: int,
        base_image_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        np.random.seed(seed)
        random.seed(seed)

        # Select base image
        if base_image_idx is None:
            base_image_idx = np.random.randint(0, len(self.base_images))
        base_image_name, base_image = self.base_images[base_image_idx]

        # Convert to RGB for processing
        if len(base_image.shape) == 2:
            image_rgb = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = base_image.copy()

        void_type = params['void_type']
        package_type = params.get('package_type', '53440')

        # Create temp file path for get_rect_coords (it reads the image to get dimensions)
        temp_path = self.base_image_dir / base_image_name

        # Route to appropriate generation function based on void_type
        if void_type == 'circular_shadow':
            result, mask = self._generate_circular_shadow(image_rgb, temp_path, package_type, params)
        elif void_type == 'complex_splatter':
            result, mask = self._generate_complex_splatter(image_rgb, temp_path, package_type, params)
        elif void_type == 'dark_bezel':
            result, mask = self._generate_dark_bezel(image_rgb, temp_path, package_type, params)
        elif void_type == 'hole_like':
            result, mask = self._generate_hole_like(image_rgb, temp_path, package_type, params)
        elif void_type == 'main_splatter':
            result, mask = self._generate_main_splatter(image_rgb, temp_path, package_type, params)
        elif void_type == 'structured_void':
            result, mask = self._generate_structured_void(image_rgb, temp_path, package_type, params)
        else:
            raise ValueError(f"Unknown void_type: {void_type}")

        # Convert result to grayscale
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

        # Rotate if needed (some package types require rotation)
        if result.shape == (220, 250):
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

        metadata = {
            'params': params,
            'seed': seed,
            'base_image_id': base_image_name,
            'base_image_idx': base_image_idx,
            'void_type': void_type,
            'mask': mask
        }

        return result, metadata

    def _generate_circular_shadow(self, image_rgb, image_path, package_type, params):
        rect_coords = get_rect_coords_circular(image_path, package_type)

        # Sample from the parameter ranges (each sample has different min/max values from ParameterSampler)
        num_splatters = random.randint(int(params['num_splatters_min']), int(params['num_splatters_max']))

        # Call generation function for each splatter
        result = image_rgb.copy()
        combined_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)

        for _ in range(num_splatters):
            radius = random.randint(int(params['radius_min']), int(params['radius_max']))
            irregularity = random.uniform(float(params['irregularity_min']), float(params['irregularity_max']))
            shadow_width = random.uniform(float(params['shadow_width_min']), float(params['shadow_width_max']))
            shadow_opacity = random.uniform(float(params['shadow_opacity_min']), float(params['shadow_opacity_max']))

            result_single, mask_single = create_outer_shadow_splatter(
                result,
                position=None,
                radius=radius,
                irregularity=irregularity,
                rect_coords=rect_coords,
                outer_shadow_width=shadow_width,
                outer_shadow_opacity=shadow_opacity
            )
            result = result_single
            combined_mask = np.maximum(combined_mask, mask_single)

        return result, combined_mask

    def _generate_complex_splatter(self, image_rgb, image_path, package_type, params):
        rect_coords = get_rect_coords_complex(image_path, package_type)

        # Sample parameters
        num_splatters = random.randint(params['num_splatters_min'], params['num_splatters_max'])
        min_size = random.randint(params['min_size_min'], params['min_size_max'])
        max_size = random.randint(params['max_size_min'], params['max_size_max'])
        main_darkness_range = (params['main_darkness_min'], params['main_darkness_max'])
        droplet_darkness_range = (params['droplet_darkness_min'], params['droplet_darkness_max'])
        droplet_count_range = (params['droplet_count_min'], params['droplet_count_max'])
        droplet_size_range = (params['droplet_size_min'], params['droplet_size_max'])
        droplet_distance_range = (params['droplet_distance_min'], params['droplet_distance_max'])
        gradient_direction = params['gradient_direction']

        # Call generation function (boolean params hardcoded to True)
        result, mask = create_complex_splatter_void(
            image_rgb,
            rect_coords,
            num_splatters=num_splatters,
            min_size=min_size,
            max_size=max_size,
            main_darkness_range=main_darkness_range,
            droplet_darkness_range=droplet_darkness_range,
            droplet_count_range=droplet_count_range,
            droplet_size_range=droplet_size_range,
            droplet_distance_range=droplet_distance_range,
            generate_main_splatter=True,  # Hardcoded
            generate_droplets=True,  # Hardcoded
            shadow_intensity=params.get('shadow_intensity', 0.6),
            shadow_radius_factor=params.get('shadow_radius_factor', 1.0),
            shadow_noise=params.get('shadow_noise', 0.5),
            shadow_blur=params.get('shadow_blur', 2.0),
            light_intensity=params.get('light_intensity', 0.7),
            light_radius_factor=params.get('light_radius_factor', 1.5),
            halo_noise=params.get('halo_noise', 0.5),
            gradient_direction=gradient_direction,
            droplet_blur=params.get('droplet_blur', 1.5),
            main_blur=params.get('main_blur', 2.0)
        )

        return result, mask

    def _generate_dark_bezel(self, image_rgb, image_path, package_type, params):
        rect_coords = get_rect_coords_dark_bezel(image_path, package_type)

        num_voids = random.randint(params['num_voids_min'], params['num_voids_max'])
        void_size = random.randint(params['void_size_min'], params['void_size_max'])

        result, mask = create_dark_bezel_void(
            image_path,
            rect_coords,
            num_voids=num_voids,
            void_size=void_size
        )

        return result, mask

    def _generate_hole_like(self, image_rgb, image_path, package_type, params):
        rect_coords = get_rect_coords_hole_like(image_path, package_type)

        num_splatters = random.randint(params['num_splatters_min'], params['num_splatters_max'])
        min_size = random.randint(params['min_size_min'], params['min_size_max'])
        max_size = random.randint(params['max_size_min'], params['max_size_max'])
        droplet_darkness_range = (params['droplet_darkness_min'], params['droplet_darkness_max'])
        droplet_count_range = (params['droplet_count_min'], params['droplet_count_max'])
        droplet_size_range = (params['droplet_size_min'], params['droplet_size_max'])
        droplet_distance_range = (params['droplet_distance_min'], params['droplet_distance_max'])

        result, mask = create_hole_like_void(
            image_rgb,
            rect_coords,
            num_splatters=num_splatters,
            min_size=min_size,
            max_size=max_size,
            main_darkness_range=(0.0, 0.0),  # Not used for hole-like
            droplet_darkness_range=droplet_darkness_range,
            droplet_count_range=droplet_count_range,
            droplet_size_range=droplet_size_range,
            droplet_distance_range=droplet_distance_range,
            generate_main_splatter=False,  # Hardcoded for hole-like
            generate_droplets=True  # Hardcoded
        )

        return result, mask

    def _generate_main_splatter(self, image_rgb, image_path, package_type, params):
        rect_coords = get_rect_coords_main_splatter(image_path, package_type)

        num_splatters = random.randint(params['num_splatters_min'], params['num_splatters_max'])
        min_size = random.randint(params['min_size_min'], params['min_size_max'])
        max_size = random.randint(params['max_size_min'], params['max_size_max'])
        main_darkness_range = (params['main_darkness_min'], params['main_darkness_max'])
        droplet_darkness_range = (params['droplet_darkness_min'], params['droplet_darkness_max'])
        droplet_count_range = (params['droplet_count_min'], params['droplet_count_max'])
        droplet_size_range = (params['droplet_size_min'], params['droplet_size_max'])
        droplet_distance_range = (params['droplet_distance_min'], params['droplet_distance_max'])

        result, mask = create_main_splatter_void(
            image_rgb,
            rect_coords,
            num_splatters=num_splatters,
            min_size=min_size,
            max_size=max_size,
            main_darkness_range=main_darkness_range,
            droplet_darkness_range=droplet_darkness_range,
            droplet_count_range=droplet_count_range,
            droplet_size_range=droplet_size_range,
            droplet_distance_range=droplet_distance_range,
            generate_main_splatter=True,  # Hardcoded
            generate_droplets=True  # Hardcoded
        )

        return result, mask

    def _generate_structured_void(self, image_rgb, image_path, package_type, params):
        rect_coords = get_rect_coords_structured(image_path, package_type)

        num_voids = random.randint(params['num_voids_min'], params['num_voids_max'])
        void_size = random.randint(params['void_size_min'], params['void_size_max'])
        direction = params.get('direction', 'random')
        stretch_min = params['stretch_min']
        stretch_max = params['stretch_max']
        darkness_min = params['darkness_min']
        darkness_max = params['darkness_max']

        result, mask = create_structured_void(
            image_path,
            rect_coords,
            num_voids=num_voids,
            void_size=void_size,
            direction=direction,
            stretch_range=(stretch_min, stretch_max),
            enable_random_halo=True,  # Hardcoded
            enable_random_structure=True,  # Hardcoded
            darkness_range=(darkness_min, darkness_max)
        )

        return result, mask

    def generate_batch(
        self,
        param_sets: List[Dict],
        replications: int = 1,
        save_dir: Optional[Path] = None,
        seed_offset: int = 0
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        images = []
        metadata_list = []

        for param_idx, params in enumerate(param_sets):
            for rep in range(replications):
                seed = seed_offset + param_idx * replications + rep
                img, metadata = self.generate_single(params, seed)

                # Add tracking IDs
                if 'distribution_id' in params:
                    distribution_id = params['distribution_id']
                else:
                    distribution_id = param_idx

                metadata['distribution_id'] = distribution_id
                metadata['replication_id'] = rep
                metadata['sample_id'] = f"dist_{distribution_id:03d}_rep{rep}"

                images.append(img)
                metadata_list.append(metadata)

                # Save if directory provided
                if save_dir:
                    save_dir = Path(save_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    img_path = save_dir / f"{metadata['sample_id']}.png"
                    cv2.imwrite(str(img_path), img)

        return images, metadata_list
