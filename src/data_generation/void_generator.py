"""
Void generator for creating synthetic chip images with voids.

Overlays voids on clean chip images based on controllable parameters.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class VoidGenerator:
    """Generate synthetic void images by overlaying on base chips"""

    def __init__(self, base_image_dir: Path):
        """
        Initialize void generator with base chip images.

        Args:
            base_image_dir: Directory containing clean chip images
        """
        self.base_image_dir = Path(base_image_dir)
        self.base_images = self._load_base_images()

    def _load_base_images(self) -> List[Tuple[str, np.ndarray]]:
        """Load all base chip images (excluding masks)"""
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
        """
        Generate a single void image.

        Args:
            params: Parameter dictionary with:
                - void_shape: 'circle', 'ellipse', or 'irregular'
                - void_count: number of voids to generate
                - base_size: base void size in pixels (with distribution)
                - rotation: rotation angle in degrees for ellipses
                - center_x: x-coordinate of void cluster center (fraction of width)
                - center_y: y-coordinate of void cluster center (fraction of height)
                - position_spread: 0.0-1.0, fraction of image for position distribution around center
            seed: Random seed for reproducibility
            base_image_idx: Optional index to select specific base image, otherwise random

        Returns:
            image: Generated image with voids
            metadata: Dictionary with generation details
        """
        np.random.seed(seed)

        # Select base image
        if base_image_idx is None:
            base_image_idx = np.random.randint(0, len(self.base_images))
        base_image_name, base_image = self.base_images[base_image_idx]

        # Create working copy
        img = base_image.copy().astype(np.float32)
        height, width = img.shape

        # Extract controlled parameters
        void_shape = params['void_shape']
        void_count = params['void_count']
        base_size = params['base_size']
        rotation = params['rotation']
        center_x = params['center_x']
        center_y = params['center_y']
        position_spread = params['position_spread']

        # Uncontrolled parameters (random per image)
        brightness_factor = np.random.uniform(0.3, 0.8)
        edge_blur = np.random.randint(1, 5)  # random edge blur [1, 4]

        # Create unified void mask - all voids will have same intensity
        unified_void_mask = np.zeros(img.shape, dtype=np.float32)

        # Generate voids and union them into single mask
        void_metadata = []
        for i in range(void_count):
            # Use base_size directly (no size_std anymore)
            void_size = max(1, base_size)

            # Calculate maximum radius considering shape variations
            # Irregular shapes can vary up to 1.5x the size
            # Ellipses can have aspect ratio up to 1.5x the size
            # Add safety margin for edge blur
            max_radius = void_size * 1.5 + edge_blur

            # Convert center fractions to pixel coordinates
            # center_x and center_y are now controlled parameters (0.0-1.0)
            cluster_center_x = center_x * width
            cluster_center_y = center_y * height

            # Calculate spread range (from cluster center)
            spread_x = (width / 2) * position_spread
            spread_y = (height / 2) * position_spread

            # Sample position within spread area, keeping voids within boundaries
            # Add margin equal to max_radius from all edges
            x_min = max(cluster_center_x - spread_x, max_radius)
            x_max = min(cluster_center_x + spread_x, width - max_radius)
            y_min = max(cluster_center_y - spread_y, max_radius)
            y_max = min(cluster_center_y + spread_y, height - max_radius)

            # Ensure valid range (in case spread is too large or size is too big)
            if x_max <= x_min:
                x_min = max_radius
                x_max = width - max_radius
            if y_max <= y_min:
                y_min = max_radius
                y_max = height - max_radius

            # Sample position
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)

            # Draw void based on shape
            if void_shape == 'circle':
                void_mask = self._create_circle_mask(
                    img.shape, x, y, void_size, edge_blur
                )
            elif void_shape == 'ellipse':
                void_mask = self._create_ellipse_mask(
                    img.shape, x, y, void_size, edge_blur, rotation
                )
            elif void_shape == 'irregular':
                void_mask = self._create_irregular_mask(
                    img.shape, x, y, void_size, edge_blur
                )
            else:
                raise ValueError(f"Unknown void_shape: {void_shape}")

            # Union operation: take maximum to merge voids with smooth blending
            unified_void_mask = np.maximum(unified_void_mask, void_mask)

            void_metadata.append({
                'void_id': i,
                'x': float(x),
                'y': float(y),
                'size': float(void_size),
                'shape': void_shape
            })

        # Apply single unified void mask to image - all voids same intensity
        img = img * (1 - unified_void_mask * (1 - brightness_factor))

        # Clip to valid range and convert back to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Build metadata
        metadata = {
            'params': params,
            'seed': seed,
            'base_image_id': base_image_name,
            'base_image_idx': base_image_idx,
            'uncontrolled_params': {
                'brightness_factor': float(brightness_factor),
                'edge_blur': int(edge_blur)
            },
            'voids': void_metadata,
            'image_shape': (height, width)
        }

        return img, metadata

    def _create_circle_mask(
        self,
        shape: Tuple[int, int],
        x: float,
        y: float,
        size: float,
        edge_blur: int
    ) -> np.ndarray:
        """Create circular void mask (0-1 float)"""
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(
            mask,
            (int(x), int(y)),
            int(size),
            255,
            -1  # filled
        )

        # Apply edge blur for softer edges
        if edge_blur > 0:
            mask = cv2.GaussianBlur(mask, (edge_blur * 2 + 1, edge_blur * 2 + 1), 0)

        return mask.astype(np.float32) / 255.0

    def _create_ellipse_mask(
        self,
        shape: Tuple[int, int],
        x: float,
        y: float,
        size: float,
        edge_blur: int,
        rotation: float
    ) -> np.ndarray:
        """Create elliptical void mask (0-1 float)"""
        mask = np.zeros(shape, dtype=np.uint8)
        aspect_ratio = np.random.uniform(0.5, 1.5)
        axes = (int(size), int(size * aspect_ratio))

        cv2.ellipse(
            mask,
            (int(x), int(y)),
            axes,
            rotation,
            0, 360,
            255,
            -1  # filled
        )

        # Apply edge blur for softer edges
        if edge_blur > 0:
            mask = cv2.GaussianBlur(mask, (edge_blur * 2 + 1, edge_blur * 2 + 1), 0)

        return mask.astype(np.float32) / 255.0

    def _create_irregular_mask(
        self,
        shape: Tuple[int, int],
        x: float,
        y: float,
        size: float,
        edge_blur: int
    ) -> np.ndarray:
        """Create irregular blob-like void mask (0-1 float)"""
        mask = np.zeros(shape, dtype=np.uint8)

        # Generate random polygon points around center
        num_points = np.random.randint(5, 12)
        angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))

        points = []
        for angle in angles:
            # Vary radius to create irregular shape
            radius = size * np.random.uniform(0.5, 1.5)
            px = int(x + radius * np.cos(angle))
            py = int(y + radius * np.sin(angle))
            points.append([px, py])

        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

        # Apply edge blur for softer edges
        if edge_blur > 0:
            mask = cv2.GaussianBlur(mask, (edge_blur * 2 + 1, edge_blur * 2 + 1), 0)

        return mask.astype(np.float32) / 255.0

    def generate_batch(
        self,
        param_sets: List[Dict],
        replications: int = 3,
        save_dir: Optional[Path] = None,
        seed_offset: int = 0
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Generate batch of images from parameter sets.

        Args:
            param_sets: List of parameter dictionaries
            replications: Number of samples per parameter set (different seeds)
            save_dir: Optional directory to save images
            seed_offset: Offset added to calculated seed (for reproducibility across batches)

        Returns:
            images: List of generated images
            metadata_list: List of metadata dictionaries
        """
        images = []
        metadata_list = []

        for param_idx, params in enumerate(param_sets):
            for rep in range(replications):
                seed = seed_offset + param_idx * replications + rep
                img, metadata = self.generate_single(params, seed)

                # Add tracking IDs
                # Use distribution_id if present (for distribution optimization), otherwise use index
                if 'distribution_id' in params:
                    param_set_id = f"dist_{params['distribution_id']:03d}"
                else:
                    param_set_id = f"ps_{param_idx:03d}"

                metadata['param_set_id'] = param_set_id
                metadata['replication_id'] = rep
                metadata['sample_id'] = f"{param_set_id}_rep{rep}"

                images.append(img)
                metadata_list.append(metadata)

                # Save if directory provided
                if save_dir:
                    save_dir = Path(save_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    img_path = save_dir / f"{metadata['sample_id']}.png"
                    cv2.imwrite(str(img_path), img)

        return images, metadata_list
