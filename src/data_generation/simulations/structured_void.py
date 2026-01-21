import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from pathlib import Path
import random
from tqdm import tqdm


# ===========================
# Configuration Parameters
# ===========================

NAME_PREFIX = "Voids_structNoise_stretch_rotate"
NUM_SPLATTERS = 2
MIN_SIZE = 20
MAX_SIZE = 25
NUMBER_OF_SYNTHETIC_IMAGES = 500
PACKAGE_TYPE = "53440"  # Package type identifier

# Source directory containing good unit images
SOURCE_DIR = Path(fr"U:\Easy_Panda\Datasets\Train_images_500_random_{PACKAGE_TYPE}\extracted_chips_template_matching\TestSet\GoodUnits")

# Destination directory for synthetic defect images
DESTINATION_DIR = Path(fr"Train_images_500_random_{PACKAGE_TYPE}\extracted_chips_template_matching\Synth_Data_{NAME_PREFIX}")




def generate_noise_texture(shape, scale, octaves=4, persistence=0.5):
    """
    Generate multi-octave noise texture for realistic surface patterns.
    
    Args:
        shape (tuple): Dimensions of the noise texture (height, width)
        scale (float): Base scale for noise features
        octaves (int): Number of noise layers to combine
        persistence (float): Amplitude decay factor for each octave
    
    Returns:
        numpy.ndarray: Normalized noise texture in range [0, 1]
    """
    noise = np.zeros(shape, dtype=np.float32)
    total_amplitude = 0
    frequency = 1
    amplitude = 1

    # Combine multiple noise layers at different frequencies
    for _ in range(octaves):
        noise_layer = gaussian_filter(np.random.normal(0, 1, shape), sigma=scale/frequency)
        noise += amplitude * noise_layer
        total_amplitude += amplitude
        frequency *= 2
        amplitude *= persistence

    # Normalize to [0, 1] range
    return (noise / total_amplitude - noise.min()) / (noise.max() - noise.min())


def generate_perlin_noise(shape, scale=4, octaves=3, persistence=0.1):
    """
    Generate Perlin-like noise for texture variation.
    
    Args:
        shape (tuple): Dimensions of the noise texture (height, width)
        scale (float): Base scale for noise features
        octaves (int): Number of noise layers to combine
        persistence (float): Amplitude decay factor for each octave
    
    Returns:
        numpy.ndarray: Normalized Perlin noise in range [0, 1]
    """
    noise = np.zeros(shape, dtype=np.float32)
    frequency = 1
    amplitude = 1
    total_amplitude = 0
    
    # Combine multiple octaves of noise
    for _ in range(octaves):
        noise_layer = np.random.normal(0, 1, shape)
        noise_layer = gaussian_filter(noise_layer, sigma=scale/frequency)
        noise += amplitude * noise_layer
        total_amplitude += amplitude
        frequency *= 2
        amplitude *= persistence
    
    # Normalize to [0, 1] range
    noise = (noise / total_amplitude)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    return noise


def create_cloud_patches(mask, size, num_clouds=3):
    """
    Create small cloud-like dark patches within the mask region.
    
    Args:
        mask (numpy.ndarray): Base mask where clouds will be generated
        size (int): Approximate size of the void (used to scale clouds)
        num_clouds (int): Number of cloud patterns to generate
    
    Returns:
        numpy.ndarray: Cloud pattern mask with values in [0, 1]
    """
    height, width = mask.shape
    cloud_mask = np.zeros_like(mask)
    
    # Find valid positions within the mask
    valid_positions = np.where(mask > 0.5)
    if len(valid_positions[0]) == 0:
        return cloud_mask
    
    # Generate each cloud patch
    for _ in range(num_clouds):
        # Select random center point within valid positions
        idx = np.random.randint(0, len(valid_positions[0]))
        cy = valid_positions[0][idx]
        cx = valid_positions[1][idx]
        
        # Generate cloud shape with irregular edges
        cloud_size = int(size * np.random.uniform(0.2, 0.4))
        num_points = np.random.randint(8, 15)
        angles = np.linspace(0, 2*np.pi, num_points)
        radii = cloud_size * (0.5 + 0.5 * np.random.rand(num_points))
        
        # Create cloud points
        points = []
        for angle, radius in zip(angles, radii):
            px = cx + int(radius * np.cos(angle))
            py = cy + int(radius * np.sin(angle))
            points.append([[px, py]])
        
        points = np.array(points, dtype=np.int32)
        
        # Draw cloud and apply gaussian blur for soft edges
        temp_cloud = np.zeros_like(mask)
        cv2.fillPoly(temp_cloud, [points], 0.5)
        temp_cloud = gaussian_filter(temp_cloud, sigma=2)
        
        # Accumulate clouds
        cloud_mask = np.maximum(cloud_mask, temp_cloud)
    
    # Apply only within the original mask
    cloud_mask *= mask
    
    return cloud_mask


def create_void_effect_stretch_rotate(
    image_path, rect_coords, num_voids=2, void_size=50, direction='random', 
    stretch_range=(2.5, 4.5), enable_random_halo=True, enable_random_structure=True, 
    darkness_range=(0.45, 0.55)
):
    """
    Create stretched and rotated structured void defects with optional halo and internal structure.
    
    Args:
        image_path (Path): Path to the input image file
        rect_coords (list): Bounding box [x1, y1, x2, y2] for void placement
        num_voids (int): Number of void defects to create
        void_size (int): Approximate size of each void
        direction (str): Stretch direction ('vertical', 'horizontal', or 'random')
        stretch_range (tuple): Range of stretch factors (min, max)
        enable_random_halo (bool): Whether to randomly enable halo effect
        enable_random_structure (bool): Whether to randomly enable internal structure
        darkness_range (tuple): Range of darkness values (min, max) for voids (0-1)
    
    Returns:
        tuple: (result_image, binary_mask)
            - result_image: Modified image with void effects (RGB format)
            - binary_mask: Binary mask of all void regions (uint8)
    
    Raises:
        ValueError: If image cannot be read
    """
    # Load and validate image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Could not read image")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = rect_coords

    # Create rectangle mask for bounding
    rect_mask = np.zeros((height, width), dtype=np.float32)
    rect_mask[y1:y2, x1:x2] = 1.0

    def generate_void_mask(x, y, size, direction):
        """
        Generate a stretched and rotated void mask.
        
        Returns:
            numpy.ndarray: Void mask with values in [0, 1]
        """
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Create irregular shape
        num_points = np.random.randint(15, 30)
        angles = np.linspace(0, 2*np.pi, num_points)
        radii = size * (0.5 + 0.5 * np.random.rand(num_points))
        
        # Determine stretch factors based on direction
        if direction == 'vertical':
            width_factor, height_factor = 1.0, np.random.uniform(*stretch_range)
        elif direction == 'horizontal':
            width_factor, height_factor = np.random.uniform(*stretch_range), 1.0
        else:  # random
            if np.random.rand() < 0.5:
                width_factor, height_factor = 1.0, np.random.uniform(*stretch_range)
            else:
                width_factor, height_factor = np.random.uniform(*stretch_range), 1.0
        
        # Compensate for area increase due to stretching
        compensate = 1.0 / np.sqrt(max(width_factor, height_factor))
        width_factor *= compensate
        height_factor *= compensate
        
        # Create stretched points
        points = []
        for angle, radius in zip(angles, radii):
            px = x + int(radius * np.cos(angle) * width_factor)
            py = y + int(radius * np.sin(angle) * height_factor)
            points.append([[px, py]])
        
        points = np.array(points, dtype=np.int32)
        
        # Fill and smooth the mask
        cv2.fillPoly(mask, [points], 1.0)
        mask = gaussian_filter(mask, sigma=3)
        
        # Apply random rotation
        angle = np.random.uniform(0, 360)
        M = cv2.getRotationMatrix2D((x, y), angle, 1.0)
        mask = cv2.warpAffine(mask, M, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)
        
        # Bound the void strictly within rect_coords
        mask *= rect_mask
        
        return mask

    # Initialize result image and mask
    result = image_rgb.copy()
    combined_mask = np.zeros((height, width), dtype=np.float32)
    
    # Calculate image statistics
    mean_color = np.mean(image_rgb, axis=(0, 1))
    std_dev = np.std(image_rgb, axis=(0, 1))
    
    # Generate base Perlin noise for global texture
    base_perlin = generate_perlin_noise((height, width), scale=50, octaves=5, persistence=0.5)
    base_perlin_clamped = np.clip(base_perlin, 0.1, 0.9)

    # Generate each void
    for _ in range(num_voids):
        # Random center position within bounds
        x = np.random.randint(x1 + void_size, x2 - void_size)
        y = np.random.randint(y1 + void_size, y2 - void_size)
        
        # Generate void mask
        void_mask = generate_void_mask(x, y, void_size, direction)
        
        # Add noise variation to void edges
        noise = np.random.normal(0, 1, (height, width))
        noise = gaussian_filter(noise, sigma=1.5)
        void_mask *= (1 + 0.3 * noise)
        void_mask = np.clip(void_mask, 0, 1)
        
        # Add Perlin noise texture to void
        void_perlin = generate_perlin_noise((height, width), scale=20, octaves=2, persistence=0.3)
        void_mask *= (1 + 0.15 * void_perlin)
        void_mask = np.clip(void_mask, 0, 1)
        
        # Create cloud patches for depth variation
        cloud_patches = create_cloud_patches(void_mask, void_size)
        cloud_perlin = generate_perlin_noise((height, width), scale=void_size/8, octaves=1, persistence=0.2)
        cloud_perlin_clamped = np.clip(cloud_perlin, 0.1, 0.9)
        cloud_patches *= (1 + 0.1 * cloud_perlin)
        cloud_patches = np.clip(cloud_patches, 0, 1)
        
        # Apply darkness effect
        void_darkness = np.random.uniform(*darkness_range)
        for c in range(3):
            # Create dark and lighter regions
            dark_gray = mean_color[c] * (void_darkness + 0.25 * base_perlin_clamped)
            cloud_gray = mean_color[c] * (0.65 + 0.25 * cloud_perlin_clamped)
            
            # Blend colors based on cloud patterns
            effect = (result[:, :, c] * (1 - void_mask) + 
                     dark_gray * void_mask * (1 - cloud_patches) +
                     cloud_gray * void_mask * cloud_patches)
            result[:, :, c] = effect.astype(np.uint8)
        
        # Add halo effect if enabled
        halo_enabled = random.choice([True, False]) if enable_random_halo else True
        if halo_enabled:
            # Create halo mask
            void_mask_uint8 = (void_mask > 0.3).astype(np.uint8) * 255
            kernel_size = int(max(void_size * 0.15, 1))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            halo = cv2.dilate(void_mask_uint8, kernel)
            halo[void_mask_uint8 > 0] = 0
            
            # Smooth halo
            halo = cv2.GaussianBlur(halo, (0, 0), sigmaX=3)
            
            # Add randomness to halo intensity
            ran_size = (np.array(halo.shape) // 6).astype(np.int32)
            ran_size = np.maximum(ran_size, 1)
            ran = np.random.random(ran_size)
            ran = cv2.resize(ran, (halo.shape[1], halo.shape[0]))
            
            # Apply halo intensity
            halo = halo.astype(float)
            halo_intensity = 0.95
            halo = halo * halo_intensity
            halo = np.clip(halo, 0, 255).astype(np.uint8)
            
            # Add halo to result
            for c in range(3):
                result[:, :, c] = np.clip(result[:, :, c].astype(float) + halo, 0, 255).astype(np.uint8)
        
        # Add internal structure if enabled
        structure_enabled = random.choice([True, False]) if enable_random_structure else True
        if structure_enabled:
            # Create internal shading
            internal_shade = gaussian_filter(void_mask, sigma=1)
            internal_shade *= (1 + 0.5 * void_perlin)
            
            for c in range(3):
                shade_intensity = std_dev[c] * 0.3
                result[:, :, c] = np.clip(
                    result[:, :, c] - internal_shade * shade_intensity,
                    0, 255
                ).astype(np.uint8)
            
            # Add structured texture within void
            structure_texture = generate_perlin_noise(result.shape[:2], scale=12, octaves=4, persistence=0.6)
            structure_texture = (structure_texture - 0.5) * 2
            structure_texture = np.clip(structure_texture, -0.5, 0.5)
            structure_strength = 2
            
            # Create hard mask for structure application
            hard_mask = (void_mask > 0.8).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded_mask = cv2.erode(hard_mask, kernel, iterations=1).astype(np.float32)
            
            # Apply structure texture
            for c in range(3):
                channel = result[:, :, c].astype(np.float32)
                channel = channel * (1 - eroded_mask) + (
                    channel + structure_strength * structure_texture * std_dev[c]
                ) * eroded_mask
                result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        # Accumulate masks
        combined_mask = np.maximum(combined_mask, void_mask)
    
    # Create binary mask with threshold
    binary_mask = (combined_mask > 0.2).astype(np.uint8) * 255
    
    return result, binary_mask


def get_rect_coords(image_path, package_type):
    """
    Calculate valid region for defect placement based on package type.
    
    Args:
        image_path (Path): Path to the image file
        package_type (str): Package identifier ("53438", "53439", or "53440")
    
    Returns:
        list: Rectangle coordinates [x1, y1, x2, y2] defining valid placement area
    """
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    
    # Define margins based on package type
    if package_type == "53438":
        margin = 14
        rect_coords = [margin + 3, margin, width - margin - 4, height - margin]
    
    elif package_type == "53439":
        margin = 10
        rect_coords = [margin + 2, margin, width - margin - 4, height - margin + 3]
    
    elif package_type == "53440" and (height, width) == (220, 250):
        margin = 6
        rect_coords = [margin, margin, width - margin - 6, height - 2*margin - 6]
    
    elif package_type == "53440" and (height, width) == (250, 220):
        margin = 12
        rect_coords = [margin + 3, margin - 2, width - margin, height - margin + 4]
    
    else:
        rect_coords = [0, 0, width, height]
    
    return rect_coords





