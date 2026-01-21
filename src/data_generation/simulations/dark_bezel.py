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

NAME_PREFIX = "Voids_dark_bezel_shaped"  # Prefix for output filenames
NUM_SPLATTERS = 2  # Maximum number of voids per image
MIN_SIZE = 25  # Minimum void size (pixels)
MAX_SIZE = 50  # Maximum void size (pixels)

NUMBER_OF_SYNTHETIC_IMAGES = 5  # Number of synthetic images to generate
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


def create_cloud_pattern(mask, size, num_clouds=3):
    """
    Create cloud-like patterns within the mask for texture variation.
    
    Args:
        mask (numpy.ndarray): Base mask where clouds will be generated
        size (int): Approximate size of the void (used to scale clouds)
        num_clouds (int): Number of cloud patterns to generate
    
    Returns:
        numpy.ndarray: Cloud pattern mask with values in [0, 1]
    """
    # Find valid positions within the mask
    valid_positions = np.where(mask > 0.5)
    if len(valid_positions[0]) == 0:
        return np.zeros_like(mask)
    
    cloud_mask = np.zeros_like(mask)
    
    for _ in range(num_clouds):
        # Select random center point within valid positions
        idx = np.random.randint(len(valid_positions[0]))
        center = valid_positions[0][idx], valid_positions[1][idx]
        
        # Generate cloud shape with irregular edges
        cloud_size = int(size * np.random.uniform(0.2, 0.4))
        angles = np.linspace(0, 2*np.pi, np.random.randint(8, 15))
        radii = cloud_size * (0.5 + 0.5 * np.random.rand(len(angles)))
        
        # Create cloud points
        points = np.array([[[
            center[1] + int(r * np.cos(a)),
            center[0] + int(r * np.sin(a))
        ]] for a, r in zip(angles, radii)], dtype=np.int32)
        
        # Draw cloud and apply gaussian blur for soft edges
        temp = np.zeros_like(mask)
        cv2.fillPoly(temp, [points], 0.5)
        cloud_mask = np.maximum(cloud_mask, gaussian_filter(temp, sigma=2))
    
    return cloud_mask * mask


def create_void_effect(image_path, rect_coords, num_voids=1, void_size=20):
    """
    Generate dark bezel-shaped void effects with cloud patterns within specified bounds.
    
    Args:
        image_path (Path): Path to the input image file
        rect_coords (list): Bounding box [x1, y1, x2, y2] for void placement
        num_voids (int): Number of void defects to create
        void_size (int): Approximate size of each void
    
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
    
    # Extract bounds for void placement
    x1, y1, x2, y2 = rect_coords
    
    result = image_rgb.copy()
    combined_mask = np.zeros((height, width), dtype=np.float32)
    
    # Calculate image statistics for realistic coloring
    mean_color = np.mean(image_rgb, axis=(0, 1))
    std_dev = np.std(image_rgb, axis=(0, 1))
    
    # Generate base texture for variation
    base_texture = generate_noise_texture((height, width), scale=30)
    
    # Generate each void
    for _ in range(num_voids):
        # Generate void position within bounds
        x = np.random.randint(x1, x2)
        y = np.random.randint(y1, y2)
        current_size = int(void_size * np.random.uniform(0.7, 1.3))
        
        # Create irregular bezel-shaped outline
        angles = np.linspace(0, 2*np.pi, np.random.randint(25, 35))
        radii = current_size * (0.3 + 0.7 * np.random.rand(len(angles)))
        
        # Add wave patterns for organic shape
        radii *= (1 + 0.3 * np.sin(angles * np.random.randint(2, 4)) + 
                 0.2 * np.cos(angles * np.random.randint(3, 5)))
        
        # Create points and clip to bounds
        points = np.array([[[ 
            np.clip(x + int(r * np.cos(a)), x1 + 5, x2 - 5),
            np.clip(y + int(r * np.sin(a)), y1 + 5, y2 - 5)
        ]] for a, r in zip(angles, radii)], dtype=np.int32)
        
        # Create and process void mask
        void_mask = np.zeros((height, width), dtype=np.float32)
        cv2.fillPoly(void_mask, [points], 1.0)
        void_mask = gaussian_filter(void_mask, sigma=2)
        
        # Add texture variations to void
        void_texture = generate_noise_texture((height, width), scale=void_size/4, octaves=3)
        void_mask *= (1 + 0.3 * void_texture)
        void_mask = np.clip(void_mask, 0, 1)
        
        # Add cloud patterns for depth variation
        clouds = create_cloud_pattern(void_mask, current_size)
        
        # Apply color effects to each channel
        for c in range(3):
            # Create darker regions
            dark = mean_color[c] * (0.5 + 0.1 * base_texture)
            darker = mean_color[c] * (0.3 + 0.1 * void_texture)
            
            # Blend colors based on cloud patterns
            result[:, :, c] = (result[:, :, c] * (1 - void_mask) + 
                              dark * void_mask * (1 - clouds) +
                              darker * void_mask * clouds).astype(np.uint8)
            
            # Add shading effect
            shade = gaussian_filter(void_mask, sigma=5) * (1 + 0.2 * void_texture)
            result[:, :, c] = np.clip(
                result[:, :, c] - shade * std_dev[c] * 0.3,
                0, 255
            ).astype(np.uint8)
        
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
    
    Raises:
        ValueError: If image path doesn't exist
    """
    if not image_path.exists():
        raise ValueError(f"Image path {image_path} does not exist.")
    
    # Read image to get dimensions
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]

    # Define margins based on package type
    if package_type == "53438":
        margin = 14
        rect_coords = [margin + 3, margin, width - margin - 4, height - margin]
    
    elif package_type == "53439":
        margin = 10
        rect_coords = [margin + 2, margin, width - margin - 4, height - margin + 3]
    
    elif package_type == "53440" and image.shape[0:2] == (220, 250):
        margin = 6
        rect_coords = [margin, margin, width - margin - 6, height - 2*margin - 6]
    
    elif package_type == "53440" and image.shape[0:2] == (250, 220):
        margin = 12
        rect_coords = [margin + 3, margin - 2, width - margin, height - margin + 4]
    
    return rect_coords





