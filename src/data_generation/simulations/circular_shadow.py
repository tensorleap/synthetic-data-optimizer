import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
from pathlib import Path
from tqdm import tqdm

# ===========================
# Configuration Parameters
# ===========================

NAME_PREFIX = "Voids_circular_outerShadow1"  # Prefix for output filenames
RADIUS_RANGE = (25, 50)  # Min and max radius for circular void (pixels)
IRREGULAIRTY_RANGE = (0.05, 0.1)  # Min and max edge irregularity factor
SHADOW_WIDTH_RANGE = (0.15, 0.2)  # Shadow width as fraction of radius
SHADOW_OPACITY_RANGE = (0.8, 0.95)  # Shadow opacity range (0-1)

NUMBER_OF_SYNTHETIC_IMAGES = 5  # Number of synthetic images to generate

PACKAGE_TYPE = "53440"  # Package type identifier

# Source directory containing good unit images
SOURCE_DIR = Path(fr"U:\Easy_Panda\Datasets\Train_images_500_random_{PACKAGE_TYPE}\extracted_chips_template_matching\TestSet\GoodUnits")

# Destination directory for synthetic defect images
DESTINATION_DIR = Path(fr"Train_images_500_random_{PACKAGE_TYPE}\extracted_chips_template_matching\Synth_Data_{NAME_PREFIX}")





def generate_noise_texture(shape, scale=10, octaves=6, persistence=0.5):
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


def create_outer_shadow_splatter(
    image, position=None, radius=50, irregularity=0.05, rect_coords=None,
    outer_shadow_width=0.15, outer_shadow_opacity=0.9
):
    """
    Create a circular void defect with outer shadow effect on an image.
    
    Args:
        image (numpy.ndarray): Input image (BGR format)
        position (tuple): Center position (x, y). If None, randomly placed within rect_coords
        radius (int): Base radius of the circular void
        irregularity (float): Amount of edge irregularity (0 = perfect circle)
        rect_coords (list): Bounding box [x1, y1, x2, y2] for random placement
        outer_shadow_width (float): Shadow width as fraction of radius
        outer_shadow_opacity (float): Shadow opacity (0-1)
    
    Returns:
        tuple: (result_image, shadow_mask)
            - result_image: Modified image with shadow effect (BGR format)
            - shadow_mask: Binary mask of the shadow region (uint8)
    """
    # Convert to RGB for processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    height, width = image_rgb.shape[:2]

    # Determine splatter center position
    if position is None:
        x = np.random.randint(rect_coords[0], rect_coords[2])
        y = np.random.randint(rect_coords[1], rect_coords[3])
    else:
        x, y = position

    # Create coordinate grids and distance map
    y_coords, x_coords = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
    angles = np.arctan2(y_coords - y, x_coords - x)

    # Generate noise pattern for irregular edges
    noise = np.random.normal(0, 1, (height, width))
    noise = gaussian_filter(noise, sigma=10)
    noise = 2 * (noise - noise.min()) / (noise.max() - noise.min()) - 1

    # Create base circular mask with optional irregularity
    raw_splatter_mask = np.zeros((height, width), dtype=np.float32)
    if irregularity > 0:
        radius_var = radius * (1 + irregularity * noise * np.cos(angles * 8))
        raw_splatter_mask[dist_from_center < radius_var] = 1.0
    else:
        cv2.circle(raw_splatter_mask, (x, y), radius, 1.0, -1)

    # Create outer shadow mask
    outer_shadow_mask = np.zeros((height, width), dtype=np.float32)
    outer_shadow_radius = radius * (1 + outer_shadow_width)
    
    # Generate noise for shadow irregularity
    outer_noise = gaussian_filter(np.random.normal(0, 1, (height, width)), sigma=18)
    outer_noise = 2 * (outer_noise - outer_noise.min()) / (outer_noise.max() - outer_noise.min()) - 1
    
    if irregularity > 0:
        outer_radius_var = outer_shadow_radius * (1 + 0.6 * irregularity * outer_noise * np.cos(angles * 6))
        outer_shadow_mask[dist_from_center < outer_radius_var] = 1.0
    else:
        cv2.circle(outer_shadow_mask, (x, y), int(outer_shadow_radius), 1.0, -1)
    
    # Isolate shadow region (exclude center void)
    outer_shadow_mask = np.clip(outer_shadow_mask - raw_splatter_mask, 0, 1)

    # Create gradient fade from inner to outer edge
    inner_radius = radius
    outer_radius = outer_shadow_radius
    outer_shadow_gradient = np.ones((height, width), dtype=np.float32)
    fade_zone = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)
    outer_shadow_gradient[fade_zone] = 1.0 - (dist_from_center[fade_zone] - inner_radius) / (outer_radius - inner_radius)
    outer_shadow_gradient[dist_from_center > outer_radius] = 0.0
    outer_shadow_gradient = outer_shadow_gradient * outer_shadow_mask

    # Apply Gaussian blur to smooth shadow edges
    auto_blur = max(1, int(outer_shadow_width * radius * 0.8))
    if auto_blur > 0:
        pad_size = int(auto_blur * 1)
        padded_outer_shadow = np.pad(outer_shadow_gradient, pad_size, mode='reflect')
        padded_outer_shadow = cv2.GaussianBlur(padded_outer_shadow, (0, 0), auto_blur)
        outer_shadow_gradient = padded_outer_shadow[pad_size:pad_size+height, pad_size:pad_size+width]

    # Add visible noise texture to shadow
    visible_noise = generate_noise_texture((height, width), scale=radius/3, octaves=8, persistence=0.6)
    visible_noise = (visible_noise - 0.5) * 0.9
    outer_shadow_gradient_noisy = np.clip(outer_shadow_gradient + visible_noise * outer_shadow_gradient, 0, 1)

    # Apply shadow effect to image
    result = image_rgb.copy()
    mean_gray = np.mean(image_rgb)
    outer_shadow_color = np.array([0.001, 0.001, 0.001]) * mean_gray

    # Blend shadow with original image for each color channel
    for c in range(3):
        channel = result[:, :, c].astype(float)
        result[:, :, c] = channel * (1 - outer_shadow_gradient_noisy * outer_shadow_opacity) + \
                        outer_shadow_color[c] * outer_shadow_gradient_noisy * outer_shadow_opacity

    # Create binary mask for shadow region (threshold at 0.12)
    shadow_mask_combined = (outer_shadow_gradient_noisy > 0.12).astype(np.uint8) * 255

    # Convert back to BGR format
    result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return result_bgr, shadow_mask_combined


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






