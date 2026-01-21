import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path
import random
from tqdm import tqdm

# ===========================
# Configuration Parameters
# ===========================

NAME_PREFIX = "Voids_Holes_Like_Droplets"  # Prefix for output filenames
NUM_SPLATTERS_RANGE = [1, 2]  # Range for number of splatter centers
MIN_SIZE = 25  # Minimum splatter region size
MAX_SIZE = 50  # Maximum splatter region size
MAIN_DARKNESS_RANGE = (0.0, 0.0)  # Not used (main splatter disabled)
DROPLET_DARKNESS_RANGE = (0.15, 0.25)  # Darkness range for droplets
DROPLET_COUNT_RANGE = (1, 2)  # Number of droplets per splatter
DROPLET_SIZE_RANGE = (0.01, 0.02)  # Droplet size as fraction of diagonal
DROPLET_DISTANCE_RANGE = (0.1, 0.9)  # Droplet distance from center
GENERATE_MAIN_SPLATTER = False  # Don't generate main splatter shape
GENERATE_DROPLETS = True  # Generate droplet patterns
HALO_BLUR_SIGMA = 3  # Halo effect blur radius
VOID_BLUR_SIGMA = 21  # Void region blur radius
BLACK_INTENSITY_RANGE = (0.1, 0.2)  # Dark gray intensity range

NUMBER_OF_SYNTHETIC_IMAGES = 5  # Number of synthetic images to generate
PACKAGE_TYPE = "53438"  # Package type identifier

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


def mask_void_if_above_threshold(result, mask, threshold=60):
    """
    Remove mask pixels where the image intensity is above a threshold.
    
    This helps prevent masking bright regions that shouldn't be marked as voids.
    
    Args:
        result (numpy.ndarray): Image array (grayscale or RGB)
        mask (numpy.ndarray): Binary mask array
        threshold (int): Intensity threshold above which to remove mask pixels
    
    Returns:
        numpy.ndarray: Modified mask with high-intensity regions removed
    """
    mask_modified = mask.copy()
    void_indices = np.where(mask == 255)
    
    # Calculate intensity at void locations
    if result.ndim == 3:
        void_intensity = np.mean(result[void_indices], axis=1)
    else:
        void_intensity = result[void_indices]
    
    # Remove mask pixels where intensity is above threshold
    mask_modified[void_indices[0][void_intensity > threshold], 
                  void_indices[1][void_intensity > threshold]] = 0
    
    return mask_modified


def add_halo(mask, intensity=0.7, radius=15):
    """
    Add a halo (light ring) effect around the edges of the mask.
    
    Args:
        mask (numpy.ndarray): Binary mask defining void regions
        intensity (float): Intensity of the halo effect (0-1)
        radius (int): Radius of the halo in pixels
    
    Returns:
        numpy.ndarray: Halo mask with values in range [0, 255]
    """
    # Detect edges in the mask
    edges = cv2.Canny(mask, 100, 200)
    
    # Dilate edges to create halo region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    halo = cv2.dilate(edges, kernel)
    
    # Smooth the halo
    halo = cv2.GaussianBlur(halo, (radius, radius), 0)
    
    # Apply intensity scaling
    halo = (halo.astype(np.float32) / 255.0) * intensity * 255
    
    return halo.astype(np.uint8)


def create_splatter_void(
    image_rgb, rect_coords, num_splatters=2, min_size=60, max_size=80, 
    main_darkness_range=(0.0, 0.0), droplet_darkness_range=(0.0, 0.0),
    droplet_count_range=(1, 2), droplet_size_range=(0.01, 0.02), 
    droplet_distance_range=(0.2, 1.1), generate_main_splatter=False, 
    generate_droplets=True, rect_coords_override=None
):
    """
    Create hole-like void defects by generating dark droplet patterns.
    
    Args:
        image_rgb (numpy.ndarray): Input RGB image
        rect_coords (list): Bounding box [x1, y1, x2, y2] for void placement
        num_splatters (int): Number of splatter centers to create
        min_size (int): Minimum size of splatter region
        max_size (int): Maximum size of splatter region
        main_darkness_range (tuple): Darkness range for main splatter (not used when generate_main_splatter=False)
        droplet_darkness_range (tuple): Darkness range for droplets (0-1)
        droplet_count_range (tuple): Range for number of droplets per splatter (min, max)
        droplet_size_range (tuple): Droplet size as fraction of image diagonal
        droplet_distance_range (tuple): Distance of droplets from center as fraction of size
        generate_main_splatter (bool): Whether to generate main splatter shape (typically False for holes)
        generate_droplets (bool): Whether to generate droplets
        rect_coords_override (list): Override for rect_coords (not currently used)
    
    Returns:
        tuple: (result_image, combined_mask)
            - result_image: Modified image with void effects (RGB format)
            - combined_mask: Binary mask of all void regions (uint8)
    """
    height, width = image_rgb.shape[:2]
    image_diagonal = np.sqrt(width**2 + height**2)
    
    result = image_rgb.copy()
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    color_base = np.array([0, 0, 0])  # Black color for droplets

    def generate_splatter_shape(x, y, size, droplet_count_range, 
                              droplet_size_range, droplet_distance_range):
        """
        Generate droplet patterns around a center point.
        
        Returns:
            tuple: (main_mask, droplet_mask, combined_mask)
        """
        main_mask = np.zeros((height, width), dtype=np.float32)
        droplet_mask = np.zeros((height, width), dtype=np.float32)
        
        # Generate droplets if enabled
        if generate_droplets:
            num_droplets = np.random.randint(*droplet_count_range)
            
            for _ in range(num_droplets):
                # Calculate random droplet position
                drop_angle = np.random.uniform(0, 2*np.pi)
                drop_dist = np.random.uniform(*droplet_distance_range) * size
                drop_x = int(x + drop_dist * np.cos(drop_angle))
                drop_y = int(y + drop_dist * np.sin(drop_angle))
                
                # Calculate droplet size
                size_factor = np.random.uniform(*droplet_size_range)
                drop_size = max(1, int(image_diagonal * size_factor))
                
                # Clip to valid bounds
                drop_x = np.clip(drop_x, rect_coords[0] + drop_size, rect_coords[2] - drop_size)
                drop_y = np.clip(drop_y, rect_coords[1] + drop_size, rect_coords[3] - drop_size)
                
                # Draw droplet
                cv2.circle(droplet_mask, (drop_x, drop_y), drop_size, 1.0, -1)
            
            # Smooth droplet edges
            droplet_mask = gaussian_filter(droplet_mask, sigma=1.5)
        
        combined = np.maximum(main_mask, droplet_mask)
        return main_mask, droplet_mask, combined

    # Generate each splatter
    for _ in range(num_splatters):
        # Random center position
        x = np.random.randint(rect_coords[0] + 20, rect_coords[2] - 20)
        y = np.random.randint(rect_coords[1] + 20, rect_coords[3] - 20)
        size = np.random.randint(min_size, max_size)
        
        # Generate splatter masks
        main_mask, droplet_mask, splatter_mask = generate_splatter_shape(
            x, y, size, 
            droplet_count_range,
            droplet_size_range,
            droplet_distance_range
        )
        
        # Generate color noise for texture variation
        color_noise1 = generate_noise_texture((height, width), scale=size/5)
        color_noise2 = generate_noise_texture((height, width), scale=size/10)
        color_noise = 0.5 * color_noise1 + 0.5 * color_noise2
        
        droplet_darkness = np.random.uniform(*droplet_darkness_range)
        
        # Apply droplet effect
        if generate_droplets and np.max(droplet_mask) > 0:
            # Create hard mask (no gradient)
            droplet_density = (droplet_mask > 0.2).astype(np.float32)
            droplet_darkness_factor = droplet_darkness + 0.15 * color_noise
            
            # Create dark colors for droplets
            droplet_colors = np.zeros((height, width, 3), dtype=float)
            for c in range(3):
                droplet_colors[:, :, c] = color_base[c] * droplet_darkness_factor
            
            # Apply colors to image
            for c in range(3):
                channel = result[:,:,c].astype(float)
                channel = channel * (1 - droplet_density) + droplet_colors[:,:,c] * droplet_density
                result[:,:,c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        # Accumulate masks
        combined_mask = np.maximum(combined_mask, (splatter_mask > 0.2).astype(np.uint8) * 255)
    
    # Remove mask pixels in bright regions
    combined_mask = mask_void_if_above_threshold(result, combined_mask, threshold=155)
    
    return result, combined_mask


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



#



