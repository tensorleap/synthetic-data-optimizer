import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
from pathlib import Path
from tqdm import tqdm

# ===========================
# Configuration Parameters - Main Splatter with Droplets
# ===========================

NAME_PREFIX = "MainSplatter_with_Droplets"
NUM_SPLATTERS = 1
MIN_SIZE = 25
MAX_SIZE = 50
MAIN_DARKNESS_RANGE = (0.25, 0.55)  # Moderate darkness range
DROPLET_DARKNESS_RANGE = (0.15, 0.25)  # Less contrast with main splatter
DROPLET_COUNT_RANGE = (10, 25)
DROPLET_SIZE_RANGE = (0.001, 0.008)  # Smaller droplets
DROPLET_DISTANCE_RANGE = (0.9, 1.3)  # Droplets a bit further from center
GENERATE_MAIN_SPLATTER = True
GENERATE_DROPLETS = True

NUMBER_OF_SYNTHETIC_IMAGES = 2000  # Number of synthetic images to generate
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


def mask_void_if_above_threshold(result, mask, threshold=60):
    """
    Remove mask pixels where the image intensity is above a threshold.
    
    This helps prevent masking bright regions that shouldn't be marked as voids.
    
    Args:
        result (numpy.ndarray): Image array (grayscale or RGB)
        mask (numpy.ndarray): Binary mask (void=255, background=0)
        threshold (int): Intensity threshold above which to remove mask pixels
    
    Returns:
        numpy.ndarray: Modified mask with high-intensity regions removed
    """
    mask_modified = mask.copy()
    void_indices = np.where(mask == 255)
    
    # Calculate mean intensity for each void pixel
    if result.ndim == 3:
        void_intensity = np.mean(result[void_indices], axis=1)
    else:
        void_intensity = result[void_indices]
    
    # Remove mask pixels where intensity is above threshold
    mask_modified[void_indices[0][void_intensity > threshold], 
                  void_indices[1][void_intensity > threshold]] = 0
    
    return mask_modified


def create_splatter_void(
    image_rgb, rect_coords, num_splatters=3, min_size=30, max_size=80, 
    main_darkness_range=(0.2, 0.3), droplet_darkness_range=(0.1, 0.15),
    droplet_count_range=(5, 15), droplet_size_range=(0.005, 0.015), 
    droplet_distance_range=(0.8, 1.3), generate_main_splatter=True, 
    generate_droplets=True
):
    """
    Generate splatter-like void effects with varying density and edge characteristics.
    
    Args:
        image_rgb (numpy.ndarray): RGB image to apply void effects to
        rect_coords (list): Bounds for the void [x1, y1, x2, y2]
        num_splatters (int): Number of splatter voids to create
        min_size (int): Minimum size of main splatter
        max_size (int): Maximum size of main splatter
        main_darkness_range (tuple): Range of darkness values (min, max) for main splatter (0-1)
        droplet_darkness_range (tuple): Range of darkness values (min, max) for droplets (0-1)
        droplet_count_range (tuple): Range for number of droplets (min, max) per splatter
        droplet_size_range (tuple): Range for droplet size as fraction of image diagonal (0-1)
        droplet_distance_range (tuple): Range for droplet distance from center relative to splatter size
        generate_main_splatter (bool): Whether to generate the main splatter shape
        generate_droplets (bool): Whether to generate droplets around the splatter
    
    Returns:
        tuple: (result_image, combined_mask)
            - result_image: Modified image with void effect (RGB format)
            - combined_mask: Binary mask of the void region (uint8)
    """
    height, width = image_rgb.shape[:2]
    image_diagonal = np.sqrt(width**2 + height**2)
    
    result = image_rgb.copy()
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate image statistics - using median color as the base
    color_base = np.median(image_rgb.reshape(-1, 3), axis=0)
    
    def generate_splatter_shape(x, y, size, droplet_count_range, 
                              droplet_size_range, droplet_distance_range):
        """
        Create an irregular splatter shape with droplets, returning both masks separately.
        
        Returns:
            tuple: (main_mask, droplet_mask, combined_mask)
        """
        main_mask = np.zeros((height, width), dtype=np.float32)
        droplet_mask = np.zeros((height, width), dtype=np.float32)
        
        # Generate main splatter if enabled
        if generate_main_splatter:
            num_points = np.random.randint(20, 35)
            angles = np.linspace(0, 2*np.pi, num_points)
            
            # Create varying radii with deliberate irregularity 
            base_radii = size * np.random.uniform(0.3, 1.0, num_points)
            
            # Add "splatter" effect by making some points extend further
            splatter_effect = np.random.choice(
                [1.0, 1.5, 2.0, 2.5], 
                num_points, 
                p=[0.7, 0.15, 0.1, 0.05]
            )
            radii = base_radii * splatter_effect
            
            # Create points with some randomness
            points = []
            for angle, radius in zip(angles, radii):
                # Add noise to angle for irregularity
                noisy_angle = angle + np.random.normal(0, 0.1)
                px = x + int(radius * np.cos(noisy_angle))
                py = y + int(radius * np.sin(noisy_angle))
                
                # Clip to bounds
                px = np.clip(px, rect_coords[0] + 5, rect_coords[2] - 5)
                py = np.clip(py, rect_coords[1] + 5, rect_coords[3] - 5)
                points.append([[px, py]])
            
            # Create mask for main splatter
            cv2.fillPoly(main_mask, [np.array(points, dtype=np.int32)], 1.0)
            main_mask = gaussian_filter(main_mask, sigma=2)
        
        # Generate droplets if enabled
        if generate_droplets:
            num_droplets = np.random.randint(*droplet_count_range)
            for _ in range(num_droplets):
                # Random angle and distance from center
                drop_angle = np.random.uniform(0, 2*np.pi)
                drop_dist = np.random.uniform(*droplet_distance_range) * size
                
                # Position
                drop_x = int(x + drop_dist * np.cos(drop_angle))
                drop_y = int(y + drop_dist * np.sin(drop_angle))
                
                # Size based on image diagonal rather than splatter size
                size_factor = np.random.uniform(*droplet_size_range)
                drop_size = int(image_diagonal * size_factor)
                
                # Ensure minimum size of 1 pixel and valid position
                drop_size = max(1, drop_size)
                drop_x = np.clip(drop_x, rect_coords[0] + drop_size, rect_coords[2] - drop_size)
                drop_y = np.clip(drop_y, rect_coords[1] + drop_size, rect_coords[3] - drop_size)
                
                # Draw droplet (small circle) on separate mask
                cv2.circle(droplet_mask, (drop_x, drop_y), drop_size, 1.0, -1)
            
            # Slightly sharper edges for droplets
            droplet_mask = gaussian_filter(droplet_mask, sigma=1.5)
        
        # Combined mask for visualization
        combined = np.maximum(main_mask, droplet_mask)
        
        return main_mask, droplet_mask, combined
    
    def generate_density_mask(mask, x, y):
        """
        Create density variation within the splatter - darker in center, lighter at edges.
        
        Args:
            mask (numpy.ndarray): Splatter mask
            x (int): Center x coordinate
            y (int): Center y coordinate
        
        Returns:
            numpy.ndarray: Density mask with gradient and noise
        """
        # Create gradient from center
        y_coords, x_coords = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
        
        # Normalize distance to 0-1 range
        max_dist = np.max(dist_from_center * mask) if np.max(mask) > 0 else 1
        norm_dist = dist_from_center / max_dist
        
        # Create gradient (darker in center, lighter at edges)
        gradient = 1.0 - 0.8 * norm_dist
        
        # Apply texture variation with multiple noise scales
        noise1 = generate_noise_texture((height, width), scale=max_dist/4)
        noise2 = generate_noise_texture((height, width), scale=max_dist/8)
        combined_noise = 0.7 * noise1 + 0.3 * noise2
        
        # Combine gradient with noise
        density = gradient * (0.4 + 0.6 * combined_noise)
        
        # Apply only within the splatter mask
        return density * mask
    
    # Generate splatters
    for _ in range(num_splatters):
        # Random position within bounds
        x = np.random.randint(rect_coords[0] + 20, rect_coords[2] - 20)
        y = np.random.randint(rect_coords[1] + 20, rect_coords[3] - 20)
        
        # Random size between min_size and max_size
        size = np.random.randint(min_size, max_size)
        
        # Generate splatter shape with separate masks for main body and droplets
        main_mask, droplet_mask, splatter_mask = generate_splatter_shape(
            x, y, size, 
            droplet_count_range,
            droplet_size_range,
            droplet_distance_range
        )
        
        # Generate noise for color variation (shared across channels)
        color_noise1 = generate_noise_texture((height, width), scale=size/5)
        color_noise2 = generate_noise_texture((height, width), scale=size/10)
        color_noise = 0.5 * color_noise1 + 0.5 * color_noise2
        
        # Random darkness levels from specified ranges
        main_darkness = np.random.uniform(*main_darkness_range)
        droplet_darkness = np.random.uniform(*droplet_darkness_range)
        
        # Apply main splatter if enabled
        if generate_main_splatter and np.max(main_mask) > 0:
            # Generate density variation for main splatter
            density_mask = generate_density_mask(main_mask, x, y)
            
            # Create darkness factors with noise influence
            main_darkness_factor = main_darkness + 0.15 * color_noise
            
            # Create RGB colors for main splatter
            main_colors = np.zeros((height, width, 3), dtype=float)
            for c in range(3):
                main_colors[:, :, c] = color_base[c] * main_darkness_factor
            
            # Apply the colors with proper masking
            for c in range(3):
                channel = result[:,:,c].astype(float)
                channel = channel * (1 - density_mask) + main_colors[:,:,c] * density_mask
                result[:,:,c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        # Apply droplets if enabled
        if generate_droplets and np.max(droplet_mask) > 0:
            # Generate density for droplets - make them more uniform/darker
            droplet_density = gaussian_filter(droplet_mask, sigma=1)
            
            # Create darkness factors with noise influence
            droplet_darkness_factor = droplet_darkness + 0.15 * color_noise
            
            # Create RGB colors for droplets
            droplet_colors = np.zeros((height, width, 3), dtype=float)
            for c in range(3):
                droplet_colors[:, :, c] = color_base[c] * droplet_darkness_factor
            
            # Apply the colors with proper masking
            for c in range(3):
                channel = result[:,:,c].astype(float)
                channel = channel * (1 - droplet_density) + droplet_colors[:,:,c] * droplet_density
                result[:,:,c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        # Add to combined mask
        combined_mask = np.maximum(combined_mask, (splatter_mask > 0.2).astype(np.uint8) * 255)
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


def apply_splatter_voids(
    image_path, num_splatters=3, min_size=30, max_size=80, 
    main_darkness_range=(0.2, 0.3), droplet_darkness_range=(0.1, 0.15),
    droplet_count_range=(5, 15), droplet_size_range=(0.005, 0.015), 
    droplet_distance_range=(0.8, 1.3), generate_main_splatter=True, 
    generate_droplets=True, rect_coords=None
):
    """
    Apply splatter void effects to an image and display results.
    
    Args:
        image_path (str): Path to the input image
        num_splatters (int): Number of splatter voids to create
        min_size (int): Minimum size of main splatter
        max_size (int): Maximum size of main splatter
        main_darkness_range (tuple): Range of darkness values (min, max) for main splatter (0-1)
        droplet_darkness_range (tuple): Range of darkness values (min, max) for droplets (0-1)
        droplet_count_range (tuple): Range for number of droplets (min, max) per splatter
        droplet_size_range (tuple): Range for droplet size as fraction of image diagonal (0-1)
        droplet_distance_range (tuple): Range for droplet distance from center relative to splatter size
        generate_main_splatter (bool): Whether to generate the main splatter shape
        generate_droplets (bool): Whether to generate droplets around the splatter
        rect_coords (list or None): Bounds for the void [x1, y1, x2, y2]. If None, will be auto-calculated
    
    Returns:
        tuple: (result_image, mask)
    """
    # Load and prepare the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    
    # Define image bounds if not provided
    if rect_coords is None:
        margin = 6
        rect_coords = [margin, margin, width - margin - 6, height - 2*margin - 6]
        
    # Create splatter voids with customizable parameters
    result, mask = create_splatter_void(
        image_rgb, rect_coords, 
        num_splatters=num_splatters,
        min_size=min_size,
        max_size=max_size,
        main_darkness_range=main_darkness_range,
        droplet_darkness_range=droplet_darkness_range,
        droplet_count_range=droplet_count_range,
        droplet_size_range=droplet_size_range,
        droplet_distance_range=droplet_distance_range,
        generate_main_splatter=generate_main_splatter,
        generate_droplets=generate_droplets
    )

    # Determine title based on what was generated
    if generate_main_splatter and generate_droplets:
        title_suffix = "Complete Splatter"
    elif generate_main_splatter:
        title_suffix = "Main Splatter Only"
    elif generate_droplets:
        title_suffix = "Droplets Only"
    else:
        title_suffix = "Nothing Generated (Both Disabled)"

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title("Splatter Void Mask")
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(result)
    plt.title(f"Result with {title_suffix}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return result, mask




