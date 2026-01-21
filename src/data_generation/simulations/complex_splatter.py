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

NAME_PREFIX = "ComplexVoid_Struct_with_Droplets"  # Prefix for output filenames
NUM_SPLATTERS = 15  # Number of splatter voids to create
MIN_SIZE = 15  # Minimum size of main splatter
MAX_SIZE = 35  # Maximum size of main splatter
MAIN_DARKNESS_RANGE = (0.35, 0.45)  # Darkness range for main splatter (0-1)
DROPLET_DARKNESS_RANGE = (0.1, 0.15)  # Darkness range for droplets (0-1)
DROPLET_COUNT_RANGE = (5, 15)  # Range for number of droplets per splatter
DROPLET_SIZE_RANGE = (0.002, 0.009)  # Droplet size as fraction of image diagonal
DROPLET_DISTANCE_RANGE = (0.8, 1.3)  # Droplet distance from center
GENERATE_MAIN_SPLATTER = True  # Whether to generate main splatter
GENERATE_DROPLETS = True  # Whether to generate droplets
SHADOW_INTENSITY = 0.6  # Shadow effect intensity (0-1)
SHADOW_RADIUS_FACTOR = 1.0  # Shadow spread factor
SHADOW_NOISE = 0.5  # Shadow noise amount (0-1)
SHADOW_BLUR = 2.0  # Shadow blur sigma
LIGHT_INTENSITY = 0.7  # Light halo intensity (0-1)
LIGHT_RADIUS_FACTOR = 1.5  # Light halo size factor
HALO_NOISE = 0.5  # Halo noise amount (0-1)
GRADIENT_DIRECTION = "edge_dark"  # Gradient direction: "center_dark" or "edge_dark"
DROPLET_BLUR = 1.5  # Droplet blur sigma
MAIN_BLUR = 2.0  # Main splatter blur sigma

PACKAGE_TYPE = "53439"  # Package type identifier
NUMBER_OF_SYNTHETIC_IMAGES = 10  # Number of synthetic images to generate

# Source directory containing good unit images
SOURCE_DIR = Path(fr"U:\Easy_Panda\Datasets\Train_images_500_random_{PACKAGE_TYPE}\extracted_chips_template_matching\TestSet\GoodUnits")

# Destination directory for synthetic defect images
DESTINATION_DIR = Path(fr"Train_images_500_random_{PACKAGE_TYPE}\extracted_chips_template_matching\Synth_Data_{NAME_PREFIX}")





def generate_noise_texture(shape, scale=5, octaves=8, persistence=0.5):
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


def generate_enhanced_noise(shape, octaves=6, persistence=0.6, scale=30.0, smoothness=2.0):
    """
    Generate enhanced Perlin-like noise with more natural patterns.
    
    Args:
        shape (tuple): Size of the output (height, width)
        octaves (int): Number of noise layers to combine
        persistence (float): How quickly amplitudes decrease for higher octaves (0.0-1.0)
        scale (float): Base feature size - lower values = more fine detail
        smoothness (float): Controls noise smoothing - higher values = smoother noise
    
    Returns:
        numpy.ndarray: Enhanced noise texture with natural patterns
    """
    height, width = shape
    noise = np.zeros(shape, dtype=np.float32)
    total_amplitude = 0.0
    
    # Create noise at multiple octaves
    for octave in range(octaves):
        frequency = 2**octave
        amplitude = persistence**octave
        
        # Adjust scale for this octave
        octave_scale = scale / frequency
        
        # Create initial random noise
        base_noise = np.random.normal(0, 1, 
                        (int(height/octave_scale*2), int(width/octave_scale*2)))
        
        # Apply smoothing with adaptive sigma
        smooth_sigma = smoothness / frequency
        smooth_noise = gaussian_filter(base_noise, sigma=smooth_sigma)
        
        # Resize to target size
        resized_noise = cv2.resize(smooth_noise, (width, height))
        
        # Add directional bias to create more structured patterns
        if octave % 2 == 0:
            # Create gradient pattern
            y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)
            x_grad = np.sin(x_coords / width * np.pi * (octave % 3 + 1))
            y_grad = np.cos(y_coords / height * np.pi * (octave % 2 + 1))
            gradient = x_grad * y_grad * 0.3
            
            # Blend with noise
            resized_noise = resized_noise * 0.7 + gradient * 0.3
        
        # Add to the accumulated noise
        noise += amplitude * resized_noise
        total_amplitude += amplitude
    
    # Normalize and enhance contrast
    normalized = noise / total_amplitude
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
    
    # Apply contrast enhancement
    enhanced = np.power(normalized, 0.8)
    
    return enhanced


def create_splatter_void(
    image_rgb, rect_coords, num_splatters=3, min_size=30, max_size=80, 
    main_darkness_range=(0.2, 0.3), droplet_darkness_range=(0.1, 0.15),
    droplet_count_range=(5, 15), droplet_size_range=(0.005, 0.015), 
    droplet_distance_range=(0.8, 1.3), generate_main_splatter=True, 
    generate_droplets=True, shadow_intensity=0.3, shadow_radius_factor=1.0, 
    shadow_noise=0.5, shadow_blur=2.0, light_intensity=0.15, 
    light_radius_factor=1.5, halo_noise=0.5, gradient_direction="center_dark", 
    droplet_blur=1.5, main_blur=2.0
):
    """
    Generate complex splatter-like void defects with varying density and edge characteristics.
    
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
        shadow_intensity (float): Intensity of shadow effect inside the void edges (0.0-1.0)
        shadow_radius_factor (float): Factor to control shadow spread from edges
        shadow_noise (float): Amount of noise to add to the shadow effect (0.0-1.0)
        shadow_blur (float): Additional Gaussian blur sigma to apply to shadow mask
        light_intensity (float): Intensity of light effect outside the void (0.0-1.0)
        light_radius_factor (float): Factor to control the size of the light halo
        halo_noise (float): Amount of noise to add to the halo effect (0.0-1.0)
        gradient_direction (str): "center_dark" (darker in center) or "edge_dark" (darker at edges)
        droplet_blur (float): Sigma for gaussian blur on droplets - higher = softer edges
        main_blur (float): Sigma for gaussian blur on main splatter - higher = softer edges
    
    Returns:
        tuple: (result_image, binary_mask)
            - result_image: Modified image with void effect (RGB format)
            - binary_mask: Binary mask of the void region (uint8)
    """
    height, width = image_rgb.shape[:2]
    image_diagonal = np.sqrt(width**2 + height**2)
    
    result = image_rgb.copy()
    color_base = np.median(image_rgb.reshape(-1, 3), axis=0)
    
    # Validate gradient_direction parameter
    if gradient_direction not in ["center_dark", "edge_dark"]:
        print(f"Warning: Invalid gradient_direction '{gradient_direction}'. Using 'center_dark'.")
        gradient_direction = "center_dark"
    
    def generate_splatter_shape(x, y, size, droplet_count_range, 
                                droplet_size_range, droplet_distance_range):
        """
        Create an irregular splatter shape with wavy protrusions.
        
        Returns both visual effect masks (blurred) and structure masks (binary).
        """
        # Visual effect masks (will be blurred)
        main_effect_mask = np.zeros((height, width), dtype=np.float32)
        droplet_effect_mask = np.zeros((height, width), dtype=np.float32)
        
        # Structure masks (sharp/binary)
        main_binary_mask = np.zeros((height, width), dtype=np.float32)
        droplet_binary_mask = np.zeros((height, width), dtype=np.float32)
        
        # Generate main splatter if enabled
        if generate_main_splatter:
            # Create anchor points with varying radii
            num_main_points = np.random.randint(12, 20)
            main_angles = np.linspace(0, 2*np.pi, num_main_points, endpoint=False)
            
            base_radii = size * np.random.uniform(0.3, 1.0, num_main_points)
            
            # Add splatter effect by making some points extend further
            splatter_effect = np.random.choice(
                [1.0, 1.5, 2.0, 2.5], 
                num_main_points, 
                p=[0.7, 0.15, 0.1, 0.05]
            )
            main_radii = base_radii * splatter_effect
            
            # Create main anchor points
            main_points = []
            for angle, radius in zip(main_angles, main_radii):
                noisy_angle = angle + np.random.normal(0, 0.05)
                px = x + int(radius * np.cos(noisy_angle))
                py = y + int(radius * np.sin(noisy_angle))
                
                px = np.clip(px, rect_coords[0] + 5, rect_coords[2] - 5)
                py = np.clip(py, rect_coords[1] + 5, rect_coords[3] - 5)
                main_points.append([px, py])
            
            # Create wavy lines between the main points
            points = []
            for i in range(num_main_points):
                p1 = main_points[i]
                p2 = main_points[(i + 1) % num_main_points]
                
                segment_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                num_interp = max(3, int(segment_length / 10))
                
                points.append([[p1[0], p1[1]]])
                
                angle1 = np.arctan2(p1[1] - y, p1[0] - x)
                angle2 = np.arctan2(p2[1] - y, p2[0] - x)
                
                # Ensure we go the shorter way around the circle
                if abs(angle2 - angle1) > np.pi:
                    if angle1 < angle2:
                        angle1 += 2 * np.pi
                    else:
                        angle2 += 2 * np.pi
                
                # Create intermediate points with wavy perturbations
                for j in range(1, num_interp):
                    t = j / num_interp
                    
                    base_x = p1[0] * (1 - t) + p2[0] * t
                    base_y = p1[1] * (1 - t) + p2[1] * t
                    
                    r = np.sqrt((base_x - x)**2 + (base_y - y)**2)
                    interp_angle = angle1 * (1 - t) + angle2 * t
                    
                    # Apply wave pattern
                    wave_freq = np.random.uniform(2, 8)
                    wave_amp = r * np.random.uniform(0.08, 0.25)
                    wave_effect = wave_amp * np.sin(wave_freq * np.pi * t)
                    
                    # Apply perpendicular offset
                    perp_angle = interp_angle + np.pi/2
                    wavy_x = base_x + wave_effect * np.cos(perp_angle)
                    wavy_y = base_y + wave_effect * np.sin(perp_angle)
                    
                    wavy_x = np.clip(wavy_x, rect_coords[0] + 5, rect_coords[2] - 5)
                    wavy_y = np.clip(wavy_y, rect_coords[1] + 5, rect_coords[3] - 5)
                    points.append([[int(wavy_x), int(wavy_y)]])
            
            # Create mask for main splatter
            pts_array = np.array(points, dtype=np.int32)
            cv2.fillPoly(main_binary_mask, [pts_array], 1.0)
            cv2.fillPoly(main_effect_mask, [pts_array], 1.0)
            
            # Blur only the effect mask
            main_effect_mask = gaussian_filter(main_effect_mask, sigma=main_blur)
        
        # Generate droplets if enabled
        if generate_droplets:
            num_droplets = np.random.randint(*droplet_count_range)
            for _ in range(num_droplets):
                drop_angle = np.random.uniform(0, 2*np.pi)
                drop_dist = np.random.uniform(*droplet_distance_range) * size
                
                drop_x = int(x + drop_dist * np.cos(drop_angle))
                drop_y = int(y + drop_dist * np.sin(drop_angle))
                
                size_factor = np.random.uniform(*droplet_size_range)
                drop_size = max(1, int(image_diagonal * size_factor))
                
                drop_x = np.clip(drop_x, rect_coords[0] + drop_size, rect_coords[2] - drop_size)
                drop_y = np.clip(drop_y, rect_coords[1] + drop_size, rect_coords[3] - drop_size)
                
                cv2.circle(droplet_binary_mask, (drop_x, drop_y), drop_size, 1.0, -1)
                cv2.circle(droplet_effect_mask, (drop_x, drop_y), drop_size, 1.0, -1)
            
            # Blur only the effect mask
            droplet_effect_mask = gaussian_filter(droplet_effect_mask, sigma=droplet_blur)
        
        # Combined masks
        combined_effect = np.maximum(main_effect_mask, droplet_effect_mask)
        combined_binary = np.maximum(main_binary_mask, droplet_binary_mask)
        
        return (main_effect_mask, droplet_effect_mask, combined_effect, 
                main_binary_mask, droplet_binary_mask, combined_binary, 
                (x, y, size))
    
    def generate_density_mask(mask, x, y, grad_direction):
        """
        Create density variation within the splatter with configurable gradient direction.
        """
        y_coords, x_coords = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
        
        max_dist = np.max(dist_from_center * mask) if np.max(mask) > 0 else 1
        norm_dist = dist_from_center / max_dist
        
        # Create gradient with high contrast between center and edge
        if grad_direction == "center_dark":
            gradient = 0.9 - 0.5 * np.power(norm_dist, 0.9)
        else:  # "edge_dark"
            gradient = 0.1 + 0.9 * np.power(norm_dist, 1.3)
        
        # Generate enhanced noise
        noise1 = generate_enhanced_noise((height, width), octaves=7, persistence=0.65, 
                                        scale=max_dist/2, smoothness=2.0)
        noise2 = generate_enhanced_noise((height, width), octaves=4, persistence=0.5, 
                                        scale=max_dist/8, smoothness=1.5)
        
        combined_noise = 0.7 * noise1 + 0.3 * noise2
        
        # Blend gradient with noise
        blend_factor = 0.65
        density = blend_factor * gradient + (1 - blend_factor) * combined_noise
        
        return density * mask
    
    # Generate all masks
    all_main_effect_masks = []
    all_droplet_effect_masks = []
    all_main_binary_masks = []
    all_droplet_binary_masks = []
    all_splatter_info = []
    
    for _ in range(num_splatters):
        x = np.random.randint(rect_coords[0] + 20, rect_coords[2] - 20)
        y = np.random.randint(rect_coords[1] + 20, rect_coords[3] - 20)
        size = np.random.randint(min_size, max_size)
        
        main_effect, droplet_effect, combined_effect, main_binary, droplet_binary, combined_binary, info = generate_splatter_shape(
            x, y, size, 
            droplet_count_range,
            droplet_size_range,
            droplet_distance_range
        )
        
        all_main_effect_masks.append(main_effect)
        all_droplet_effect_masks.append(droplet_effect)
        all_main_binary_masks.append(main_binary)
        all_droplet_binary_masks.append(droplet_binary)
        all_splatter_info.append(info)
    
    # Create combined binary masks
    combined_main_binary = np.zeros((height, width), dtype=np.float32)
    if generate_main_splatter:
        for mask in all_main_binary_masks:
            combined_main_binary = np.maximum(combined_main_binary, mask)
    
    combined_all_binary = combined_main_binary.copy()
    if generate_droplets:
        for mask in all_droplet_binary_masks:
            combined_all_binary = np.maximum(combined_all_binary, mask)
    
    # Convert to uint8
    main_binary_mask_uint8 = (combined_main_binary > 0.5).astype(np.uint8) * 255
    all_binary_mask_uint8 = (combined_all_binary > 0.5).astype(np.uint8) * 255
    
    # Create combined effect masks
    combined_main_effect = np.zeros((height, width), dtype=np.float32)
    for mask in all_main_effect_masks:
        combined_main_effect = np.maximum(combined_main_effect, mask)
    
    combined_droplet_effect = np.zeros((height, width), dtype=np.float32)
    for mask in all_droplet_effect_masks:
        combined_droplet_effect = np.maximum(combined_droplet_effect, mask)
    
    # Find connected components for consistent coloring
    num_labels, labels = cv2.connectedComponents(main_binary_mask_uint8, connectivity=8)
    
    # Create shadow effect
    shadow_mask = np.zeros((height, width), dtype=np.float32)
    if shadow_intensity > 0 and generate_main_splatter and np.max(main_binary_mask_uint8) > 0:
        effective_shadow_radius = min(10.0, max(0.5, shadow_radius_factor))
        
        kernel_size = max(3, int(effective_shadow_radius * 2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        eroded = cv2.erode(main_binary_mask_uint8, kernel)
        edge_mask = main_binary_mask_uint8 - eroded
        
        shadow_mask = gaussian_filter(edge_mask.astype(float), sigma=effective_shadow_radius)
        
        if shadow_mask.max() > 0:
            shadow_mask = shadow_mask / shadow_mask.max()
            shadow_mask = np.power(shadow_mask, 0.5)
        
        # Add noise for texture
        if shadow_noise > 0:
            noise_size = (np.array(shadow_mask.shape) // 4).astype(np.int32)
            noise_size = np.maximum(noise_size, 1)
            noise = np.random.random(noise_size)
            noise = cv2.resize(noise, (width, height))
            shadow_mask = shadow_mask * (1.0 - shadow_noise * 0.3 + shadow_noise * 0.6 * noise)
            shadow_mask = np.clip(shadow_mask, 0, 1.0)
        
        # Apply additional blur
        if shadow_blur > 0:
            shadow_mask = gaussian_filter(shadow_mask, sigma=shadow_blur)
    
    # Create light/halo effect
    halo = np.zeros((height, width), dtype=np.uint8)
    if light_intensity > 0 and np.max(combined_main_binary) > 0:
        kernel_size = int(max(min_size * light_radius_factor * 0.1, 5))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        halo = cv2.dilate(main_binary_mask_uint8, kernel)
        
        halo[main_binary_mask_uint8 > 0] = 0
        halo = cv2.blur(halo, (7, 7))
        
        if halo_noise > 0:
            ran_size = (np.array(halo.shape) // 6).astype(np.int32)
            ran_size = np.maximum(ran_size, 1)
            ran = np.random.random(ran_size)
            ran = cv2.resize(ran, (width, height))
            halo = halo.astype(float) * ran
        
        halo = halo * light_intensity
        halo = np.clip(halo, 0, 255).astype(np.uint8)
    
    # Process main splatter connected components
    if generate_main_splatter:
        for label in range(1, num_labels):
            component_mask = (labels == label).astype(np.float32)
            
            if np.sum(component_mask) == 0:
                continue
                
            y_indices, x_indices = np.nonzero(component_mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                center_x = np.mean(x_indices)
                center_y = np.mean(y_indices)
                
                dists = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
                approx_size = np.max(dists) if len(dists) > 0 else 30
                
                darkness = np.random.uniform(*main_darkness_range)
                component_effect_mask = component_mask * combined_main_effect
                
                # Generate color noise
                color_noise1 = generate_enhanced_noise((height, width), octaves=6, persistence=0.65, 
                                                    scale=approx_size/2, smoothness=1.8)
                color_noise2 = generate_enhanced_noise((height, width), octaves=3, persistence=0.5, 
                                                    scale=approx_size/6, smoothness=1.0)
                color_noise = 0.6 * color_noise1 + 0.4 * color_noise2

                darkness_factor = darkness + 0.35 * color_noise
                
                density_mask = generate_density_mask(component_effect_mask, center_x, center_y, gradient_direction)
                
                # Create and apply colors
                component_colors = np.zeros((height, width, 3), dtype=float)
                for c in range(3):
                    component_colors[:, :, c] = color_base[c] * darkness_factor
                
                for c in range(3):
                    channel = result[:,:,c].astype(float)
                    channel = channel * (1 - density_mask) + component_colors[:,:,c] * density_mask
                    result[:,:,c] = np.clip(channel, 0, 255).astype(np.uint8)
    
    # Process droplets individually
    if generate_droplets:
        for i, (droplet_mask, droplet_binary) in enumerate(zip(all_droplet_effect_masks, all_droplet_binary_masks)):
            if np.max(droplet_mask) > 0:
                x, y, size = all_splatter_info[i]
                
                # Remove overlapping areas with main splatter
                overlap_mask = droplet_binary * combined_main_binary
                if np.max(overlap_mask) > 0:
                    valid_droplet_mask = droplet_mask * (1 - combined_main_binary)
                    valid_droplet_binary = droplet_binary * (1 - combined_main_binary)
                    if np.max(valid_droplet_mask) <= 0:
                        continue
                else:
                    valid_droplet_mask = droplet_mask
                    valid_droplet_binary = droplet_binary
                    
                # Generate noise for color variation
                color_noise1 = generate_noise_texture((height, width), scale=approx_size/3, octaves=8, persistence=0.65)
                color_noise2 = generate_noise_texture((height, width), scale=approx_size/8, octaves=4, persistence=0.6)
                color_noise = 0.6 * color_noise1 + 0.4 * color_noise2
                
                droplet_darkness = np.random.uniform(*droplet_darkness_range)
                droplet_density = gaussian_filter(valid_droplet_mask, sigma=1)
                droplet_darkness_factor = droplet_darkness + 0.25 * color_noise
                
                # Create and apply droplet colors
                droplet_colors = np.zeros((height, width, 3), dtype=float)
                for c in range(3):
                    droplet_colors[:, :, c] = color_base[c] * droplet_darkness_factor
                
                for c in range(3):
                    channel = result[:,:,c].astype(float)
                    channel = channel * (1 - droplet_density) + droplet_colors[:,:,c] * droplet_density
                    result[:,:,c] = np.clip(channel, 0, 255).astype(np.uint8)
    
    # Apply shadow as a separate effect
    if shadow_intensity > 0 and generate_main_splatter and np.max(shadow_mask) > 0:
        effective_shadow_intensity = shadow_intensity
        
        for c in range(3):
            shadow_effect = shadow_mask * effective_shadow_intensity
            result[:,:,c] = np.clip(
                result[:,:,c].astype(float) * (1.0 - shadow_effect),
                0, 255
            ).astype(np.uint8)
    
    # Apply halo to all channels
    if light_intensity > 0:
        for c in range(3):
            result[:,:,c] = np.clip(result[:,:,c].astype(float) + halo, 0, 255).astype(np.uint8)
    
    return result, all_binary_mask_uint8


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




