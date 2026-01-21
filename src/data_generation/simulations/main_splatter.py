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



