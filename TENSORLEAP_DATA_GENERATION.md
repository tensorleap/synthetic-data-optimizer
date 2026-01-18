# Tensorleap Data Generation Guide

This guide explains how to generate synthetic void datasets for Tensorleap integration and optimization.

## Overview

The optimization workflow has two phases:
1. **Epoch 0** - Initial dataset with real + synthetic data (with train/val/test splits)
2. **Epoch > 0** - Synthetic-only datasets based on optimizer recommendations (no splits)

## Epoch 0: Initial Dataset Generation

Generate the initial dataset with real data and synthetic data from predefined distributions.

### Command
```bash
poetry run python -m scripts.generate_tensorleap_data
```

### Output Structure
```
data/tensorleap_dataset/
├── train/
│   ├── real/                (70 images + 70 masks)
│   ├── synthetic_circle/    (images + masks)
│   ├── synthetic_ellipse/   (images + masks)
│   └── synthetic_irregular/ (images + masks)
├── val/
│   └── (same structure)
├── test/
│   └── (same structure)
└── csv/
    └── metadata.csv
```

### Customization

Edit [scripts/generate_tensorleap_data.py](scripts/generate_tensorleap_data.py) to modify:
- Distribution parameters per shape (`circle_params`, `ellipse_params`, `irregular_params`)
- Sample counts (`n_samples_per_shape_dict`)
- Real data distribution type (`real_distribution_type`)

## Epoch > 0: Generate from Tensorleap Recommendations

After running optimization in Tensorleap, generate new synthetic data based on optimizer suggestions.

### Step 1: Get Recommendations CSV from Tensorleap

Download the suggestions/recommendations CSV from Tensorleap and save it to:
```
data/suggestions-<name>-epoch<N>.csv
```

Example: `data/suggestions-more_distributions-epoch0.csv`

### Step 2: Update Script

Edit [scripts/generate_from_recommendations.py](scripts/generate_from_recommendations.py):

```python
recommendations_csv = Path(__file__).parent.parent / "data" / "suggestions-your-file.csv"
output_dir = Path(__file__).parent.parent / "data" / "tensorleap_dataset_epoch1"
```

Change `distribution_id='dist_1'` if you want a different distribution from the CSV.

### Step 3: Generate Dataset

```bash
poetry run python -m scripts.generate_from_recommendations
```

### Output Structure
```
data/tensorleap_dataset_epoch1/
├── synthetic_circle/      (images + masks)
├── synthetic_ellipse/     (images + masks)
├── synthetic_irregular/   (images + masks)
└── csv/
    └── metadata.csv
```

**No train/val/test splits** - flat structure with only synthetic data.

## CSV Format

Both epoch 0 and epoch > 0 generate `metadata.csv` with these columns:

- `script_name` - Shape identifier ("mixed", "circle", "ellipse", "irregular")
- `image_name` - Unique filename (e.g., "circle_0000.png")
- `mask_name` - Mask filename (e.g., "circle_0000_mask.png")
- `package_type` - Always "test"
- `dataset_split` - "train", "val", "test" (epoch 0 only)
- Distribution parameters: `void_count_mean`, `void_count_std`, `base_size_mean`, etc.

## How It Works

### Epoch 0
Uses predefined distribution parameters for each shape type with manual configuration.

### Epoch > 0
1. Reads Tensorleap recommendations CSV
2. Extracts parameters for specified `distribution_id`
3. Maps `simulation_1`→circle, `simulation_2`→ellipse, `simulation_3`→irregular
4. Normalizes sample counts to ~300 total (maintains proportions)
5. Enforces minimum 20 samples per shape
6. Uses defaults for missing parameters
7. Generates synthetic-only dataset

## Files

- [scripts/generate_tensorleap_data.py](scripts/generate_tensorleap_data.py) - Epoch 0 generation
- [scripts/generate_from_recommendations.py](scripts/generate_from_recommendations.py) - Epoch > 0 generation
- [scripts/translate_tensorleap_recommendations.py](scripts/translate_tensorleap_recommendations.py) - CSV translation utilities
- [src/data_generation/tensorleap_data_generator.py](src/data_generation/tensorleap_data_generator.py) - Core generator
- [src/embedding/unet_embedder.py](src/embedding/unet_embedder.py) - Production UNet embedder (if needed locally)

## Notes

- All images are grayscale PNG files (250×220 pixels)
- Binary masks indicate void locations (1=void, 0=background)
- Shapes are conditional parameters - each synthetic distribution uses ONE specific shape
- Real data uses mixed shapes (target distribution)
