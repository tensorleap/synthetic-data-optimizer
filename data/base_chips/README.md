# Base Chip Images

This directory contains clean chip images (no voids) used as base for synthetic void generation.

## Files

- 5 chip images: `NoVoids_53438_00XX_result.png`
- 5 corresponding masks: `NoVoids_53438_00XX_result_mask.png`

## Source

Copied from: `/Users/ranhomri/tensorleap/data/infineon-cv/sample_void_detection_data_toTensorLeap/NoVoids_real_data/`

## Usage

The void generator randomly selects from these 5 base images when creating synthetic void images. Each selection is tracked in metadata for analysis.
