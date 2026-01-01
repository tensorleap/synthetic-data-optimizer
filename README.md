# Synthetic Data Optimizer

Parameter calibration for synthetic data generation via iterative latent space optimization.

## Overview

This tool optimizes synthetic data generation parameters by iteratively refining them based on latent space feedback. It uses Bayesian optimization to discover parameter settings that make synthetic samples match real data distribution in latent space.

## Architecture

- **data_generation/**: Generate synthetic void images from parameters
- **embedding/**: DiNOv2 embedder and PCA projector
- **optimization/**: Parameter optimizer (Bayesian optimization)
- **orchestration/**: Full pipeline orchestration
- **visualization/**: Latent space and convergence plots

## Setup

1. Install dependencies with Poetry:
```bash
poetry install
```

2. Base chip images are located in `data/base_chips/`

3. Configure experiment in `configs/experiment_config.yaml`

## Usage

Coming soon after implementation.

## Documentation

See `plans/` directory for detailed design documents:
- [parameter_calibration_approach.md](plans/parameter_calibration_approach.md) - Original vision
- [infrastructure_design.md](plans/infrastructure_design.md) - Implementation design
