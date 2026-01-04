# Synthetic Data Optimizer - Infrastructure Design

## Overview

Self-contained pipeline for optimizing synthetic data generation parameters through iterative latent space optimization. All components (generation, embedding, PCA, optimization) contained in this repository.

---

## Architecture

```
synthetic-data-optimizer/
├── src/
│   ├── data_generation/
│   │   ├── void_generator.py           # Generate void images from params
│   │   ├── parameter_sampler.py        # Sample real/close/far distributions
│   │   └── base_image_loader.py        # Load base chip images
│   │
│   ├── embedding/
│   │   ├── dinov2_embedder.py          # DiNOv2 model wrapper
│   │   └── pca_projector.py            # Fit PCA (iter 0), transform (iter 1+)
│   │
│   ├── optimization/
│   │   ├── optimizer.py                # BLACK BOX - yields next params + current best
│   │   │                               # Interface defined, implementation TBD
│   │   └── metrics.py                  # Distribution similarity, distances
│   │
│   ├── orchestration/
│   │   ├── pipeline.py                 # Full iteration loop - no human in loop
│   │   └── state_manager.py            # Persist state across iterations
│   │
│   └── visualization/
│       ├── latent_plots.py             # 2D PCA scatter plots
│       └── convergence_plots.py        # Progress tracking
│
├── configs/
│   └── experiment_config.yaml          # All parameters, sizes, paths
│
├── data/
│   ├── base_chips/                     # 5 NoVoids images (copied to repo)
│   └── experiments/
│       └── {experiment_id}/
│           ├── iteration_0/
│           │   ├── images/             # Generated images
│           │   ├── embeddings.npy      # DiNOv2 embeddings
│           │   ├── pca_model.pkl       # Fitted PCA (saved at iter 0)
│           │   ├── pca_2d.npy          # 2D projections
│           │   └── metadata.json       # Parameters, dataset_type, etc.
│           ├── iteration_1/
│           └── ...
│
├── notebooks/
│   └── analysis.ipynb                  # Interactive exploration
│
├── plans/
│   ├── parameter_calibration_approach.md  # Original vision document
│   └── infrastructure_design.md           # This document
│
├── pyproject.toml                      # Poetry dependencies
└── README.md
```

---

## Parameter Schema

### 6 Optimizable Parameters

**Deterministic Parameters** (single values used directly):
1. **`void_shape`** - categorical: {'circle', 'ellipse', 'irregular'}
   - Implementation: Direct shape drawing function selection

2. **`void_count`** - integer: [3, 20]
   - Implementation: Number of void objects to draw (exactly N voids)

3. **`base_size`** - continuous: [5.0, 40.0] pixels
   - Implementation: Base radius for void drawing

4. **`brightness_factor`** - continuous: [0.3, 0.8]
   - Implementation: Multiply pixel values (0=black, 1=unchanged)

**Stochastic Parameters** (control amount of randomness):
5. **`size_std`** - continuous: [0.0, 15.0] pixels
   - Implementation: Each void size = base_size + N(0, size_std)

6. **`position_spread`** - continuous: [0.1, 0.8] normalized
   - Implementation: Void positions sampled from uniform([0, position_spread * img_size])

### Uncontrolled Parameters (not optimized)

These parameters are fixed or random per image to simulate real-world constraints:
- **`rotation`**: uniform(0, 360°) - random per image
- **`edge_blur`**: 2 pixels - fixed

**Rationale**: Client cannot control everything in real manufacturing. Tests if optimizer can find good parameters despite uncontrollable noise.

---

## Parameter Set Example

```python
# A single parameter set (what optimizer recommends)
param_set = {
    'void_shape': 'ellipse',           # categorical value
    'void_count': 7,                   # exactly 7 voids
    'base_size': 18.5,                 # base size in pixels
    'brightness_factor': 0.55,         # brightness reduction
    'size_std': 6.2,                   # controls size randomness
    'position_spread': 0.45,           # controls position randomness
}

# Generate 3 replications with seeds [0, 1, 2]
for seed in [0, 1, 2]:
    img = generate_void_image(base_chip, param_set, seed)
    # Each image has:
    # - Exactly 7 voids (deterministic)
    # - Base size 18.5, brightness 0.55 (deterministic)
    # - Different actual sizes and positions (stochastic, seed-dependent)
```

---

## Dataset Configuration

### Iteration 0: Initialize with Real/Close/Far

```python
DATASET_CONFIG = {
    'real': {
        'size': 16,                    # number of samples
        'replications': 3,             # 3 samples per param set
        'param_sets': 5,               # 5 unique param sets (5*3 = 15 images, ~16)
        'source': 'Sample from REAL_DISTRIBUTION'
    },

    'close': {
        'size': 24,                    # 8 param sets * 3 replications
        'param_sets': 8,
        'replications': 3,
        'source': 'Sample from CLOSE_DISTRIBUTION (~20% shift from real)'
    },

    'far': {
        'size': 24,                    # 8 param sets * 3 replications
        'param_sets': 8,
        'replications': 3,
        'source': 'Sample from FAR_DISTRIBUTION (~50%+ shift from real)'
    }
}
```

**Total iteration 0**: ~64 images (16 real + 24 close + 24 far)

### Iterations 1+: Optimization Loop

```python
ITERATION_CONFIG = {
    'param_sets_per_iteration': 8,     # test 8 new parameter configurations
    'replications': 3,                 # 24 new images per iteration
    'max_iterations': 10,              # total ~300 images if all iterations run
    'convergence_threshold': 0.05,     # distribution similarity metric
    'early_stop_patience': 3,          # stop if no improvement for N iterations
}
```

---

## Parameter Distributions

**Note**: These are initial guesses. Will be tuned empirically after embedding model is working.

```python
# Real distribution (ground truth to discover)
REAL_DISTRIBUTION = {
    'void_shape': {
        'probabilities': {'circle': 0.2, 'ellipse': 0.5, 'irregular': 0.3}
    },
    'void_count': {'mean': 8, 'std': 3},
    'base_size': {'mean': 20, 'std': 5},
    'brightness_factor': {'mean': 0.6, 'std': 0.1},
    'size_std': {'mean': 7, 'std': 2},
    'position_spread': {'mean': 0.5, 'std': 0.15},
}

# Close distribution (slightly perturbed)
CLOSE_DISTRIBUTION = {
    'void_shape': {
        'probabilities': {'circle': 0.25, 'ellipse': 0.45, 'irregular': 0.3}
    },
    'void_count': {'mean': 10, 'std': 3},
    'base_size': {'mean': 18, 'std': 5},
    'brightness_factor': {'mean': 0.55, 'std': 0.1},
    'size_std': {'mean': 8, 'std': 2},
    'position_spread': {'mean': 0.55, 'std': 0.15},
}

# Far distribution (significantly different)
FAR_DISTRIBUTION = {
    'void_shape': {
        'probabilities': {'circle': 0.7, 'ellipse': 0.2, 'irregular': 0.1}
    },
    'void_count': {'mean': 15, 'std': 3},
    'base_size': {'mean': 10, 'std': 5},
    'brightness_factor': {'mean': 0.4, 'std': 0.1},
    'size_std': {'mean': 3, 'std': 1},
    'position_spread': {'mean': 0.8, 'std': 0.1},
}
```

---

## Component Details

### 1. Void Generator

```python
class VoidGenerator:
    """Generate synthetic void images by overlaying on base chips"""

    def __init__(self, base_image_dir: Path):
        self.base_images = load_base_images(base_image_dir)  # 5 NoVoids images

    def generate_batch(
        self,
        param_sets: List[Dict],
        replications: int = 3,
        save_dir: Path = None
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Args:
            param_sets: List of parameter dictionaries
            replications: Samples per param set (different seeds)

        Returns:
            images: List of generated images
            metadata: List of dicts with {params, seed, sample_id, base_image_id}
        """

    def generate_single(
        self,
        base_chip: np.ndarray,
        params: Dict,
        seed: int
    ) -> np.ndarray:
        """
        Generate single void image

        Implementation:
        1. Randomly select base chip from 5 images (tracked in metadata)
        2. Set random seed for reproducibility
        3. Apply deterministic params directly (void_count, base_size, etc.)
        4. Sample from stochastic distributions (size_std, position_spread)
        5. Apply uncontrolled params (rotation=random, edge_blur=2)
        6. Draw voids on base chip
        """
```

**Base Image Selection**:
- Randomly select from 5 base chip images per sample
- Track selected base image ID in metadata
- Allows analysis of base image effect on embeddings

### 2. DiNOv2 Embedder

```python
class DinoV2Embedder:
    """Extract embeddings using DiNOv2 model"""

    def __init__(self, model_name: str = "dinov2_vitb14"):
        # Load pretrained DiNOv2 (768-dim embeddings)
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)

    def embed_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Returns:
            embeddings: (N, 768) array
        """
```

**Model Choice**:
- `dinov2_vitb14` (768-dim)
- Runs on M3 Mac with MPS (Metal Performance Shaders) backend
- Fallback to CPU if MPS unavailable

### 3. PCA Projector

```python
class PCAProjector:
    """Fit PCA on iteration 0, transform subsequent iterations"""

    def __init__(self, n_components: int = 2):
        self.pca = None  # sklearn PCA
        self.fitted = False

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit PCA on iteration 0 embeddings (all real+close+far)"""
        self.pca = PCA(n_components=2)
        pca_2d = self.pca.fit_transform(embeddings)
        self.fitted = True
        return pca_2d

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform new embeddings using FIXED PCA from iteration 0"""
        assert self.fitted, "PCA must be fitted first"
        return self.pca.transform(embeddings)

    def save(self, path: Path):
        """Persist PCA model for iteration 0"""
        joblib.dump(self.pca, path)
```

**Critical Design**: PCA fitted once on iteration 0, all subsequent iterations use same projection for cross-iteration comparison.

### 4. Parameter Optimizer (BLACK BOX)

```python
class ParameterOptimizer:
    """
    BLACK BOX: Interface defined, implementation TBD

    Future implementation: Bayesian optimization with GP surrogate model
    """

    def __init__(self, param_bounds: Dict, acquisition_function: str = "ucb"):
        self.param_bounds = param_bounds
        self.acquisition_fn = acquisition_function

    def recommend_next_batch(
        self,
        all_embeddings: np.ndarray,       # cumulative from all iterations
        all_metadata: List[Dict],         # cumulative metadata
        target_embeddings: np.ndarray,    # real samples (fixed)
        batch_size: int = 8
    ) -> Dict:
        """
        Returns:
        {
            'next_params': List[Dict],      # 8 param sets to try next
            'current_best': List[Dict],     # top 5 param sets so far
            'scores': {
                'distribution_similarity': float,  # MMD or Wasserstein
                'per_sample_distances': np.ndarray
            }
        }
        """
        # PLACEHOLDER: Random sampling for now
        # TODO: Implement GP + UCB/EI acquisition
```

### 5. Orchestration Pipeline

```python
class OptimizationPipeline:
    """Fully automated iteration loop - no human in the middle"""

    def run_full_experiment(self, max_iterations: int = 10):
        """
        Main entry point:
        1. Run iteration 0 (real/close/far)
        2. Fit PCA on iteration 0
        3. Loop: recommend → generate → embed → transform → check convergence
        4. Generate final report
        """

    def run_iteration_0(self) -> Dict:
        """
        Initialize: generate real/close/far datasets, fit PCA

        Steps:
        1. Sample parameters from real/close/far distributions
        2. Generate images with replications
        3. Extract DiNOv2 embeddings
        4. Fit PCA on all iteration 0 data
        5. Save iteration 0 state (images, embeddings, PCA model, metadata)
        """

    def run_iteration_n(self, iteration: int, param_sets: List[Dict]) -> Dict:
        """
        Run iteration N:
        1. Generate images with recommended params
        2. Extract embeddings
        3. Transform with FIXED PCA from iteration 0
        4. Save iteration N state
        """

    def check_convergence(self, scores: Dict) -> bool:
        """
        Convergence criteria:
        - Threshold: distribution_similarity < 0.05
        - Plateau: no improvement for 3 consecutive iterations
        """
```

---

## Metadata Schema

Each sample has associated metadata:

```python
{
    'sample_id': 'iter0_real_000',
    'iteration': 0,
    'dataset_type': 'real',  # 'real', 'close', 'far', 'synthetic'
    'param_set_id': 'ps_000',
    'replication_seed': 0,
    'base_image_id': 'NoVoids_53438_0003_result.png',
    'params': {
        'void_shape': 'ellipse',
        'void_count': 7,
        'base_size': 18.5,
        'brightness_factor': 0.55,
        'size_std': 6.2,
        'position_spread': 0.45,
    },
    'uncontrolled_params': {
        'rotation': 127.3,  # sampled per image
        'edge_blur': 2      # fixed
    },
    'timestamp': '2026-01-01T10:30:00Z',
}
```

---

## Configuration File

**configs/experiment_config.yaml**

```yaml
# Experiment setup
experiment_name: "infineon_void_poc"
experiment_dir: "data/experiments/infineon_void_poc"

# Base data
base_image_dir: "data/base_chips"  # 5 NoVoids images

# Model settings
dino_model: "dinov2_vitb14"  # 768-dim embeddings
pca_components: 2            # for visualization

# Dataset sizes
real:
  param_sets: 5
  replications: 3
  total_size: 16  # ~5*3

close:
  param_sets: 8
  replications: 3
  total_size: 24

far:
  param_sets: 8
  replications: 3
  total_size: 24

# Iteration settings
iteration_batch_size: 8      # param sets per iteration
replications_per_iteration: 3
max_iterations: 10

# Parameter space (search bounds)
param_bounds:
  void_shape: ['circle', 'ellipse', 'irregular']
  void_count: [3, 20]
  base_size: [5.0, 40.0]
  brightness_factor: [0.3, 0.8]
  size_std: [0.0, 15.0]
  position_spread: [0.1, 0.8]

# Parameter distributions (initial guesses, tune empirically)
param_distributions:
  real:
    void_shape: {circle: 0.2, ellipse: 0.5, irregular: 0.3}
    void_count: {mean: 8, std: 3}
    base_size: {mean: 20, std: 5}
    brightness_factor: {mean: 0.6, std: 0.1}
    size_std: {mean: 7, std: 2}
    position_spread: {mean: 0.5, std: 0.15}

  close:
    void_shape: {circle: 0.25, ellipse: 0.45, irregular: 0.3}
    void_count: {mean: 10, std: 3}
    base_size: {mean: 18, std: 5}
    brightness_factor: {mean: 0.55, std: 0.1}
    size_std: {mean: 8, std: 2}
    position_spread: {mean: 0.55, std: 0.15}

  far:
    void_shape: {circle: 0.7, ellipse: 0.2, irregular: 0.1}
    void_count: {mean: 15, std: 3}
    base_size: {mean: 10, std: 5}
    brightness_factor: {mean: 0.4, std: 0.1}
    size_std: {mean: 3, std: 1}
    position_spread: {mean: 0.8, std: 0.1}

# Convergence criteria
convergence_threshold: 0.05  # distribution similarity metric
early_stop_patience: 3       # stop if no improvement for N iterations

# Optimization (placeholder)
acquisition_function: "ucb"  # or "ei", "pi"
```

---

## Dependencies

**pyproject.toml**

```toml
[tool.poetry]
name = "synthetic-data-optimizer"
version = "0.1.0"
description = "Parameter calibration for synthetic data via latent space optimization"
python = "^3.10"

[tool.poetry.dependencies]
python = "^3.10"
torch = "^2.0.0"
torchvision = "^0.15.0"
numpy = "^1.24.0"
scikit-learn = "^1.3.0"
Pillow = "^10.0.0"
matplotlib = "^3.7.0"
seaborn = "^0.12.0"
pyyaml = "^6.0"
joblib = "^1.3.0"
tqdm = "^4.65.0"
opencv-python = "^4.8.0"

# Optional: for Bayesian optimization later
# botorch = "^0.9.0"
# gpytorch = "^1.11"

[tool.poetry.dev-dependencies]
jupyter = "^1.0.0"
pytest = "^7.4.0"
```

---

## Implementation Phases

### Phase 1: Foundation (No ML dependencies)
1. Project structure - directories, configs
2. Parameter sampler - sample real/close/far distributions
3. Void generator - overlay voids on base chips
4. Copy base images to repo

**Deliverable**: Can generate synthetic void images with controllable parameters

### Phase 2: ML Components
1. DiNOv2 embedder - load model, extract embeddings
2. PCA projector - fit/transform logic, persistence
3. Metrics module - distribution similarity, distances

**Deliverable**: Can extract embeddings and project to 2D

### Phase 3: Integration
1. Optimizer stub - interface defined, random sampling placeholder
2. Pipeline orchestration - iteration 0 + iteration loop
3. State manager - save/load iteration data

**Deliverable**: Full pipeline runs end-to-end with random optimization

### Phase 4: Visualization & Analysis
1. Latent plots - 2D scatter with real/close/far/iterations
2. Convergence tracking - metrics over iterations
3. Analysis notebook - interactive exploration

**Deliverable**: Can visualize and analyze optimization progress

### Phase 5: Real Optimization (Future)
1. Implement GP surrogate model
2. Implement acquisition functions (UCB/EI)
3. Replace random sampling with Bayesian optimization

**Deliverable**: Real parameter optimization

---

## Success Metrics

**Distribution-Level**:
- Synthetic vs real divergence (MMD/Wasserstein) approaches threshold
- Visual clustering: synthetic samples overlay real distribution in 2D PCA

**Per-Sample**:
- Mean/median distance to nearest real decreases across iterations
- Percentage of synthetic samples within ε-ball of real samples increases

**Convergence**:
- Optimization converges within 10 iterations
- Early stopping triggered when no improvement for 3 consecutive iterations

---

## Notes

- All components self-contained in this repo
- No human in the loop during iteration pipeline
- Parameter distributions will be tuned empirically after first runs
- Optimization algorithm is black box for now (random sampling)
- Base images randomly selected per sample and tracked in metadata
