# Synthetic Data Calibration Tool via Iterative Latent Space Optimization

## Vision

**General-Purpose Tool**: Optimize synthetic data generation parameters for any domain using latent space feedback and Bayesian optimization.

**Current POC**: Infineon void detection (5 DOF: location x,y, size, shape, color reduction %, void count)

**Goal**: Iteratively refine generation parameters until synthetic data distribution matches real data distribution in latent space.

---

## Tool Design

### Core Concept
Real data defines target distribution in latent space. Through iterative optimization, discover parameter settings that make synthetic samples cluster close to real samples.

### Inputs (Per Iteration)
1. **Data subsets**: Real samples, generated samples with labels
2. **Iteration index**: 0, 1, 2, ... (iteration 0 = initial synthetic batch)
3. **Generation parameters**: Dictionary/JSON per sample with n-dimensional parameters
4. **Latent embeddings**: Pre-computed latent space vectors per sample

### Outputs (Per Iteration)
1. **Distribution similarity score**: Single metric (MMD/Wasserstein) quantifying real vs synthetic divergence
2. **Per-sample scores**: Distance from each synthetic to nearest real sample
3. **Parameter recommendations**: Proposed n-dimensional parameter values/ranges for next iteration
4. **Visualization**: 2D projection (UMAP/t-SNE) showing real vs synthetic, fixed after iteration 0

---

## Iterative Optimization Pipeline

### Iteration 0: Baseline
1. Infer real + initial synthetic data → obtain latent embeddings
2. Fit 2D dimensionality reduction (UMAP/t-SNE) on combined data
3. Compute distribution similarity and per-sample scores
4. Initialize surrogate model (Gaussian Process) with `score = f(parameters)`
5. Visualize real vs synthetic in 2D latent space

### Iterations 1...N: Refinement
1. **Recommend**: Use acquisition function (UCB/EI) to propose next parameter batch
2. **Generate**: Client generates new synthetic samples with recommended parameters
3. **Project**: Infer new samples → project to **same** latent space as iteration 0
4. **Score**: Compute similarity metrics for new samples
5. **Update**: Add (parameters, scores) to surrogate model training data
6. **Visualize**: Overlay new samples on same 2D plot, track progress
7. **Converge**: Check stopping criteria (threshold met or plateau detected)

### Key Design Decisions

**Surrogate Model**: Gaussian Process
- Captures parameter interdependencies
- Provides uncertainty estimates for unexplored regions
- Enables principled exploration/exploitation trade-off

**Acquisition Function**: Upper Confidence Bound (UCB) with adaptive β
- `UCB(params) = predicted_score(params) + β × uncertainty(params)`
- Balances exploitation (good predictions) and exploration (high uncertainty)
- Start with high β (exploration), decay toward exploitation as iterations progress
- Alternative: Expected Improvement (EI) for faster convergence

**Fixed Latent Space**: Critical for cross-iteration comparison
- 2D reduction fitted once on iteration 0
- All subsequent iterations projected to same coordinate system
- Enables visual tracking of "movement" toward real distribution

---

## Tool Components (Modular Architecture)

### 1. Latent Space Analyzer
- **Input**: Real embeddings, synthetic embeddings, iteration index
- **Output**: Distribution similarity score, per-sample distances, 2D visualization
- **Metrics**:
  - Per-sample: Euclidean distance to nearest real (alternatives: cosine, Mahalanobis)
  - Distribution: MMD or Wasserstein distance
- **Visualization**: UMAP/t-SNE fitted on iteration 0, fixed thereafter

### 2. Parameter-Performance Learner (Surrogate Model)
- **Input**: Accumulated (parameters, scores) from all iterations
- **Model**: Gaussian Process Regression
  - Learns `similarity_score = f(param_1, ..., param_n)`
  - Captures parameter interactions and non-linearities
  - Outputs mean prediction + uncertainty (std dev)
- **Alternative**: Random Forest for non-probabilistic baseline

### 3. Parameter Recommender (Acquisition Optimizer)
- **Input**: Trained surrogate model, parameter bounds, acquisition function choice
- **Acquisition Functions**:
  - **UCB (default)**: `predicted_score + β × uncertainty` (balanced)
  - **Expected Improvement**: `E[max(0, score - best_so_far)]` (greedy)
  - **Probability of Improvement**: `P(score > best_so_far)` (conservative)
- **Output**: Batch of recommended parameter sets for next iteration
- **Optimization**: Grid search, random sampling, or gradient-based (depending on dimensionality)

### 4. Progress Tracker
- **Storage**: All iterations with parameters, scores, embeddings
- **Convergence Detection**:
  - Threshold: Distribution similarity < ε (user-defined)
  - Plateau: No improvement over k consecutive iterations
  - Budget: Max iterations reached
- **Visualization**: Convergence curves, parameter evolution heatmaps

---

## Success Metrics

**Distribution-Level**:
- Synthetic vs real divergence (MMD/Wasserstein) approaches baseline or threshold
- Visual clustering: synthetic samples overlay real distribution in 2D

**Per-Sample**:
- Mean/median distance to nearest real decreases across iterations
- Percentage of synthetic samples within ε-ball of real samples increases

**Downstream Task** (optional):
- Model trained on calibrated synthetic performs well on real test set

---

## Implementation Considerations

### For General Tool
1. **Parameter bounds**: Accept as input or infer from iteration 0 data
2. **Batch size**: Configurable number of samples per iteration (trade-off: exploration vs cost)
3. **Categorical parameters**: One-hot encode or use specialized kernels (e.g., Hamming distance)
4. **Multi-objective**: Extend to optimize for diversity within synthetic + closeness to real

### For Infineon POC
1. **Limited real data** (16 samples): Heavy emphasis on visualization, distribution metrics over per-sample
2. **5D parameter space**: Moderate dimensionality, GP should handle well
3. **Shape parameter**: Treat as categorical initially, consider shape descriptors if needed
4. **Convergence threshold**: To be defined empirically after iteration 0 baseline

---

## Scope

**In Scope**:
- Generic tool for n-dimensional parameter optimization
- Latent space analysis (embeddings provided externally)
- Bayesian optimization with configurable acquisition functions
- Visualization and progress tracking
- Infineon void detection as POC

**Out of Scope**:
- Model training/retraining (use existing models)
- Synthetic data generation pipeline (client responsibility)
- Latent space extraction (assume pre-computed embeddings)
- Domain-specific preprocessing or feature engineering

---

## Next Steps

1. **Architecture**: Implement modular components (Analyzer, Learner, Recommender, Tracker)
2. **POC Setup**: Load Infineon data (16 real, 38 synthetic), extract embeddings, parameter metadata
3. **Iteration 0**: Establish baseline metrics, visualize initial gap
4. **Iteration 1-3**: Test optimization loop with small batches, validate surrogate model
5. **Convergence**: Define stopping criteria based on observed improvement rates
6. **Generalization**: Test on secondary dataset to validate tool portability
