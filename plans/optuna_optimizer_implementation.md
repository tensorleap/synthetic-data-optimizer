# Implementation Plan: Optuna-Based Optimizer Module

## Objective
Replace the placeholder optimizer with an Optuna-based Bayesian optimization implementation that:
1. Optimizes multiple objectives (MMD, Wasserstein, NN distance) simultaneously
2. Outputs best parameter sets and next candidates after each iteration
3. Provides covariance matrix for parameter relationships
4. Works in a **decoupled workflow**: optimizer recommends parameters, client generates samples externally

## Critical Design Corrections

### ⚠️ Parameter Space Correction: void_shape
**IMPORTANT**: void_shape is NOT a categorical parameter. It's actually **3 continuous probability parameters**:
- `void_shape_circle_prob`: [0.0, 1.0] - probability of circle shape
- `void_shape_ellipse_prob`: [0.0, 1.0] - probability of ellipse shape
- `void_shape_irregular_prob`: [0.0, 1.0] - probability of irregular shape
- **Constraint**: Must sum to 1.0 (simplex constraint)

**Current implementation**: Uses `np.random.choice()` with probabilities (line 81 in parameter_sampler.py)

**Optimization approach**:
- Option A: Optimize 3 unconstrained values, then normalize to sum=1 (softmax)
- Option B: Optimize 2 values, compute 3rd as 1-sum (simplex parameterization)
- **Recommended**: Option B (2 free params, 3rd constrained)

### 📊 Actual Parameter Space (from config)
**Total: 7 optimized parameters**:
1. `void_shape_probs` → 2 free params (circle_prob, ellipse_prob; irregular = 1 - sum)
2. `void_count`: [1, 10] - integer
3. `base_size`: [5.0, 15.0] - continuous (precision: 1 decimal)
4. `rotation`: [0.0, 360.0] - continuous (precision: 1 decimal)
5. `center_x`: [0.2, 0.8] - continuous (precision: 2 decimals, fraction of image width)
6. `center_y`: [0.2, 0.8] - continuous (precision: 2 decimals, fraction of image height)
7. `position_spread`: [0.1, 0.8] - continuous (precision: 2 decimals)

### 🔄 Decoupled Workflow: Client-Driven Execution
**Real-world scenario**: Optimizer does NOT have access to sample generator.

**Workflow**:
1. **Optimizer outputs**: Best candidates (n) + Next candidates (m)
2. **Client decides**: Whether to run another iteration
3. **Client generates**: Samples for next iteration using recommended parameters
4. **Client provides**: New embeddings and parameters back to optimizer
5. **Repeat**: Optimizer updates model and suggests next batch

**Implication**: Optimizer must support:
- Saving/loading state between sessions
- Accepting externally-generated samples (not triggering generation)
- Clear separation: suggest → wait → update → suggest

**Why Optuna is PERFECT for this workflow**:
✅ **Persistent storage**: SQLite database stores all trial history
✅ **Ask/Tell interface**: `study.ask()` suggests trials, `study.tell()` updates results
✅ **Resumable**: Can load study from database and continue optimization
✅ **Stateless**: Optimizer doesn't need to keep state in memory between sessions
✅ **Client controls execution**: Client decides when to run iterations, optimizer just recommends

**This framework is IDEAL for your decoupled use case!**

## Summary: Optuna Capabilities for Your Requirements

### ✅ What Optuna Provides Out-of-the-Box:

1. **Best Candidates (n sets)**:
   - `study.best_trials` returns Pareto front for multi-objective optimization
   - Can rank and filter to get top n candidates
   - Full trial history with parameters and objective values

2. **Next Candidates (m sets)**:
   - `study.ask()` suggests next parameter configuration using TPE algorithm
   - Can call m times to get batch of suggestions
   - Balances exploration (try new regions) vs exploitation (refine good regions)

3. **Trial History & Metadata**:
   - All trials stored in SQLite database (persistent)
   - Each trial includes: parameters, objectives, state, timestamps, iteration info
   - Can filter, query, and analyze full optimization history

4. **Study State Persistence**:
   - Automatic save/load from database
   - Can resume experiments seamlessly
   - User attributes for custom metadata storage

### ⚠️ What Needs Custom Implementation:

1. **Covariance Matrix**:
   - Optuna doesn't expose surrogate model directly (TPE is not GP)
   - **Solution**: Compute empirical covariance from trial parameters
   - Straightforward: extract params → one-hot encode categoricals → `np.cov()`

2. **Trend Tracking Across Iterations**:
   - Optuna stores trials but not iteration-level summaries
   - **Solution**: Store best metrics per iteration in `study.user_attrs`
   - Build evolution timeline from stored attributes

3. **Output Formatting**:
   - Raw Optuna data needs formatting into your JSON specifications
   - **Solution**: Create utility functions to format outputs

## Overview of Approach

### Algorithm Selection
- **Primary sampler**: TPE (Tree-structured Parzen Estimator) via `optuna.samplers.TPESampler`
  - Excellent for continuous and integer parameters
  - Robust and battle-tested
  - **Note**: void_shape is NOT categorical - it's 3 continuous probability parameters
- **Multi-objective**: 3 objectives to minimize (mmd_rbf, wasserstein, mean_nn_distance)
- **Batch strategy**: Sequential ask (8 trials per iteration)
- **Replication handling**: Aggregate metrics via mean across 3 replications per config
- **Decoupled workflow**: Optimizer only recommends parameters, client generates samples externally

### Output Requirements Analysis

#### 1. Best Candidates (n sets showing improvement)
- **Optuna provides**: `study.best_trials` for Pareto front
- **Implementation**: Filter trials by dominated status, sort by objectives
- **Output**: Top n parameter sets from current Pareto front

#### 2. Next Candidates (m sets to try next)
- **Optuna provides**: `study.ask()` for next trial suggestions
- **Implementation**: Call ask() m times to get next batch
- **Output**: m parameter configurations for next iteration

#### 3. Covariance Matrix
- **Challenge**: Optuna/TPE doesn't expose surrogate model internals directly (TPE is not GP-based)
- **Solution**: Compute empirical covariance from trial history
  - Extract all completed trials' parameters
  - One-hot encode categorical parameters (void_shape)
  - Compute sample covariance matrix: `np.cov(param_matrix, rowvar=False)`
  - This shows parameter correlations discovered during optimization
  - Example: If trials with high base_size tend to have low brightness_factor, covariance shows this relationship

**Note**: Empirical covariance is interpretable and sufficient for understanding parameter relationships. If GP-based surrogate model covariance is needed later, we can switch to GPSampler and access BoTorch internals.

## Phase 1: Exploration (Already Completed)
✅ Understood current codebase structure
✅ Identified placeholder optimizer interface
✅ Analyzed parameter space and metrics
✅ Researched Optuna capabilities

## Phase 2: Design Details

### File Structure
```
src/optimization/
├── optimizer_placeholder.py          # DELETE or rename to _old
├── optuna_optimizer.py               # NEW: Main optimizer class
├── optuna_outputs.py                 # NEW: Output formatting utilities
└── metrics.py                        # EXISTING: Keep as-is
```

### Core Components

#### Component 1: OptunaOptimizer Class
**File**: `src/optimization/optuna_optimizer.py`

**Key methods (Decoupled API)**:
```python
class OptunaOptimizer:
    def __init__(self, config: Dict, exp_dir: str)
        # Initialize Optuna study with TPE sampler
        # Setup multi-objective optimization (3 objectives)
        # Configure storage backend: sqlite:///{exp_dir}/optuna.db
        # Load existing study if resuming

    @classmethod
    def load(cls, exp_dir: str) -> 'OptunaOptimizer':
        # Load existing optimizer from experiment directory
        # Resume from saved study state

    def store_baseline_metrics(self, baseline_metrics: Dict):
        # Store iteration 0 baseline for trend comparison
        # Save in study.user_attrs['baseline_metrics']

    def suggest_next_batch(self, batch_size: int = 8) -> List[Dict]:
        # Generate next batch of parameter configurations
        # Returns: List of param dicts with void_shape_probs, etc.
        # Does NOT trigger sample generation

    def update_with_results(
        self,
        parameters: List[Dict],      # Parameter sets that were evaluated
        metrics: List[Dict]           # Aggregated metrics per param set
    ):
        # Tell Optuna the results after client generates samples
        # Updates surrogate model for next suggestions
        # Stores iteration metrics in user_attrs for trend tracking

    def get_best_candidates(self, n: int) -> Dict:
        # Return top n from Pareto front with full trend history
        # Format matches best_candidates.json spec

    def get_next_candidates_output(self, batch_size: int = 8) -> Dict:
        # Wrapper around suggest_next_batch with output formatting
        # Format matches next_candidates.json spec

    def get_parameter_covariance(self) -> Tuple[np.ndarray, List[str]]:
        # Compute empirical covariance matrix (8×8)
        # Returns: (cov_matrix, param_names)

    def save_state(self):
        # Persist study state to SQLite
        # (Optuna handles this automatically, but explicit for clarity)

    def is_converged(self) -> bool:
        # Check convergence based on iteration count
        # (Simple check for initial implementation)
```

**State management**:
- Optuna study stored in SQLite: `{exp_dir}/optuna.db`
- Maintains pending trials list for ask/tell pattern
- Tracks iteration number internally
- Stores historical metrics in study.user_attrs for trend tracking:
  ```python
  study.set_user_attr(f"iter_{iteration}_best_metrics", best_metrics)
  study.set_user_attr(f"iter_{iteration}_pareto_size", pareto_size)
  ```

#### Component 2: Output Utilities
**File**: `src/optimization/optuna_outputs.py`

**Functions**:
```python
def format_best_candidates(study, n: int, current_iteration: int) -> Dict:
    # Extract Pareto front trials
    # Format as JSON-serializable output
    # Include: params, objectives, trial_id, improvement_over_baseline
    # Build pareto_front_evolution from study.user_attrs (iterations 0 to current)

def format_next_candidates(param_sets: List[Dict]) -> Dict:
    # Format next batch for iteration i+1
    # Include: params, acquisition_scores (if available)

def compute_parameter_covariance(study, param_names: List[str]) -> np.ndarray:
    # Empirical covariance from completed trials
    # Handle categorical params (one-hot encoding)

def compute_parameter_importance(study) -> Dict[str, float]:
    # Optional: Use Optuna's built-in importance evaluator

def build_pareto_evolution_history(study, current_iteration: int) -> Dict:
    # Extract historical metrics from study.user_attrs
    # Build timeline showing Pareto front growth over iterations
    # Return: {iteration_i: {pareto_size, best_metrics}, ...}
```

#### Component 3: Decoupled Workflow API
**Key principle**: Optimizer never triggers sample generation. Client controls execution.

**API Flow**:
```python
# === ITERATION 0: Initialize ===
optimizer = OptunaOptimizer(config)

# Client generates iteration 0 samples (real/close/far) externally
# ...

# Optimizer stores iteration 0 as baseline
optimizer.store_baseline_metrics(baseline_metrics)

# === ITERATION 1+: Optimize ===
# Step 1: Optimizer suggests next batch
next_candidates = optimizer.suggest_next_batch(batch_size=8)
# Output: List[Dict] with parameters for 8 configs

# Client saves next_candidates.json for reference
# Client decides: "Yes, I'll run iteration i+1"

# --- Client generates samples externally ---
# (days/weeks may pass)

# Step 2: Client provides results back
optimizer.update_with_results(
    parameters=param_sets,       # What was generated
    embeddings=embeddings,       # Extracted embeddings
    metrics=metrics_list         # Computed metrics
)

# Step 3: Generate outputs
best_candidates = optimizer.get_best_candidates(n=5)
# Includes trend analysis across all iterations

# Step 4: Save state
optimizer.save_state()  # Persist to SQLite
```

**State Persistence**:
- Optuna study in SQLite: `{exp_dir}/optuna.db`
- Can load and resume: `OptunaOptimizer.load(exp_dir)`
- All trial history preserved

#### Component 4: Integration with Experiment Runner (OPTIONAL)
**File**: `src/orchestration/experiment_runner.py` (MODIFY for testing)

**Note**: In real-world usage, experiment_runner.py is NOT used (client generates samples).
For testing/validation purposes, we can keep the runner but it should:
1. Use the decoupled API (suggest → generate → update)
2. Demonstrate the proper workflow for clients

**Changes needed**:
1. Replace import from `optimizer_placeholder` → `optuna_optimizer`
2. Use decoupled API: `suggest_next_batch()` → generate → `update_with_results()`
3. Add output generation after each iteration:
   - Save best candidates to `iter_{i}/best_candidates.json`
   - Save next candidates to `iter_{i}/next_candidates.json`
   - Save covariance matrix to `iter_{i}/param_covariance.npy`

### Detailed Implementation Steps

#### Step 1: Create OptunaOptimizer Core Class
- Setup multi-objective study with TPE sampler
- Implement parameter space mapping (config → Optuna format)
- Handle categorical (void_shape), integer (void_count), continuous params
- Implement ask/tell pattern for batch suggestions

#### Step 2: Implement Output Methods
- `get_best_candidates()`: Query Pareto front, rank by hypervolume contribution
- `get_parameter_covariance()`: Compute from completed trials
  - Extract parameter vectors from all completed trials
  - Handle categorical: one-hot encode void_shape
  - Compute covariance matrix (6×6 or 8×8 if one-hot)
  - Return with parameter name mapping

#### Step 3: Modify Experiment Runner
- Import new optimizer
- Modify `run_iteration()` flow:
  ```python
  # OLD: params = optimizer.suggest_next_parameters(...)
  # NEW:
  params = optimizer.suggest_next_parameters(...)
  # ... generate images, compute metrics ...
  optimizer.update_with_results(trials, metrics_list)

  # NEW: Generate outputs
  best_candidates = optimizer.get_best_candidates(n=5)
  next_candidates = optimizer.get_next_candidates(m=8)
  covariance = optimizer.get_parameter_covariance()

  # Save outputs
  save_json(best_candidates, f"{iter_dir}/best_candidates.json")
  save_json(next_candidates, f"{iter_dir}/next_candidates.json")
  np.save(f"{iter_dir}/param_covariance.npy", covariance)
  ```

#### Step 4: Configuration Updates
**File**: `configs/experiment_config.yaml`

Add optimizer section:
```yaml
optimizer:
  type: "optuna"
  sampler: "tpe"  # or "gp" for future experimentation
  n_startup_trials: 10  # Random exploration first
  multivariate: true    # Consider parameter interactions

  output:
    best_candidates_count: 5   # n best from Pareto front
    next_candidates_count: 8   # m for next iteration
    save_covariance: true
    save_parameter_importance: true
```

#### Step 5: Testing & Validation
- Unit tests for OptunaOptimizer methods
- Integration test with small experiment
- Verify outputs are generated correctly
- Check covariance matrix is symmetric and reasonable

## Technical Details

### Parameter Space Mapping (CORRECTED)
```python
# Config format → Optuna format
# CRITICAL: void_shape is 3 probability parameters, not categorical!

def create_trial_params(trial: optuna.Trial, config: Dict) -> Dict:
    # Void shape probabilities (simplex constraint: sum to 1.0)
    # Optimize 2 free parameters, compute 3rd
    circle_prob = trial.suggest_float('void_shape_circle_prob', 0.0, 1.0)
    ellipse_prob = trial.suggest_float('void_shape_ellipse_prob', 0.0, 1.0 - circle_prob)
    irregular_prob = 1.0 - circle_prob - ellipse_prob

    return {
        'void_shape_probs': {
            'circle': circle_prob,
            'ellipse': ellipse_prob,
            'irregular': irregular_prob
        },
        'void_count': trial.suggest_int(
            'void_count',
            config['bounds']['void_count']['min'],
            config['bounds']['void_count']['max']
        ),
        'base_size': trial.suggest_float(
            'base_size',
            config['bounds']['base_size']['min'],
            config['bounds']['base_size']['max']
        ),
        'rotation': trial.suggest_float(
            'rotation',
            config['bounds']['rotation']['min'],
            config['bounds']['rotation']['max']
        ),
        'center_x': trial.suggest_float(
            'center_x',
            config['bounds']['center_x']['min'],
            config['bounds']['center_x']['max']
        ),
        'center_y': trial.suggest_float(
            'center_y',
            config['bounds']['center_y']['min'],
            config['bounds']['center_y']['max']
        ),
        'position_spread': trial.suggest_float(
            'position_spread',
            config['bounds']['position_spread']['min'],
            config['bounds']['position_spread']['max']
        )
    }

# Total: 7 parameters
# - 2 for void_shape (3rd is 1-sum)
# - 1 integer (void_count)
# - 5 continuous (base_size, rotation, center_x, center_y, position_spread)
```

### Multi-Objective Optimization
```python
study = optuna.create_study(
    study_name=f"synthetic_opt_{exp_name}",
    storage=f"sqlite:///{exp_dir}/optuna.db",
    load_if_exists=True,
    directions=["minimize", "minimize", "minimize"],  # MMD, Wass, NN
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=10,
        multivariate=True,
        seed=config['random_seed']
    )
)
```

### Covariance Computation (Empirical) - CORRECTED
```python
def compute_parameter_covariance(study):
    # Get all completed trials
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]

    # Extract parameter vectors
    # CORRECTED: void_shape is 2 free parameters + 1 constrained
    param_names = [
        'void_shape_circle_prob',
        'void_shape_ellipse_prob',
        # Note: void_shape_irregular_prob = 1 - circle - ellipse (dependent, excluded)
        'void_count',
        'base_size',
        'rotation',
        'center_x',
        'center_y',
        'position_spread'
    ]

    param_matrix = []
    for trial in completed:
        param_vec = [trial.params[name] for name in param_names]
        param_matrix.append(param_vec)

    param_matrix = np.array(param_matrix)  # Shape: (n_trials, 8)
    covariance = np.cov(param_matrix, rowvar=False)  # Shape: (8, 8)

    return covariance, param_names

# Interpretation:
# - covariance[0, 1]: relationship between circle_prob and ellipse_prob
# - covariance[0, 3]: relationship between circle_prob and base_size
# - etc.
```

## Output Format Specifications

### 1. Best Candidates JSON (with Trend Tracking)
```json
{
  "iteration": 3,
  "n_best": 5,
  "pareto_front_size": 12,
  "baseline_metrics": {
    "iteration": 0,
    "close_samples": {
      "mmd_rbf": 0.110,
      "wasserstein": 0.057,
      "mean_nn_distance": 4.00
    }
  },
  "candidates": [
    {
      "rank": 1,
      "trial_id": 42,
      "discovered_at_iteration": 2,
      "parameters": {
        "void_shape_probs": {
          "circle": 0.15,
          "ellipse": 0.65,
          "irregular": 0.20
        },
        "void_count": 3,
        "base_size": 8.5,
        "rotation": 125.5,
        "center_x": 0.55,
        "center_y": 0.48,
        "position_spread": 0.45
      },
      "objectives": {
        "mmd_rbf": 0.023,
        "wasserstein": 0.015,
        "mean_nn_distance": 1.85
      },
      "improvement_over_baseline": {
        "mmd_rbf": -0.087,
        "wasserstein": -0.042,
        "mean_nn_distance": -2.15
      },
      "improvement_percentage": {
        "mmd_rbf": -79.1,
        "wasserstein": -73.7,
        "mean_nn_distance": -53.8
      },
      "dominated_by": [],
      "dominates_count": 8
    }
    // ... 4 more candidates
  ],
  "pareto_front_evolution": {
    "iteration_0": {
      "pareto_size": 0,
      "best_mmd_rbf": 0.110,
      "best_wasserstein": 0.057,
      "best_mean_nn_distance": 4.00
    },
    "iteration_1": {
      "pareto_size": 5,
      "best_mmd_rbf": 0.065,
      "best_wasserstein": 0.032,
      "best_mean_nn_distance": 2.90
    },
    "iteration_2": {
      "pareto_size": 9,
      "best_mmd_rbf": 0.034,
      "best_wasserstein": 0.021,
      "best_mean_nn_distance": 2.20
    },
    "iteration_3": {
      "pareto_size": 12,
      "best_mmd_rbf": 0.023,
      "best_wasserstein": 0.015,
      "best_mean_nn_distance": 1.85
    }
  }
}
```

### 2. Next Candidates JSON
```json
{
  "iteration": 3,
  "next_iteration": 4,
  "m_candidates": 8,
  "candidates": [
    {
      "index": 0,
      "trial_id": "pending_50",
      "parameters": {
        "void_shape_probs": {
          "circle": 0.70,
          "ellipse": 0.20,
          "irregular": 0.10
        },
        "void_count": 5,
        "base_size": 11.2,
        "rotation": 245.3,
        "center_x": 0.62,
        "center_y": 0.41,
        "position_spread": 0.62
      },
      "suggested_by": "tpe_sampler",
      "exploration_score": 0.73
    }
    // ... 7 more candidates
  ]
}
```

### 3. Covariance Matrix (NPY + metadata JSON)
```json
{
  "parameter_names": [
    "void_shape_circle_prob",
    "void_shape_ellipse_prob",
    "void_count",
    "base_size",
    "rotation",
    "center_x",
    "center_y",
    "position_spread"
  ],
  "matrix_shape": [8, 8],
  "n_trials_used": 32,
  "note": "void_shape_irregular_prob is constrained (= 1 - circle - ellipse), so not included in covariance"
}
```

## Questions & Clarifications Needed

### ✅ Resolved: Number of Best Candidates (n)
**Decision**: Top n from Pareto front, make n configurable in config.yaml
- Default: n=5
- User can adjust based on preference

### ✅ Resolved: Trend Tracking Across Iterations
**Decision**: Enable full trend visibility across all iterations
- Output will include historical comparison to ALL previous iterations
- Each iteration's output will show: iteration 0 baseline, and metrics from iterations 1, 2, ..., i-1
- This allows plotting improvement trends over time

**Implementation**:
- Save cumulative metrics history in study metadata
- Best candidates output includes: `improvement_over_baseline` and `metrics_by_iteration` timeline

### ✅ Resolved: Convergence Detection
**Decision**: Simple iteration count limit
- Stop after `max_iterations` (default: 10)
- No early stopping or smart convergence for initial implementation
- Can add sophisticated methods in future if needed

## Files to Create/Modify

### New Files
1. `src/optimization/optuna_optimizer.py` - Main optimizer class (~300 lines)
2. `src/optimization/optuna_outputs.py` - Output utilities (~200 lines)
3. `tests/test_optuna_optimizer.py` - Unit tests (~150 lines)

### Modified Files
1. `src/orchestration/experiment_runner.py` - Add output generation (~50 lines added)
2. `configs/experiment_config.yaml` - Add optimizer section (~15 lines)
3. `requirements.txt` or `pyproject.toml` - Add optuna dependency

### Files to Keep
- `src/optimization/metrics.py` - No changes needed
- `src/embedding/` - No changes needed
- `src/data_generation/` - No changes needed

## Dependencies
```toml
[tool.poetry.dependencies]
optuna = "^3.5.0"  # Latest stable version
```

## Implementation Order
1. ✅ Create plan (this document)
2. Add optuna dependency
3. Create `optuna_optimizer.py` skeleton with ask/tell pattern
4. Implement parameter space mapping
5. Implement output methods (best candidates, covariance)
6. Create `optuna_outputs.py` utilities
7. Modify experiment runner integration
8. Update config file
9. Write unit tests
10. Run integration test with small experiment
11. Validate outputs are correct

## Success Criteria
- [x] Optimizer suggests diverse parameter sets using TPE
- [x] Multi-objective optimization tracks Pareto front correctly
- [x] After each iteration, outputs are generated:
  - best_candidates.json with top n from Pareto front
  - next_candidates.json with m suggestions for next iteration
  - param_covariance.npy with 8×8 covariance matrix
- [x] Integration with experiment runner is seamless
- [x] Optuna study persists to SQLite database
- [x] Can resume experiments from saved study state

## Future Enhancements (Out of Scope for Initial Implementation)
1. Add GP sampler option for comparison
2. Implement true batch acquisition (qUCB) with Ax/BoTorch
3. Add parameter importance analysis visualization
4. Implement smart convergence detection
5. Add heteroscedastic noise modeling
6. Export Pareto front visualization
