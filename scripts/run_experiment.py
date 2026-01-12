"""
Production experiment runner - single iteration API.

This module provides a pure Python API for running optimization iterations.
The external system generates all data on-the-fly and calls these functions directly.

Usage:
    from scripts.run_experiment import create_runner, run_iteration

    # One-time setup
    runner = create_runner(config_path, param_bounds, group_names)
    runner.set_real_embeddings(real_embeddings_400d)

    # Each iteration
    suggestions = run_iteration(
        runner=runner,
        current_distributions=distributions,  # [(dist_id, params_dict), ...]
        synthetic_embeddings_400d=embeddings,  # np.ndarray
        synthetic_metadata=metadata            # List[Dict]
    )
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

from src.orchestration.experiment_runner import ExperimentRunner


def create_runner(
    config_path: Path,
    param_bounds: Dict[str, Dict],
    group_names: List[str]
) -> ExperimentRunner:
    """
    Create and initialize an ExperimentRunner.

    This is called once at the start of the optimization process.

    Args:
        config_path: Path to experiment config YAML
        param_bounds: Parameter bounds dict {group_name: {param: [min, max]}}
        group_names: List of group names (e.g., ['circle', 'ellipse', 'irregular'])

    Returns:
        Initialized ExperimentRunner instance
    """
    runner = ExperimentRunner(
        config_path=config_path,
        param_bounds=param_bounds,
        group_names=group_names
    )
    return runner


def run_iteration(
    runner: ExperimentRunner,
    current_distributions: List[Tuple[str, Dict]],
    synthetic_embeddings_400d: np.ndarray,
    synthetic_metadata: List[Dict]
) -> List[Tuple[str, Dict]]:
    """
    Run a single optimization iteration with in-memory data.

    Args:
        runner: ExperimentRunner instance (from create_runner)
        current_distributions: What was evaluated [(dist_id, params_dict), ...]
                               params_dict contains shape_logit_* and {shape}__{param} keys
        synthetic_embeddings_400d: Embeddings array (N, 400)
        synthetic_metadata: Metadata list with distribution_id for grouping

    Returns:
        next_suggestions: What to try next [(dist_id, params_dict), ...]
    """
    if runner.real_embeddings_400d is None:
        raise ValueError(
            "Real embeddings not set. Call runner.set_real_embeddings() first."
        )

    # Run iteration
    next_suggestions = runner.run_iteration(
        current_distributions=current_distributions,
        synthetic_embeddings_400d=synthetic_embeddings_400d,
        synthetic_metadata=synthetic_metadata
    )

    return next_suggestions


if __name__ == '__main__':
    """Usage example with mock data."""
    import pandas as pd
    from src.utils.bounds_inference import infer_bounds_from_metadata, load_distributions_from_metadata

    print("=" * 60)
    print("PRODUCTION API USAGE EXAMPLE")
    print("=" * 60)

    # Setup
    config_path = Path('configs/experiment_config.yaml')
    group_names = ['circle', 'ellipse', 'irregular']

    # Mock iteration 0: Create 3 distributions with 10 samples each
    n_distributions = 3
    samples_per_dist = 10

    # Create mock metadata DataFrame (this is what external system provides)
    # Use non-integer values to hint that these should be float distributions
    metadata_rows = []

    # Distribution 0
    for _ in range(samples_per_dist):
        metadata_rows.append({
            'distribution_id': 0,
            'shape_logit_circle': 0.5, 'shape_logit_ellipse': 0.0, 'shape_logit_irregular': -0.5,
            'circle__void_count_mean': 5.2, 'circle__void_count_std': 1.1,
            'circle__base_size_mean': 10.3, 'circle__base_size_std': 2.1,
            'circle__center_x_mean': 0.51, 'circle__center_x_std': 0.11,
            'circle__center_y_mean': 0.52, 'circle__center_y_std': 0.11,
            'circle__position_spread_mean': 0.15, 'circle__position_spread_std': 0.051,
            'ellipse__void_count_mean': 4.3, 'ellipse__void_count_std': 1.1,
            'ellipse__base_size_mean': 12.4, 'ellipse__base_size_std': 2.1,
            'ellipse__rotation_mean': 45.5, 'ellipse__rotation_std': 15.2,
            'ellipse__center_x_mean': 0.51, 'ellipse__center_x_std': 0.11,
            'ellipse__center_y_mean': 0.52, 'ellipse__center_y_std': 0.11,
            'ellipse__position_spread_mean': 0.15, 'ellipse__position_spread_std': 0.051,
            'irregular__void_count_mean': 3.4, 'irregular__void_count_std': 1.1,
            'irregular__base_size_mean': 8.5, 'irregular__base_size_std': 2.1,
            'irregular__center_x_mean': 0.51, 'irregular__center_x_std': 0.11,
            'irregular__center_y_mean': 0.52, 'irregular__center_y_std': 0.11,
            'irregular__position_spread_mean': 0.15, 'irregular__position_spread_std': 0.051
        })

    # Distribution 1
    for _ in range(samples_per_dist):
        metadata_rows.append({
            'distribution_id': 1,
            'shape_logit_circle': 0.0, 'shape_logit_ellipse': 0.5, 'shape_logit_irregular': 0.0,
            'circle__void_count_mean': 6.1, 'circle__void_count_std': 1.3,
            'circle__base_size_mean': 11.2, 'circle__base_size_std': 2.4,
            'circle__center_x_mean': 0.48, 'circle__center_x_std': 0.13,
            'circle__center_y_mean': 0.49, 'circle__center_y_std': 0.13,
            'circle__position_spread_mean': 0.18, 'circle__position_spread_std': 0.062,
            'ellipse__void_count_mean': 5.2, 'ellipse__void_count_std': 1.3,
            'ellipse__base_size_mean': 14.3, 'ellipse__base_size_std': 2.4,
            'ellipse__rotation_mean': 90.8, 'ellipse__rotation_std': 20.3,
            'ellipse__center_x_mean': 0.48, 'ellipse__center_x_std': 0.13,
            'ellipse__center_y_mean': 0.49, 'ellipse__center_y_std': 0.13,
            'ellipse__position_spread_mean': 0.18, 'ellipse__position_spread_std': 0.062,
            'irregular__void_count_mean': 4.3, 'irregular__void_count_std': 1.3,
            'irregular__base_size_mean': 9.4, 'irregular__base_size_std': 2.4,
            'irregular__center_x_mean': 0.48, 'irregular__center_x_std': 0.13,
            'irregular__center_y_mean': 0.49, 'irregular__center_y_std': 0.13,
            'irregular__position_spread_mean': 0.18, 'irregular__position_spread_std': 0.062
        })

    # Distribution 2
    for _ in range(samples_per_dist):
        metadata_rows.append({
            'distribution_id': 2,
            'shape_logit_circle': -0.5, 'shape_logit_ellipse': -0.5, 'shape_logit_irregular': 0.5,
            'circle__void_count_mean': 7.3, 'circle__void_count_std': 1.6,
            'circle__base_size_mean': 12.5, 'circle__base_size_std': 2.8,
            'circle__center_x_mean': 0.53, 'circle__center_x_std': 0.16,
            'circle__center_y_mean': 0.54, 'circle__center_y_std': 0.16,
            'circle__position_spread_mean': 0.21, 'circle__position_spread_std': 0.073,
            'ellipse__void_count_mean': 6.4, 'ellipse__void_count_std': 1.6,
            'ellipse__base_size_mean': 15.6, 'ellipse__base_size_std': 2.8,
            'ellipse__rotation_mean': 180.7, 'ellipse__rotation_std': 30.4,
            'ellipse__center_x_mean': 0.53, 'ellipse__center_x_std': 0.16,
            'ellipse__center_y_mean': 0.54, 'ellipse__center_y_std': 0.16,
            'ellipse__position_spread_mean': 0.21, 'ellipse__position_spread_std': 0.073,
            'irregular__void_count_mean': 5.5, 'irregular__void_count_std': 1.6,
            'irregular__base_size_mean': 10.7, 'irregular__base_size_std': 2.8,
            'irregular__center_x_mean': 0.53, 'irregular__center_x_std': 0.16,
            'irregular__center_y_mean': 0.54, 'irregular__center_y_std': 0.16,
            'irregular__position_spread_mean': 0.21, 'irregular__position_spread_std': 0.073
        })

    metadata_df = pd.DataFrame(metadata_rows)

    # Create mock embeddings
    total_samples = n_distributions * samples_per_dist
    real_embeddings = np.random.randn(100, 400).astype(np.float32)
    synthetic_embeddings = np.random.randn(total_samples, 400).astype(np.float32)

    print(f"\n[Setup] Created mock data:")
    print(f"  Real embeddings: {real_embeddings.shape}")
    print(f"  Synthetic embeddings: {synthetic_embeddings.shape}")
    print(f"  Metadata: {metadata_df.shape}")

    # Extract distributions from metadata
    print(f"\n[1/4] Extracting distributions from metadata...")
    distributions = load_distributions_from_metadata(metadata_df)
    print(f"  Extracted {len(distributions)} distributions")

    # Infer bounds from metadata
    print(f"[2/4] Inferring bounds from metadata...")
    param_bounds = infer_bounds_from_metadata(metadata_df, group_names)
    print(f"  Inferred bounds for {len(param_bounds)} groups")

    # Create runner
    print(f"[3/4] Creating runner...")
    runner = create_runner(config_path, param_bounds, group_names)
    runner.set_real_embeddings(real_embeddings)

    # Run iteration
    print(f"[4/4] Running iteration...")
    suggestions = run_iteration(
        runner=runner,
        current_distributions=distributions,
        synthetic_embeddings_400d=synthetic_embeddings,
        synthetic_metadata=metadata_rows
    )

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print(f"\nReceived {len(suggestions)} suggestions for next iteration")
    print(f"First suggestion ID: {suggestions[0][0]}")
    print(f"  Param keys: {len(suggestions[0][1])} total")
