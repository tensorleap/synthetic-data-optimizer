"""
Production experiment runner - single iteration API.

This module provides a pure Python API for running optimization iterations.
The external system generates all data on-the-fly and calls these functions directly.

Usage:
    from scripts.run_experiment import run_optimizer_iteration

    # Each iteration
    suggestions = run_optimizer_iteration(
        real_embeddings=real_embs,
        embeddings_per_simulation=[circle_embs, ellipse_embs, irregular_embs],
        metadata_per_simulation=[circle_df, ellipse_df, irregular_df]
    )
"""

import numpy as np
from typing import List, Tuple, Dict

from src.production.experiment_runner import ExperimentRunner
from src.production.data_utils import (
    prepare_client_data_for_optimizer,
    load_distributions_from_metadata,
    infer_bounds_from_metadata
)
from src.production.config import DEFAULT_CONFIG, GROUP_NAMES


def run_optimizer_iteration(
    real_embeddings: np.ndarray,
    embeddings_per_simulation: List[np.ndarray],
    metadata_per_simulation: List['pd.DataFrame']
) -> List[Tuple[str, Dict]]:
    """
    High-level function: run one optimization iteration from client data format.

    This function handles the complete workflow from client's per-simulation data format
    through to optimization suggestions. It's the simplest way to use the optimizer.

    Workflow:
    1. Convert per-simulation client data to unified optimizer format
    2. Extract distributions and infer bounds from metadata
    3. Create runner (or reuse existing one)
    4. Run optimization iteration
    5. Return suggestions for next iteration

    Args:
        real_embeddings: Real data embeddings (M, 400)
        embeddings_per_simulation: List of synthetic embedding arrays, one per simulation type
                                   Each array has shape (n_samples_for_that_type, 400)
        metadata_per_simulation: List of metadata DataFrames, one per simulation type
                                 Each has 'distribution_id' column and type-specific params

    Returns:
        suggestions: List of (dist_id, params_dict) suggestions for next iteration

    Example:
        >>> from scripts.run_experiment import run_optimizer_iteration
        >>>
        >>> # Each iteration
        >>> suggestions = run_optimizer_iteration(
        ...     real_embeddings=real_embs,
        ...     embeddings_per_simulation=[circle_embs, ellipse_embs, irregular_embs],
        ...     metadata_per_simulation=[circle_df, ellipse_df, irregular_df]
        ... )
    """
    # Hardcoded configuration
    config = DEFAULT_CONFIG
    group_names = GROUP_NAMES

    # Convert client format to optimizer format
    synthetic_embeddings, metadata_df = prepare_client_data_for_optimizer(
        embeddings_per_simulation, metadata_per_simulation, group_names
    )

    # Extract distributions and infer bounds
    distributions = load_distributions_from_metadata(metadata_df)
    param_bounds = infer_bounds_from_metadata(metadata_df, group_names)

    # Create runner with dict config
    runner = ExperimentRunner(
        config=config,
        param_bounds=param_bounds,
        group_names=group_names
    )
    runner.set_real_embeddings(real_embeddings)

    # Run iteration
    suggestions = runner.run_iteration(
        current_distributions=distributions,
        synthetic_embeddings_400d=synthetic_embeddings,
        synthetic_metadata=metadata_df.to_dict('records')
    )

    return suggestions


if __name__ == '__main__':
    """Usage example demonstrating the simplified high-level API."""
    import pandas as pd

    print("=" * 60)
    print("PRODUCTION API - HIGH-LEVEL USAGE EXAMPLE")
    print("=" * 60)

    # ================================================================
    # STEP 1: Client provides per-simulation data (their format)
    # ================================================================

    # Client provides 3 embedding arrays (one per shape)
    # Distribution 0: 12 circle, 8 ellipse, 5 irregular samples
    # Distribution 1: 8 circle, 10 ellipse, 7 irregular samples
    # Distribution 2: 10 circle, 9 ellipse, 6 irregular samples

    # Circle embeddings (30 total)
    circle_embeddings = np.random.randn(30, 400).astype(np.float32)

    # Ellipse embeddings (27 total)
    ellipse_embeddings = np.random.randn(27, 400).astype(np.float32)

    # Irregular embeddings (18 total)
    irregular_embeddings = np.random.randn(18, 400).astype(np.float32)

    # Client provides 3 metadata DataFrames (one per shape) with shape-specific params
    circle_metadata = pd.DataFrame({
        'distribution_id': [0]*12 + [1]*8 + [2]*10,
        'void_count_mean': [5.2]*12 + [6.1]*8 + [7.3]*10,
        'base_size_mean': [10.3]*12 + [11.2]*8 + [12.5]*10,
        'base_size_std': [2.1]*12 + [2.4]*8 + [2.8]*10,
        'center_x_mean': [0.51]*12 + [0.48]*8 + [0.53]*10,
        'center_x_std': [0.11]*12 + [0.13]*8 + [0.16]*10,
        'center_y_mean': [0.52]*12 + [0.49]*8 + [0.54]*10,
        'center_y_std': [0.11]*12 + [0.13]*8 + [0.16]*10,
        'position_spread_mean': [0.15]*12 + [0.18]*8 + [0.21]*10,
        'position_spread_std': [0.051]*12 + [0.062]*8 + [0.073]*10
    })

    ellipse_metadata = pd.DataFrame({
        'distribution_id': [0]*8 + [1]*10 + [2]*9,
        'void_count_mean': [4.3]*8 + [5.2]*10 + [6.4]*9,
        'base_size_mean': [12.4]*8 + [14.3]*10 + [15.6]*9,
        'base_size_std': [2.1]*8 + [2.4]*10 + [2.8]*9,
        'rotation_mean': [45.5]*8 + [90.8]*10 + [180.7]*9,
        'rotation_std': [15.2]*8 + [20.3]*10 + [30.4]*9,
        'center_x_mean': [0.51]*8 + [0.48]*10 + [0.53]*9,
        'center_x_std': [0.11]*8 + [0.13]*10 + [0.16]*9,
        'center_y_mean': [0.52]*8 + [0.49]*10 + [0.54]*9,
        'center_y_std': [0.11]*8 + [0.13]*10 + [0.16]*9,
        'position_spread_mean': [0.15]*8 + [0.18]*10 + [0.21]*9,
        'position_spread_std': [0.051]*8 + [0.062]*10 + [0.073]*9
    })

    irregular_metadata = pd.DataFrame({
        'distribution_id': [0]*5 + [1]*7 + [2]*6,
        'void_count_mean': [3.4]*5 + [4.3]*7 + [5.5]*6,
        'base_size_mean': [8.5]*5 + [9.4]*7 + [10.7]*6,
        'base_size_std': [2.1]*5 + [2.4]*7 + [2.8]*6,
        'center_x_mean': [0.51]*5 + [0.48]*7 + [0.53]*6,
        'center_x_std': [0.11]*5 + [0.13]*7 + [0.16]*6,
        'center_y_mean': [0.52]*5 + [0.49]*7 + [0.54]*6,
        'center_y_std': [0.11]*5 + [0.13]*7 + [0.16]*6,
        'position_spread_mean': [0.15]*5 + [0.18]*7 + [0.21]*6,
        'position_spread_std': [0.051]*5 + [0.062]*7 + [0.073]*6
    })

    print(f"\n[Client Data] Provided per-simulation data:")
    print(f"  Circle: {circle_embeddings.shape[0]} samples, metadata shape {circle_metadata.shape}")
    print(f"  Ellipse: {ellipse_embeddings.shape[0]} samples, metadata shape {ellipse_metadata.shape}")
    print(f"  Irregular: {irregular_embeddings.shape[0]} samples, metadata shape {irregular_metadata.shape}")

    # Create real embeddings
    real_embeddings = np.random.randn(100, 400).astype(np.float32)

    # ================================================================
    # STEP 2: Run optimization using high-level API (single function call!)
    # ================================================================

    print(f"\n[Running Optimizer] Using high-level API...")
    suggestions = run_optimizer_iteration(
        real_embeddings=real_embeddings,
        embeddings_per_simulation=[circle_embeddings, ellipse_embeddings, irregular_embeddings],
        metadata_per_simulation=[circle_metadata, ellipse_metadata, irregular_metadata]
    )

    # ================================================================
    # Done!
    # ================================================================

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print(f"\nReceived {len(suggestions)} suggestions for next iteration")
    print(f"First suggestion ID: {suggestions[0][0]}")
    print(f"  Param keys: {len(suggestions[0][1])} total")


