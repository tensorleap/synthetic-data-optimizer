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
import pandas as pd
from typing import List, Tuple, Dict

from src.production.experiment_runner import ExperimentRunner
from src.production.data_utils import (
    prepare_client_data_for_optimizer,
    load_distributions_from_metadata,
    infer_bounds_from_metadata
)
from src.production.config import DEFAULT_CONFIG


def suggestions_to_csv_format(
    suggestions: List[Tuple[str, Dict]],
    group_names: List[str]
) -> pd.DataFrame:
    """
    Convert optimizer suggestions from dictionary format to CSV format.

    Creates a single DataFrame where each row represents one simulation type
    from one distribution. All suggestions are combined into one table.

    Args:
        suggestions: List of (dist_id, params_dict) tuples from optimizer
                    params_dict contains shape_prob_* and {simulation}__{param} keys
        group_names: List of simulation names (e.g., ['simulation_1', 'simulation_2'])

    Returns:
        Single DataFrame with columns:
        - distribution_id: str (dist_3, dist_4, etc.)
        - simulation_type: str (simulation_1, simulation_2, etc.)
        - shape_probability: float
        - {param_name}: float for each parameter (without simulation prefix)

    Example:
        >>> suggestions = [('dist_0', {'shape_prob_simulation_1': 0.3, ...})]
        >>> df = suggestions_to_csv_format(suggestions, ['simulation_1', 'simulation_2'])
        >>> df.to_csv('suggestions.csv', index=False)
    """
    all_rows = []

    for dist_id, params in suggestions:
        for sim_name in group_names:
            # Extract probability for this simulation
            prob_key = f'shape_prob_{sim_name}'
            prob = params.get(prob_key, 0.0)

            # Extract all parameters for this simulation
            param_prefix = f'{sim_name}__'
            row = {
                'distribution_id': dist_id,
                'simulation_type': sim_name,
                'shape_probability': prob
            }

            # Add all simulation-specific parameters without prefix
            for key, value in params.items():
                if key.startswith(param_prefix):
                    param_name = key[len(param_prefix):]
                    row[param_name] = value

            all_rows.append(row)

    # Create single DataFrame with all rows
    return pd.DataFrame(all_rows)


def run_optimizer_iteration(
    real_embeddings: np.ndarray,
    embeddings_per_simulation: List[np.ndarray],
    metadata_per_simulation: List['pd.DataFrame']
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    High-level function: run one optimization iteration from client data format.

    This function handles the complete workflow from client's per-simulation data format
    through to optimization suggestions in CSV-ready DataFrame format.

    Workflow:
    1. Convert per-simulation client data to unified optimizer format
       (auto-generates simulation names: simulation_1, simulation_2, etc.)
    2. Extract distributions and infer bounds from metadata
    3. Create runner (or reuse existing one)
    4. Run optimization iteration
    5. Get best trials seen so far
    6. Convert both to CSV format

    Args:
        real_embeddings: Real data embeddings (M, 400)
        embeddings_per_simulation: List of synthetic embedding arrays, one per simulation type
                                   Each array has shape (n_samples_for_that_type, 400)
                                   Together they represent ONE joint distribution
        metadata_per_simulation: List of metadata DataFrames, one per simulation type
                                 Each DataFrame contains parameters for that simulation type
                                 All rows in a DataFrame should have identical parameter values
                                 Parameters are inferred from DataFrame column names

    Returns:
        Tuple of (suggestions_df, best_trials_df):
        - suggestions_df: Next iteration recommendations
        - best_trials_df: Best trials seen so far

        Both DataFrames have columns:
        - distribution_id: distribution/trial ID
        - simulation_type: simulation name (simulation_1, simulation_2, etc.)
        - shape_probability: probability for this simulation
        - {param_name}: parameter values (without simulation prefix)

    Example:
        >>> from scripts.run_experiment import run_optimizer_iteration
        >>>
        >>> # Each iteration with any number of simulation types
        >>> suggestions_df, best_trials_df = run_optimizer_iteration(
        ...     real_embeddings=real_embs,
        ...     embeddings_per_simulation=[sim1_embs, sim2_embs, sim3_embs],
        ...     metadata_per_simulation=[sim1_df, sim2_df, sim3_df]
        ... )
        >>> suggestions_df.to_csv('next_suggestions.csv', index=False)
        >>> best_trials_df.to_csv('best_trials.csv', index=False)
    """
    # Hardcoded configuration
    config = DEFAULT_CONFIG

    # Convert client format to optimizer format (auto-generates group_names)
    synthetic_embeddings, metadata_df, group_names = prepare_client_data_for_optimizer(
        embeddings_per_simulation, metadata_per_simulation
    )

    # Extract distributions and infer bounds
    distributions = load_distributions_from_metadata(metadata_df)
    param_bounds = infer_bounds_from_metadata(metadata_df, group_names)

    # Create runner with dict config
    runner = ExperimentRunner(
        config=config,
        param_bounds=param_bounds
    )
    runner.set_real_embeddings(real_embeddings)

    # Run iteration
    suggestions = runner.run_iteration(
        current_distributions=distributions,
        synthetic_embeddings_400d=synthetic_embeddings,
        synthetic_metadata=metadata_df.to_dict('records')
    )

    # Get best trials (same number as suggestions)
    n_suggestions = len(suggestions)
    best_trials = runner.get_best_trials(top_n=n_suggestions)

    # Convert both to CSV format
    suggestions_df = suggestions_to_csv_format(suggestions, group_names)
    best_trials_df = suggestions_to_csv_format(best_trials, group_names)

    return suggestions_df, best_trials_df


if __name__ == '__main__':
    """Usage example demonstrating the simplified high-level API with dynamic simulations."""
    import pandas as pd

    print("=" * 60)
    print("PRODUCTION API - HIGH-LEVEL USAGE EXAMPLE")
    print("=" * 60)

    # ================================================================
    # STEP 1: Client provides per-simulation data (their format)
    # ================================================================

    # Client provides 3 embedding arrays (one per simulation type)
    # TWO distributions represented across 3 simulation types
    # Each DataFrame has distribution_idx column to mark which distribution each sample belongs to

    # Simulation 1: 15 samples from dist_0 + 15 samples from dist_1 = 30 total
    sim1_embeddings = np.random.randn(30, 400).astype(np.float32)

    # Simulation 2: 10 samples from dist_0 + 10 samples from dist_1 = 20 total
    sim2_embeddings = np.random.randn(20, 400).astype(np.float32)

    # Simulation 3: 12 samples from dist_0 + 13 samples from dist_1 = 25 total
    sim3_embeddings = np.random.randn(25, 400).astype(np.float32)

    # Client provides 3 metadata DataFrames (one per simulation) with simulation-specific params
    # Each simulation can have different parameter sets (column names)
    # Rows with same distribution_idx have identical parameter values
    sim1_metadata = pd.DataFrame({
        'distribution_idx': [0] * 15 + [1] * 15,
        'void_count_mean': [5.2] * 15 + [6.5] * 15,
        'base_size_mean': [10.3] * 15 + [11.8] * 15,
        'base_size_std': [2.1] * 15 + [2.5] * 15,
        'center_x_mean': [0.51] * 15 + [0.48] * 15,
        'center_x_std': [0.11] * 15 + [0.13] * 15,
        'center_y_mean': [0.52] * 15 + [0.49] * 15,
        'center_y_std': [0.11] * 15 + [0.12] * 15,
        'position_spread_mean': [0.15] * 15 + [0.18] * 15,
        'position_spread_std': [0.051] * 15 + [0.06] * 15,
        'void_shape': ['circle'] * 30
    })

    sim2_metadata = pd.DataFrame({
        'distribution_idx': [0] * 10 + [1] * 10,
        'void_count_mean': [4.3] * 10 + [5.1] * 10,
        'base_size_mean': [12.4] * 10 + [13.2] * 10,
        'base_size_std': [2.1] * 10 + [2.3] * 10,
        'rotation_mean': [45.5] * 10 + [60.0] * 10,
        'rotation_std': [15.2] * 10 + [18.5] * 10,
        'center_x_mean': [0.51] * 10 + [0.47] * 10,
        'center_x_std': [0.11] * 10 + [0.14] * 10,
        'center_y_mean': [0.52] * 10 + [0.50] * 10,
        'center_y_std': [0.11] * 10 + [0.13] * 10,
        'position_spread_mean': [0.15] * 10 + [0.17] * 10,
        'position_spread_std': [0.051] * 10 + [0.055] * 10,
        'void_shape': ['ellipse'] * 20
    })

    sim3_metadata = pd.DataFrame({
        'distribution_idx': [0] * 12 + [1] * 13,
        'void_count_mean': [3.4] * 12 + [4.2] * 13,
        'base_size_mean': [8.5] * 12 + [9.3] * 13,
        'base_size_std': [2.1] * 12 + [2.4] * 13,
        'center_x_mean': [0.51] * 12 + [0.46] * 13,
        'center_x_std': [0.11] * 12 + [0.15] * 13,
        'center_y_mean': [0.52] * 12 + [0.48] * 13,
        'center_y_std': [0.11] * 12 + [0.14] * 13,
        'position_spread_mean': [0.15] * 12 + [0.19] * 13,
        'position_spread_std': [0.051] * 12 + [0.062] * 13,
        'void_shape': ['irregular'] * 25
    })

    print(f"\n[Client Data] Provided ONE distribution with 3 simulation types:")
    print(f"  Simulation 1: {sim1_embeddings.shape[0]} samples, {len(sim1_metadata.columns)} params")
    print(f"  Simulation 2: {sim2_embeddings.shape[0]} samples, {len(sim2_metadata.columns)} params")
    print(f"  Simulation 3: {sim3_embeddings.shape[0]} samples, {len(sim3_metadata.columns)} params")
    print(f"  Total samples: {len(sim1_embeddings) + len(sim2_embeddings) + len(sim3_embeddings)}")

    # Create real embeddings
    real_embeddings = np.random.randn(100, 400).astype(np.float32)

    # ================================================================
    # STEP 2: Run optimization using high-level API (single function call!)
    # ================================================================

    print(f"\n[Running Optimizer] Using high-level API...")
    print(f"  -> Auto-generating simulation names: simulation_1, simulation_2, simulation_3")
    suggestions_df, best_trials_df = run_optimizer_iteration(
        real_embeddings=real_embeddings,
        embeddings_per_simulation=[sim1_embeddings, sim2_embeddings, sim3_embeddings],
        metadata_per_simulation=[sim1_metadata, sim2_metadata, sim3_metadata]
    )

    # ================================================================
    # Done!
    # ================================================================

    print("\n" + "=" * 60)
    # print("EXAMPLE COMPLETE")
    # print("=" * 60)

    # print(f"\n[NEXT ITERATION SUGGESTIONS]")
    # print(f"Received {len(suggestions_df)} rows ({len(suggestions_df) // 3} distributions × 3 simulations)")
    # print(f"Shape: {suggestions_df.shape}, Columns: {list(suggestions_df.columns)}")
    # print(f"\nFirst 6 rows (2 distributions):")
    # print(suggestions_df.head(6).to_string(index=False))

    # print(f"\n[BEST TRIALS SEEN SO FAR]")
    # print(f"Top {len(best_trials_df) // 3} best trials ({len(best_trials_df)} rows)")
    # print(f"Shape: {best_trials_df.shape}")
    # print(f"\nFirst 6 rows (2 best trials):")
    # print(best_trials_df.head(6).to_string(index=False))

    # print(f"\n  -> Save with:")
    # print(f"     suggestions_df.to_csv('next_suggestions.csv', index=False)")
    # print(f"     best_trials_df.to_csv('best_trials.csv', index=False)")


