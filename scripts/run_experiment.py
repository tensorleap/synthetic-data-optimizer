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
) -> pd.DataFrame:
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
    5. Convert suggestions to CSV format

    Args:
        real_embeddings: Real data embeddings (M, 400)
        embeddings_per_simulation: List of synthetic embedding arrays, one per simulation type
                                   Each array has shape (n_samples_for_that_type, 400)
        metadata_per_simulation: List of metadata DataFrames, one per simulation type
                                 Each has 'distribution_id' column and simulation-specific params
                                 Parameters are inferred from DataFrame column names

    Returns:
        DataFrame with columns:
        - distribution_id: suggested distribution ID
        - simulation_type: simulation name (simulation_1, simulation_2, etc.)
        - shape_probability: probability for this simulation
        - {param_name}: parameter values (without simulation prefix)

    Example:
        >>> from scripts.run_experiment import run_optimizer_iteration
        >>>
        >>> # Each iteration with any number of simulation types
        >>> suggestions_df = run_optimizer_iteration(
        ...     real_embeddings=real_embs,
        ...     embeddings_per_simulation=[sim1_embs, sim2_embs, sim3_embs],
        ...     metadata_per_simulation=[sim1_df, sim2_df, sim3_df]
        ... )
        >>> suggestions_df.to_csv('suggestions.csv', index=False)
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

    # Convert to CSV format
    suggestions_df = suggestions_to_csv_format(suggestions, group_names)

    return suggestions_df


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
    # Distribution 0: 12 sim1, 8 sim2, 5 sim3 samples
    # Distribution 1: 8 sim1, 10 sim2, 7 sim3 samples
    # Distribution 2: 10 sim1, 9 sim2, 6 sim3 samples

    # Simulation 1 embeddings (30 total)
    sim1_embeddings = np.random.randn(30, 400).astype(np.float32)

    # Simulation 2 embeddings (27 total)
    sim2_embeddings = np.random.randn(27, 400).astype(np.float32)

    # Simulation 3 embeddings (18 total)
    sim3_embeddings = np.random.randn(18, 400).astype(np.float32)

    # Client provides 3 metadata DataFrames (one per simulation) with simulation-specific params
    # Each simulation can have different parameter sets (column names)
    sim1_metadata = pd.DataFrame({
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

    sim2_metadata = pd.DataFrame({
        'distribution_id': [0]*8 + [1]*10 + [2]*9,
        'void_count_mean': [4.3]*8 + [5.2]*10 + [6.4]*9,
        'base_size_mean': [12.4]*8 + [14.3]*10 + [15.6]*9,
        'base_size_std': [2.1]*8 + [2.4]*10 + [2.8]*9,
        'rotation_mean': [45.5]*8 + [90.8]*10 + [180.7]*9,
        'rotation_std': [15.2]*8 + [20.3]*10 + [30.4]*9,
        'center_x_mean': [0.51]*8 + [0.48]*10 + [0.53]*9,
        'center_x_std': [0.11]*8 + [0.13]*10 + [0.16]*9,
        'center_y_mean': [0.52]*8 + [0.49]*10 + [0.54]*9,
        'blah': [0.11]*8 + [0.13]*10 + [0.16]*9,
        'position_spread_mean': [0.15]*8 + [0.18]*10 + [0.21]*9,
        'position_spread_std': [0.051]*8 + [0.062]*10 + [0.073]*9
    })

    sim3_metadata = pd.DataFrame({
        'distribution_id': [0]*5 + [1]*7 + [2]*6,
        'void_count_mean': [3.4]*5 + [4.3]*7 + [5.5]*6,
        'base_size_mean': [8.5]*5 + [9.4]*7 + [10.7]*6,
        'base_size_max': [2.1]*5 + [2.4]*7 + [2.8]*6,
        'center_x_mean': [0.51]*5 + [0.48]*7 + [0.53]*6,
        'center_x_std': [0.11]*5 + [0.13]*7 + [0.16]*6,
        'center_y_mean': [0.52]*5 + [0.49]*7 + [0.54]*6,
        'center_y_std': [0.11]*5 + [0.13]*7 + [0.16]*6,
        'position_spread_mean': [0.15]*5 + [0.18]*7 + [0.21]*6,
        'position_spread_std': [0.051]*5 + [0.062]*7 + [0.073]*6
    })

    print(f"\n[Client Data] Provided per-simulation data:")
    print(f"  Simulation 1: {sim1_embeddings.shape[0]} samples, {len(sim1_metadata.columns)-1} params")
    print(f"  Simulation 2: {sim2_embeddings.shape[0]} samples, {len(sim2_metadata.columns)-1} params")
    print(f"  Simulation 3: {sim3_embeddings.shape[0]} samples, {len(sim3_metadata.columns)-1} params")

    # Create real embeddings
    real_embeddings = np.random.randn(100, 400).astype(np.float32)

    # ================================================================
    # STEP 2: Run optimization using high-level API (single function call!)
    # ================================================================

    print(f"\n[Running Optimizer] Using high-level API...")
    print(f"  -> Auto-generating simulation names: simulation_1, simulation_2, simulation_3")
    suggestions_df = run_optimizer_iteration(
        real_embeddings=real_embeddings,
        embeddings_per_simulation=[sim1_embeddings, sim2_embeddings, sim3_embeddings],
        metadata_per_simulation=[sim1_metadata, sim2_metadata, sim3_metadata]
    )

    # ================================================================
    # Done!
    # ================================================================

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print(f"\nReceived suggestions as DataFrame with {len(suggestions_df)} rows")
    print(f"  ({len(suggestions_df) // 3} distributions × 3 simulations)")
    print(f"\nDataFrame shape: {suggestions_df.shape}")
    print(f"Columns: {list(suggestions_df.columns)}")
    print(f"\nFirst 6 rows (2 distributions):")
    print(suggestions_df.head(6).to_string(index=False))
    print(f"\n  -> Can be saved with: suggestions_df.to_csv('suggestions.csv', index=False)")


