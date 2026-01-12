"""
Utility to infer parameter bounds from data.

Infers conditional bounds from multiple DataFrames (one per group):
  {group_name: {param: [min, max] or [categories]}}

Each group can have DIFFERENT parameters - this is the key use case for conditional
optimization (e.g., ellipse has 'rotation', circle doesn't).

For a single group (non-conditional case), pass a list of length 1.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml


def infer_bounds_from_dataframe(
    df: pd.DataFrame,
    param_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None
) -> Dict:
    """
    Infer parameter bounds from a single DataFrame.

    Args:
        df: DataFrame with sample data (each row is a sample)
        param_columns: List of parameter column names to process. If None, uses all columns.
        exclude_columns: List of columns to exclude (e.g., ['void_shape'])

    Returns:
        Dictionary mapping parameter names to their inferred bounds:
        - Numerical columns: [min, max]
        - Categorical columns: list of unique values
    """
    if param_columns is None:
        param_columns = df.columns.tolist()

    if exclude_columns:
        param_columns = [p for p in param_columns if p not in exclude_columns]

    bounds = {}

    for param in param_columns:
        if param not in df.columns:
            print(f"Warning: Column '{param}' not found in DataFrame")
            continue

        col_data = df[param]

        # Check if column is numerical or categorical
        if pd.api.types.is_numeric_dtype(col_data):
            # Numerical column: [min, max]
            bounds[param] = [float(col_data.min()), float(col_data.max())]
        else:
            # Categorical column: list of unique values
            bounds[param] = col_data.unique().tolist()

    return bounds


def infer_conditional_bounds(
    dataframes: List[pd.DataFrame],
    group_names: List[str],
    exclude_columns: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Infer conditional parameter bounds from multiple DataFrames.

    Each DataFrame represents a conditional group (e.g., a shape type) and may have
    DIFFERENT columns/parameters. This is the key use case for conditional optimization.

    Example output:
    {
        'circle': {
            'void_count': [1, 15],
            'base_size': [3.0, 20.0],
            'center_x': [0.1, 0.9],
            ...
        },
        'ellipse': {
            'void_count': [1, 12],
            'base_size': [4.0, 25.0],
            'rotation': [0.0, 360.0],  # Only ellipse has rotation
            'center_x': [0.15, 0.85],
            ...
        },
        'irregular': {
            'void_count': [1, 8],
            'base_size': [4.0, 12.0],
            ...
        }
    }

    Args:
        dataframes: List of DataFrames, one per conditional group
        group_names: Names for each group (e.g., ['circle', 'ellipse', 'irregular'])
        exclude_columns: Columns to exclude from bounds inference (e.g., ['void_shape'])

    Returns:
        Nested dict: {group_name: {param: bounds}}
    """
    if len(dataframes) != len(group_names):
        raise ValueError(
            f"Number of dataframes ({len(dataframes)}) must match "
            f"number of group names ({len(group_names)})"
        )

    if exclude_columns is None:
        exclude_columns = ['void_shape']  # Default: exclude the group identifier column

    conditional_bounds = {}

    for df, group_name in zip(dataframes, group_names):
        bounds = infer_bounds_from_dataframe(
            df,
            exclude_columns=exclude_columns
        )
        conditional_bounds[group_name] = bounds

    return conditional_bounds


def infer_bounds_from_csv(
    csv_path: Union[str, Path],
    param_columns: Optional[List[str]] = None
) -> Dict:
    """
    Infer parameter bounds from a CSV file (legacy compatibility).

    For conditional bounds, use infer_conditional_bounds() instead.

    Args:
        csv_path: Path to CSV file with sample data
        param_columns: List of parameter column names to process

    Returns:
        Dictionary mapping parameter names to their inferred bounds
    """
    df = pd.read_csv(csv_path)
    return infer_bounds_from_dataframe(df, param_columns)


def save_bounds_to_yaml(
    bounds: Dict,
    output_path: Union[str, Path]
):
    """
    Save inferred bounds to a YAML file.

    Args:
        bounds: The parameter bounds dictionary
        output_path: Path to save the YAML file
    """
    output_path = Path(output_path)

    with open(output_path, 'w') as f:
        yaml.safe_dump(bounds, f, default_flow_style=False, sort_keys=False)

    print(f"Saved bounds to {output_path}")


def print_bounds(bounds: Dict):
    """Pretty print the inferred bounds for inspection."""
    print("\nInferred parameter bounds:\n")
    print(yaml.dump(bounds, default_flow_style=False, sort_keys=False))


def get_param_bounds(data_dir: Optional[Union[str, Path]] = None) -> tuple[Dict[str, Dict], List[str]]:
    """
    Get parameter bounds and group names for the optimizer.

    Reads bounds from per-group CSV files in the data directory.
    Falls back to MockDataGenerator's DEFAULT_GROUP_SPECS if no data files found.

    Args:
        data_dir: Directory containing bounds data. Expected structure:
                 data_dir/bounds/{group_name}_params.csv
                 If None, looks for data/test_experiment/bounds/

    Returns:
        param_bounds: Dict mapping group names to their parameter bounds
        group_names: List of group names
    """
    from ..data_generation.mock_data_generator import MockDataGenerator

    group_names = list(MockDataGenerator.DEFAULT_GROUP_SPECS.keys())

    # Determine bounds directory
    if data_dir is None:
        bounds_dir = Path("data/test_experiment/bounds")
    else:
        bounds_dir = Path(data_dir) / "bounds"

    # Try to read from data files
    if bounds_dir.exists():
        dataframes = []
        all_found = True

        for group_name in group_names:
            csv_path = bounds_dir / f"{group_name}_params.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                dataframes.append(df)
            else:
                all_found = False
                break

        if all_found:
            param_bounds = infer_conditional_bounds(dataframes, group_names)
            return param_bounds, group_names

    # Fallback: use DEFAULT_GROUP_SPECS directly
    print("Warning: Bounds data files not found, using default specs")
    param_bounds = {}
    for group_name, params_spec in MockDataGenerator.DEFAULT_GROUP_SPECS.items():
        param_bounds[group_name] = {}
        for param_base, spec in params_spec.items():
            param_bounds[group_name][f'{param_base}_mean'] = spec['mean_bounds']
            param_bounds[group_name][f'{param_base}_std'] = spec['std_bounds']

    return param_bounds, group_names


def infer_bounds_from_directories(
    directories: List[Union[str, Path]],
    group_names: List[str]
) -> Dict[str, Dict]:
    """
    Infer parameter bounds from ALL data across ALL directories.

    Combines all CSVs for each shape across all directories to get global bounds.

    Args:
        directories: List of directory paths containing per-shape CSVs
        group_names: Shape names (e.g., ['circle', 'ellipse', 'irregular'])

    Returns:
        param_bounds: {group_name: {param_name: [min, max]}}
    """
    # Collect all dataframes per group across all directories
    all_dfs = {group: [] for group in group_names}

    for directory in directories:
        directory = Path(directory)
        for group_name in group_names:
            csv_path = directory / f"{group_name}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if not df.empty:
                    all_dfs[group_name].append(df)

    # Infer bounds from combined data for each group
    param_bounds = {}
    for group_name in group_names:
        if all_dfs[group_name]:
            combined_df = pd.concat(all_dfs[group_name], ignore_index=True)
            param_bounds[group_name] = infer_bounds_from_dataframe(combined_df)
        else:
            param_bounds[group_name] = {}

    return param_bounds


def load_distribution_from_directory(
    directory: Union[str, Path],
    group_names: List[str],
    param_bounds: Dict[str, Dict]
) -> tuple[str, Dict]:
    """
    Load a single distribution from a directory containing per-shape CSVs.

    Each directory = ONE trial. Sample counts determine shape probabilities.
    Distribution params are inferred from each shape's CSV mean values.

    Args:
        directory: Path to directory with per-shape CSVs
        group_names: Shape names
        param_bounds: Pre-computed bounds (from infer_bounds_from_directories)

    Returns:
        (dist_id, params_dict) tuple in joint optimizer format
    """
    from ..optimization.optuna_optimizer import OptunaOptimizer

    directory = Path(directory)
    dist_id = directory.name

    # Load CSVs and count samples
    sample_counts = {}
    dataframes = {}

    for group_name in group_names:
        csv_path = directory / f"{group_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            sample_counts[group_name] = len(df)
            dataframes[group_name] = df
        else:
            sample_counts[group_name] = 0
            dataframes[group_name] = pd.DataFrame()

    # Convert sample counts to logits
    logits = OptunaOptimizer.sample_counts_to_logits(sample_counts)

    # Build params dict: logits + all shape params
    params = dict(logits)

    for group_name in group_names:
        df = dataframes[group_name]
        group_bounds = param_bounds.get(group_name, {})

        for param_name in group_bounds.keys():
            optuna_key = f'{group_name}__{param_name}'

            if not df.empty and param_name in df.columns:
                # Use mean of samples as distribution parameter
                params[optuna_key] = float(df[param_name].mean())
            else:
                # Fallback to midpoint of bounds
                bounds = group_bounds[param_name]
                if isinstance(bounds, list) and len(bounds) == 2:
                    params[optuna_key] = (bounds[0] + bounds[1]) / 2
                else:
                    params[optuna_key] = bounds[0] if bounds else 0.0

    return (dist_id, params)


def load_distributions_from_directories(
    directories: List[Union[str, Path]],
    group_names: List[str]
) -> tuple[List[tuple[str, Dict]], Dict[str, Dict]]:
    """
    Load N distributions from N directories for initial optimizer input.

    Each directory = one trial. All N directories = N trials for one iteration.

    Steps:
    1. Infer bounds from ALL data across ALL directories (global bounds)
    2. Load each directory as a distribution using those bounds

    Args:
        directories: List of N directory paths
        group_names: Shape names (e.g., ['circle', 'ellipse', 'irregular'])

    Returns:
        distributions: List of (dist_id, params_dict) tuples
        param_bounds: Global bounds inferred from all data
    """
    # Step 1: Infer bounds from ALL data
    param_bounds = infer_bounds_from_directories(directories, group_names)

    # Step 2: Load each directory as a distribution
    distributions = []
    for directory in directories:
        dist = load_distribution_from_directory(directory, group_names, param_bounds)
        distributions.append(dist)

    return distributions, param_bounds


def infer_bounds_from_metadata(
    metadata_df: pd.DataFrame,
    group_names: List[str]
) -> Dict[str, Dict]:
    """
    Infer parameter bounds from metadata DataFrame.

    The metadata DataFrame has columns:
    - distribution_id: int (groups samples by distribution)
    - shape_logit_*: float (one per shape)
    - {shape}__{param}_mean: float
    - {shape}__{param}_std: float

    All samples from the same distribution have identical param values.
    Bounds are inferred from the range across all distributions.

    Args:
        metadata_df: DataFrame with distribution_id and all joint params as columns
        group_names: List of shape names (e.g., ['circle', 'ellipse', 'irregular'])

    Returns:
        param_bounds: {group_name: {param_name: [min, max]}}
    """
    # Get unique distributions (one row per distribution)
    unique_dists = metadata_df.drop_duplicates(subset='distribution_id')

    # Build bounds by group
    param_bounds = {}

    for group_name in group_names:
        group_bounds = {}

        # Find all columns for this group
        for col in unique_dists.columns:
            if col.startswith(f'{group_name}__'):
                # Extract param name
                param_name = col[len(f'{group_name}__'):]

                # Infer bounds from column values
                values = unique_dists[col]
                group_bounds[param_name] = [float(values.min()), float(values.max())]

        param_bounds[group_name] = group_bounds

    return param_bounds


def load_distributions_from_metadata(
    metadata_df: pd.DataFrame
) -> List[tuple[str, Dict]]:
    """
    Extract distributions from metadata DataFrame.

    The metadata DataFrame has columns:
    - distribution_id: int (groups samples by distribution)
    - shape_logit_*: float (one per shape)
    - {shape}__{param}_mean: float
    - {shape}__{param}_std: float

    All samples from the same distribution have identical param values.

    Args:
        metadata_df: DataFrame with distribution_id and all joint params as columns

    Returns:
        List of (dist_id, params_dict) tuples in joint optimizer format
    """
    # Get unique distributions (one row per distribution)
    unique_dists = metadata_df.drop_duplicates(subset='distribution_id').sort_values('distribution_id')

    distributions = []

    for _, row in unique_dists.iterrows():
        dist_id = f"dist_{int(row['distribution_id'])}"

        # Extract all param columns (everything except distribution_id)
        params = {}
        for col in row.index:
            if col != 'distribution_id':
                params[col] = float(row[col])

        distributions.append((dist_id, params))

    return distributions


if __name__ == "__main__":
    # Example usage
    csv_path = Path(__file__).parent.parent.parent / "data" / "dummy_params.csv"

    bounds = infer_bounds_from_csv(csv_path=csv_path)

    print_bounds(bounds)
