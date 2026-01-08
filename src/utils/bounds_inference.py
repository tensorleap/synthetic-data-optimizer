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
            'complexity': [5, 12],  # Only irregular has complexity
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


if __name__ == "__main__":
    # Example usage
    csv_path = Path(__file__).parent.parent.parent / "data" / "dummy_params.csv"

    bounds = infer_bounds_from_csv(csv_path=csv_path)

    print_bounds(bounds)
