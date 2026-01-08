"""
Utility to infer parameter bounds from real data CSV.

Reads a CSV with sample data and infers bounds for each column based on data type.
- Numerical columns: [min, max] bounds
- String/categorical columns: list of unique values
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml


def infer_bounds_from_csv(
    csv_path: Union[str, Path],
    param_columns: Optional[List[str]] = None
) -> Dict:
    """
    Infer parameter bounds from a CSV containing sample data.

    Args:
        csv_path: Path to CSV file with sample data (each row is a sample)
        param_columns: List of parameter column names to process. If None, uses all columns.

    Returns:
        Dictionary mapping parameter names to their inferred bounds
    """
    df = pd.read_csv(csv_path)

    if param_columns is None:
        param_columns = df.columns.tolist()

    bounds = {}

    for param in param_columns:
        if param not in df.columns:
            print(f"Warning: Column '{param}' not found in CSV")
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
