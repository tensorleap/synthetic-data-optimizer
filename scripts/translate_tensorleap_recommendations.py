import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


def translate_recommendations_csv(
    csv_path: Path,
    distribution_id: str = 'dist_1',
    default_params: Dict = None
) -> Dict[str, Dict]:
    if default_params is None:
        default_params = {
            'void_count': {'mean': 15, 'std': 3},
            'base_size': {'mean': 6, 'std': 2},
            'rotation': {'mean': 0, 'std': 0},
            'center_x': {'mean': 0.5, 'std': 0.1},
            'center_y': {'mean': 0.5, 'std': 0.1},
            'position_spread': {'mean': 0.6, 'std': 0.1},
        }

    df = pd.read_csv(csv_path)

    df_filtered = df[df['distribution_id'] == distribution_id]

    if len(df_filtered) == 0:
        raise ValueError(f"No rows found for distribution_id='{distribution_id}'")

    print(f"Extracting parameters for: {distribution_id}")

    simulation_to_shape = {
        'simulation_1': 'circle',
        'simulation_2': 'ellipse',
        'simulation_3': 'irregular'
    }

    shape_params = {}

    for _, row in df_filtered.iterrows():
        sim_type = row['simulation_type']
        shape_name = simulation_to_shape[sim_type]

        params = {}

        for param_base in ['void_count', 'base_size', 'rotation', 'center_x', 'center_y', 'position_spread']:
            mean_col = f'metadata.simulation_{param_base}_mean'
            std_col = f'metadata.simulation_{param_base}_std'

            if mean_col in df.columns and pd.notna(row[mean_col]):
                mean_val = float(row[mean_col])
            else:
                mean_val = default_params[param_base]['mean']

            if std_col in df.columns and pd.notna(row[std_col]):
                std_val = float(row[std_col])
            else:
                std_val = default_params[param_base]['std']

            params[param_base] = {'mean': mean_val, 'std': std_val}

        params['void_shape'] = {'probabilities': {shape_name: 1.0}}

        shape_params[shape_name] = params

    print("\nTranslated parameters:")
    for shape, params in shape_params.items():
        print(f"\n{shape}:")
        for param_name, values in params.items():
            if param_name != 'void_shape':
                print(f"  {param_name}: mean={values['mean']}, std={values['std']}")

    return shape_params


def calculate_n_samples_per_shape(
    csv_path: Path,
    distribution_id: str = 'dist_1',
    target_total_samples: int = 300
) -> Dict[str, int]:
    df = pd.read_csv(csv_path)

    df_filtered = df[df['distribution_id'] == distribution_id]

    if len(df_filtered) == 0:
        raise ValueError(f"No rows found for distribution_id='{distribution_id}'")

    simulation_to_shape = {
        'simulation_1': 'circle',
        'simulation_2': 'ellipse',
        'simulation_3': 'irregular'
    }

    n_samples_original = {}
    total_original = 0

    for _, row in df_filtered.iterrows():
        sim_type = row['simulation_type']
        shape_name = simulation_to_shape[sim_type]
        n_original = int(row['n_samples'])
        n_samples_original[shape_name] = n_original
        total_original += n_original

    n_samples_normalized = {}
    min_samples_per_shape = 20

    print(f"\nNormalizing sample counts to ~{target_total_samples} total:")
    print(f"  Original total: {total_original}")
    print(f"  Minimum per shape: {min_samples_per_shape}")

    for shape_name, n_original in n_samples_original.items():
        proportion = n_original / total_original
        n_normalized = max(int(proportion * target_total_samples), min_samples_per_shape)
        n_samples_normalized[shape_name] = n_normalized
        print(f"  {shape_name}: {n_original} ({proportion:.1%}) → {n_normalized}")

    actual_total = sum(n_samples_normalized.values())
    print(f"  Actual total: {actual_total}")

    return n_samples_normalized


if __name__ == '__main__':
    csv_path = Path(__file__).parent.parent / "data" / "suggestions-more_distributions-epoch0.csv"

    shape_params = translate_recommendations_csv(csv_path, distribution_id='dist_1')
    n_samples = calculate_n_samples_per_shape(csv_path, distribution_id='dist_1', target_total_samples=300)

    print(f"\n{'='*60}")
    print("READY TO GENERATE")
    print(f"{'='*60}")
    print(f"Use these parameters in generate_tensorleap_data.py:")
    print(f"  synthetic_shape_params={shape_params}")
    print(f"  n_samples_per_shape_dict={n_samples}")
