import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


def translate_recommendations_csv(
    csv_path: Path,
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

    simulation_to_shape = {
        'simulation_1': 'circle',
        'simulation_2': 'ellipse',
        'simulation_3': 'irregular'
    }

    shape_params = {}

    for _, row in df.iterrows():
        sim_type = row['simulation_type']
        shape_name = simulation_to_shape[sim_type]

        params = default_params.copy()

        for param_base in ['void_count', 'base_size', 'rotation', 'center_x', 'center_y', 'position_spread']:
            mean_col = f'metadata.simulation_{param_base}_mean'
            std_col = f'metadata.simulation_{param_base}_std'

            if mean_col in df.columns and pd.notna(row[mean_col]):
                mean_val = float(row[mean_col])
            else:
                mean_val = params[param_base]['mean']

            if std_col in df.columns and pd.notna(row[std_col]):
                std_val = float(row[std_col])
            else:
                std_val = params[param_base]['std']

            params[param_base] = {'mean': mean_val, 'std': std_val}

        params['void_shape'] = {'probabilities': {shape_name: 1.0}}

        shape_params[shape_name] = params

    print("Translated Tensorleap recommendations:")
    for shape, params in shape_params.items():
        print(f"\n{shape}:")
        for param_name, values in params.items():
            if param_name != 'void_shape':
                print(f"  {param_name}: mean={values['mean']}, std={values['std']}")

    return shape_params


def calculate_n_samples_per_shape(csv_path: Path, divisor: int = 20) -> int:
    df = pd.read_csv(csv_path)
    n_samples_original = int(df['n_samples'].iloc[0])
    n_samples_adjusted = n_samples_original // divisor

    print(f"\nSample count adjustment:")
    print(f"  Original: {n_samples_original}")
    print(f"  Divisor: {divisor}")
    print(f"  Adjusted: {n_samples_adjusted}")

    return n_samples_adjusted


if __name__ == '__main__':
    csv_path = Path(__file__).parent.parent / "data" / "next_trials-All_3ChipTypes_seg_model_deployed-epoch0 (2).csv"

    shape_params = translate_recommendations_csv(csv_path)
    n_samples = calculate_n_samples_per_shape(csv_path, divisor=20)

    print(f"\n{'='*60}")
    print("READY TO GENERATE")
    print(f"{'='*60}")
    print(f"Use these parameters in generate_tensorleap_data.py:")
    print(f"  synthetic_shape_params={shape_params}")
    print(f"  n_samples_per_shape={n_samples}")
