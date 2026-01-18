from pathlib import Path
from src.data_generation.tensorleap_data_generator import TensorleapDataGenerator

if __name__ == '__main__':
    base_image_dir = Path(__file__).parent.parent / "data" / "base_chips"
    config_path = Path(__file__).parent.parent / "configs" / "local_experiment_config.yaml"
    output_dir = Path(__file__).parent.parent / "data" / "tensorleap_dataset"

    generator = TensorleapDataGenerator(
        base_image_dir=base_image_dir,
        config_path=config_path
    )

    circle_params = {
        'void_shape': {'probabilities': {'circle': 1.0}},
        'void_count': {'mean': 15, 'std': 3},
        'base_size': {'mean': 6, 'std': 2},
        'rotation': {'mean': 0, 'std': 0},
        'center_x': {'mean': 0.5, 'std': 0.1},
        'center_y': {'mean': 0.5, 'std': 0.1},
        'position_spread': {'mean': 0.6, 'std': 0.1},
    }

    ellipse_params = {
        'void_shape': {'probabilities': {'ellipse': 1.0}},
        'void_count': {'mean': 10, 'std': 3},
        'base_size': {'mean': 8, 'std': 2},
        'rotation': {'mean': 90, 'std': 45},
        'center_x': {'mean': 0.4, 'std': 0.15},
        'center_y': {'mean': 0.6, 'std': 0.15},
        'position_spread': {'mean': 0.7, 'std': 0.1},
    }

    irregular_params = {
        'void_shape': {'probabilities': {'irregular': 1.0}},
        'void_count': {'mean': 20, 'std': 4},
        'base_size': {'mean': 4, 'std': 1},
        'rotation': {'mean': 0, 'std': 0},
        'center_x': {'mean': 0.7, 'std': 0.1},
        'center_y': {'mean': 0.3, 'std': 0.1},
        'position_spread': {'mean': 0.9, 'std': 0.05},
    }

    metadata_df = generator.generate_dataset(
        output_dir=output_dir,
        real_distribution_type='real',
        synthetic_shapes=['circle', 'ellipse', 'irregular'],
        synthetic_shape_params={
            'circle': circle_params,
            'ellipse': ellipse_params,
            'irregular': irregular_params
        },
        n_real_samples=100,
        n_samples_per_shape=30,
        seed=42
    )

    print(f"\n[METADATA PREVIEW]")
    print(metadata_df.head(10))
    print(f"\n[SCRIPT NAME SUMMARY]")
    print(metadata_df['script_name'].value_counts())
