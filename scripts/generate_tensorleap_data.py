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

    circle_params = [
        {
            'void_shape': {'probabilities': {'circle': 1.0}},
            'void_count': {'min': 2, 'max': 6},
            'base_size': {'min': 8.2, 'max': 12.4},
            'rotation': {'min': 0.0, 'max': 0.0},
            'center_x': {'min': 0.4, 'max': 0.6},
            'center_y': {'min': 0.4, 'max': 0.6},
            'position_spread': {'min': 0.4, 'max': 0.6},
        },
        {
            'void_shape': {'probabilities': {'circle': 1.0}},
            'void_count': {'min': 20, 'max': 28},
            'base_size': {'min': 2.5, 'max': 4.5},
            'rotation': {'min': 0.0, 'max': 0.0},
            'center_x': {'min': 0.15, 'max': 0.25},
            'center_y': {'min': 0.75, 'max': 0.85},
            'position_spread': {'min': 0.75, 'max': 0.9},
        },
        {
            'void_shape': {'probabilities': {'circle': 1.0}},
            'void_count': {'min': 15, 'max': 22},
            'base_size': {'min': 15.1, 'max': 20.1},
            'rotation': {'min': 0.0, 'max': 0.0},
            'center_x': {'min': 0.7, 'max': 0.85},
            'center_y': {'min': 0.1, 'max': 0.2},
            'position_spread': {'min': 0.15, 'max': 0.25},
        },
        {
            'void_shape': {'probabilities': {'circle': 1.0}},
            'void_count': {'min': 8, 'max': 12},
            'base_size': {'min': 5.3, 'max': 8.7},
            'rotation': {'min': 0.0, 'max': 0.0},
            'center_x': {'min': 0.35, 'max': 0.65},
            'center_y': {'min': 0.35, 'max': 0.65},
            'position_spread': {'min': 0.45, 'max': 0.65},
        }
    ]

    ellipse_params = [
        {
            'void_shape': {'probabilities': {'ellipse': 1.0}},
            'void_count': {'min': 3, 'max': 7},
            'base_size': {'min': 9.1, 'max': 13.1},
            'rotation': {'min': 100.0, 'max': 260.0},
            'center_x': {'min': 0.4, 'max': 0.6},
            'center_y': {'min': 0.4, 'max': 0.6},
            'position_spread': {'min': 0.4, 'max': 0.6},
        },
        {
            'void_shape': {'probabilities': {'ellipse': 1.0}},
            'void_count': {'min': 18, 'max': 25},
            'base_size': {'min': 2.8, 'max': 4.2},
            'rotation': {'min': 10.0, 'max': 50.0},
            'center_x': {'min': 0.8, 'max': 0.9},
            'center_y': {'min': 0.1, 'max': 0.2},
            'position_spread': {'min': 0.15, 'max': 0.3},
        },
        {
            'void_shape': {'probabilities': {'ellipse': 1.0}},
            'void_count': {'min': 12, 'max': 18},
            'base_size': {'min': 16.3, 'max': 22.1},
            'rotation': {'min': 280.0, 'max': 350.0},
            'center_x': {'min': 0.1, 'max': 0.2},
            'center_y': {'min': 0.75, 'max': 0.85},
            'position_spread': {'min': 0.75, 'max': 0.9},
        },
        {
            'void_shape': {'probabilities': {'ellipse': 1.0}},
            'void_count': {'min': 5, 'max': 9},
            'base_size': {'min': 6.4, 'max': 10.2},
            'rotation': {'min': 135.0, 'max': 225.0},
            'center_x': {'min': 0.35, 'max': 0.65},
            'center_y': {'min': 0.35, 'max': 0.65},
            'position_spread': {'min': 0.4, 'max': 0.6},
        }
    ]

    irregular_params = [
        {
            'void_shape': {'probabilities': {'irregular': 1.0}},
            'void_count': {'min': 2, 'max': 6},
            'base_size': {'min': 8.1, 'max': 12.3},
            'rotation': {'min': 0.0, 'max': 0.0},
            'center_x': {'min': 0.4, 'max': 0.6},
            'center_y': {'min': 0.4, 'max': 0.6},
            'position_spread': {'min': 0.4, 'max': 0.6},
            'irregularity_pattern': {'probabilities': {'smooth': 1.0}},
        },
        {
            'void_shape': {'probabilities': {'irregular': 1.0}},
            'void_count': {'min': 22, 'max': 30},
            'base_size': {'min': 2.1, 'max': 3.5},
            'rotation': {'min': 0.0, 'max': 0.0},
            'center_x': {'min': 0.75, 'max': 0.9},
            'center_y': {'min': 0.75, 'max': 0.9},
            'position_spread': {'min': 0.8, 'max': 0.95},
            'irregularity_pattern': {'probabilities': {'jagged': 1.0}},
        },
        {
            'void_shape': {'probabilities': {'irregular': 1.0}},
            'void_count': {'min': 14, 'max': 20},
            'base_size': {'min': 18.2, 'max': 24.3},
            'rotation': {'min': 0.0, 'max': 0.0},
            'center_x': {'min': 0.1, 'max': 0.2},
            'center_y': {'min': 0.1, 'max': 0.2},
            'position_spread': {'min': 0.1, 'max': 0.2},
            'irregularity_pattern': {'probabilities': {'medium': 1.0}},
        },
        {
            'void_shape': {'probabilities': {'irregular': 1.0}},
            'void_count': {'min': 4, 'max': 8},
            'base_size': {'min': 7.2, 'max': 11.4},
            'rotation': {'min': 0.0, 'max': 0.0},
            'center_x': {'min': 0.4, 'max': 0.6},
            'center_y': {'min': 0.4, 'max': 0.6},
            'position_spread': {'min': 0.4, 'max': 0.6},
            'irregularity_pattern': {'probabilities': {'smooth': 1.0}},
        }
    ]

    metadata_df = generator.generate_dataset(
        output_dir=output_dir,
        real_distribution_type='real',
        synthetic_shapes=['circle', 'ellipse', 'irregular'],
        synthetic_shape_params={
            'circle': circle_params,
            'ellipse': ellipse_params,
            'irregular': irregular_params
        },
        n_real_samples=500,
        n_samples_per_shape_dict={
            'circle': 200,
            'ellipse': 200,
            'irregular': 200
        },
        seed=42
    )

    print(f"\n[METADATA PREVIEW]")
    print(metadata_df.head(10))
    print(f"\n[SCRIPT NAME SUMMARY]")
    print(metadata_df['script_name'].value_counts())
