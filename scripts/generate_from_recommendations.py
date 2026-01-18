from pathlib import Path
from src.data_generation.tensorleap_data_generator import TensorleapDataGenerator
from scripts.translate_tensorleap_recommendations import (
    translate_recommendations_csv,
    calculate_n_samples_per_shape
)

if __name__ == '__main__':
    base_image_dir = Path(__file__).parent.parent / "data" / "base_chips"
    config_path = Path(__file__).parent.parent / "configs" / "local_experiment_config.yaml"
    recommendations_csv = Path(__file__).parent.parent / "data" / "next_trials-All_3ChipTypes_seg_model_deployed-epoch0 (2).csv"
    output_dir = Path(__file__).parent.parent / "data" / "tensorleap_dataset_epoch1"

    print("="*60)
    print("GENERATING DATASET FROM TENSORLEAP RECOMMENDATIONS")
    print("="*60)

    shape_params = translate_recommendations_csv(recommendations_csv)
    n_samples = calculate_n_samples_per_shape(recommendations_csv, divisor=20)

    generator = TensorleapDataGenerator(
        base_image_dir=base_image_dir,
        config_path=config_path
    )

    metadata_df = generator.generate_dataset(
        output_dir=output_dir,
        real_distribution_type='real',
        synthetic_shapes=['circle', 'ellipse', 'irregular'],
        synthetic_shape_params=shape_params,
        n_real_samples=100,
        n_samples_per_shape=n_samples,
        seed=42
    )

    print(f"\n[GENERATED DATASET PREVIEW]")
    print(f"Circle samples: {len(metadata_df[metadata_df['script_name'] == 'circle'])}")
    print(f"Ellipse samples: {len(metadata_df[metadata_df['script_name'] == 'ellipse'])}")
    print(f"Irregular samples: {len(metadata_df[metadata_df['script_name'] == 'irregular'])}")
    print(f"Real samples: {len(metadata_df[metadata_df['script_name'] == 'mixed'])}")

    print(f"\n[CIRCLE PARAMETERS]")
    circle_row = metadata_df[metadata_df['script_name'] == 'circle'].iloc[0]
    print(f"  base_size: {circle_row['base_size_mean']} ± {circle_row['base_size_std']}")
    print(f"  center: ({circle_row['center_x_mean']}, {circle_row['center_y_mean']})")
    print(f"  rotation: {circle_row['rotation_mean']} ± {circle_row['rotation_std']}")

    print(f"\n[ELLIPSE PARAMETERS]")
    ellipse_row = metadata_df[metadata_df['script_name'] == 'ellipse'].iloc[0]
    print(f"  base_size: {ellipse_row['base_size_mean']} ± {ellipse_row['base_size_std']}")
    print(f"  center: ({ellipse_row['center_x_mean']}, {ellipse_row['center_y_mean']})")
    print(f"  rotation: {ellipse_row['rotation_mean']} ± {ellipse_row['rotation_std']}")

    print(f"\nDataset saved to: {output_dir}")
