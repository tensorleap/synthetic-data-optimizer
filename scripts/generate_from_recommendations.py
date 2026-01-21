from pathlib import Path
import pandas as pd
from src.data_generation.tensorleap_data_generator import TensorleapDataGenerator
from scripts.translate_tensorleap_recommendations import (
    translate_recommendations_csv,
    calculate_n_samples_per_shape
)

if __name__ == '__main__':
    base_image_dir = Path(__file__).parent.parent / "data" / "base_chips"
    config_path = Path(__file__).parent.parent / "configs" / "infineon_simulation_config.yaml"
    recommendations_csv = Path(__file__).parent.parent / "data" / "next-trials-explicit_float_1.csv"
    output_dir = Path(__file__).parent.parent / "data" / "tensorleap_dataset_epoch_2"
    epoch = 2

    df = pd.read_csv(recommendations_csv)
    distribution_id = df['distribution_id'].iloc[0]

    print("="*60)
    print(f"GENERATING DATASET FROM TENSORLEAP RECOMMENDATIONS (EPOCH {epoch})")
    print(f"Distribution: {distribution_id}")
    print("="*60)

    shape_params = translate_recommendations_csv(recommendations_csv, distribution_id=distribution_id)
    n_samples_dict = calculate_n_samples_per_shape(recommendations_csv, distribution_id=distribution_id, target_total_samples=300)

    generator = TensorleapDataGenerator(
        base_image_dir=base_image_dir,
        config_path=config_path
    )

    metadata_df = generator.generate_synthetic_only(
        output_dir=output_dir,
        synthetic_shapes=['circular_shadow', 'complex_splatter', 'dark_bezel', 'hole_like', 'main_splatter', 'structured_void'],
        synthetic_shape_params=shape_params,
        n_samples_per_shape_dict=n_samples_dict,
        epoch=epoch,
        seed=42
    )

    print(f"\n[GENERATED DATASET PREVIEW]")
    for shape in ['circular_shadow', 'complex_splatter', 'dark_bezel', 'hole_like', 'main_splatter', 'structured_void']:
        print(f"{shape} samples: {len(metadata_df[metadata_df['script_name'] == shape])}")

    print(f"\nDataset saved to: {output_dir}")
