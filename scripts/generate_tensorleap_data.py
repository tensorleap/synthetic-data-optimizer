from pathlib import Path
from src.data_generation.tensorleap_data_generator import TensorleapDataGenerator

if __name__ == '__main__':
    base_image_dir = Path(__file__).parent.parent / "data" / "base_chips"
    config_path = Path(__file__).parent.parent / "configs" / "infineon_simulation_config.yaml"
    output_dir = Path(__file__).parent.parent / "data" / "tensorleap_dataset"

    generator = TensorleapDataGenerator(
        base_image_dir=base_image_dir,
        config_path=config_path
    )

    metadata_df = generator.generate_dataset(
        output_dir=output_dir,
        real_distribution_type='circular_shadow',  # Using circular_shadow as baseline
        synthetic_shapes=['circular_shadow', 'complex_splatter', 'dark_bezel', 'hole_like', 'main_splatter', 'structured_void'],
        synthetic_distribution_type='circular_shadow',
        n_real_samples=100,
        n_samples_per_shape=50,
        seed=42
    )

    print(f"\n[METADATA PREVIEW]")
    print(metadata_df.head(10))
    print(f"\n[SCRIPT NAME SUMMARY]")
    print(metadata_df['script_name'].value_counts())
