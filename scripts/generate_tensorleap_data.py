from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.tensorleap_data_generator import TensorleapDataGenerator

if __name__ == '__main__':
    base_image_dir = Path(__file__).parent.parent / "data" / "base_chips"
    config_path = Path(__file__).parent.parent / "configs" / "infineon_simulation_config.yaml"
    output_dir = Path(__file__).parent.parent / "data" / "tensorleap_dataset"

    generator = TensorleapDataGenerator(
        base_image_dir=base_image_dir,
        config_path=config_path
    )

    # Use variant distributions - each simulation type has 2 variants (a/b) for parameter variability
    from src.data_generation.tensorleap_data_generator import TensorleapDataGenerator

    circular_shadow_params = [
        generator.parameter_sampler.distributions['circular_shadow_a'],
        generator.parameter_sampler.distributions['circular_shadow_b']
    ]
    complex_splatter_params = [
        generator.parameter_sampler.distributions['complex_splatter_a'],
        generator.parameter_sampler.distributions['complex_splatter_b']
    ]
    dark_bezel_params = [
        generator.parameter_sampler.distributions['dark_bezel_a'],
        generator.parameter_sampler.distributions['dark_bezel_b']
    ]
    hole_like_params = [
        generator.parameter_sampler.distributions['hole_like_a'],
        generator.parameter_sampler.distributions['hole_like_b']
    ]
    main_splatter_params = [
        generator.parameter_sampler.distributions['main_splatter_a'],
        generator.parameter_sampler.distributions['main_splatter_b']
    ]
    structured_void_params = [
        generator.parameter_sampler.distributions['structured_void_a'],
        generator.parameter_sampler.distributions['structured_void_b']
    ]

    metadata_df = generator.generate_dataset(
        output_dir=output_dir,
        real_distribution_type='circular_shadow_a',
        synthetic_shapes=['circular_shadow', 'complex_splatter', 'dark_bezel', 'hole_like', 'main_splatter', 'structured_void'],
        synthetic_shape_params={
            'circular_shadow': circular_shadow_params,
            'complex_splatter': complex_splatter_params,
            'dark_bezel': dark_bezel_params,
            'hole_like': hole_like_params,
            'main_splatter': main_splatter_params,
            'structured_void': structured_void_params
        },
        n_real_samples=100,
        n_samples_per_shape_dict={
            'circular_shadow': 50,
            'complex_splatter': 50,
            'dark_bezel': 50,
            'hole_like': 50,
            'main_splatter': 50,
            'structured_void': 50
        },
        seed=42
    )

    print(f"\n[METADATA PREVIEW]")
    print(metadata_df.head(10))
    print(f"\n[SCRIPT NAME SUMMARY]")
    print(metadata_df['script_name'].value_counts())
