from pathlib import Path
import yaml
from src.data_generation.tensorleap_data_generator import TensorleapDataGenerator


def generate_edge_case_from_config(config_path: Path, output_base_dir: Path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    edge_case_name = config['name']
    print(f"\n{'='*60}")
    print(f"GENERATING EDGE CASE: {edge_case_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")

    output_dir = output_base_dir / edge_case_name

    base_image_dir = Path(__file__).parent.parent / "data" / "base_chips"
    tensorleap_config_path = Path(__file__).parent.parent / "configs" / "local_experiment_config.yaml"

    generator = TensorleapDataGenerator(
        base_image_dir=base_image_dir,
        config_path=tensorleap_config_path
    )

    synthetic_shapes = set()
    synthetic_shape_params = {}
    n_samples_per_shape_dict = {}

    for dist_id, dist_config in config['distributions'].items():
        shape = dist_config['shape']
        synthetic_shapes.add(shape)
        n_samples = dist_config['n_samples']

        if shape not in n_samples_per_shape_dict:
            n_samples_per_shape_dict[shape] = 0
        n_samples_per_shape_dict[shape] += n_samples

        if shape not in synthetic_shape_params:
            synthetic_shape_params[shape] = []

        shape_spec = {'void_shape': {'probabilities': {shape: 1.0}}}

        for param_name, param_value in dist_config['params'].items():
            if param_name.endswith('_min'):
                base_param = param_name[:-4]
                max_key = f'{base_param}_max'

                if max_key not in dist_config['params']:
                    raise ValueError(
                        f"Missing '{max_key}' for distribution '{dist_id}'. "
                        f"Both min and max must be provided."
                    )

                min_val = param_value
                max_val = dist_config['params'][max_key]

                shape_spec[base_param] = {'min': min_val, 'max': max_val}

            elif param_name == 'irregularity_pattern':
                shape_spec['irregularity_pattern'] = {
                    'probabilities': {param_value: 1.0}
                }

        synthetic_shape_params[shape].append(shape_spec)

    metadata_df = generator.generate_synthetic_only(
        output_dir=output_dir,
        synthetic_shapes=list(synthetic_shapes),
        synthetic_shape_params=synthetic_shape_params,
        n_samples_per_shape_dict=n_samples_per_shape_dict,
        epoch=0,
        seed=42
    )

    print(f"\n[EDGE CASE GENERATED]")
    for shape in synthetic_shapes:
        count = len(metadata_df[metadata_df['script_name'] == shape])
        print(f"  {shape}: {count} samples")
    print(f"\nOutput: {output_dir}")

    return metadata_df


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate edge case test data')
    parser.add_argument(
        '--config',
        type=str,
        help='Path to edge case config YAML file'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all edge cases'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/edge_cases',
        help='Base output directory'
    )

    args = parser.parse_args()

    edge_cases_config_dir = Path(__file__).parent.parent / "configs" / "edge_cases"
    output_base_dir = Path(args.output_dir)

    if args.all:
        config_files = sorted(edge_cases_config_dir.glob("*.yaml"))
        print(f"Found {len(config_files)} edge case configs")

        for config_file in config_files:
            try:
                generate_edge_case_from_config(config_file, output_base_dir)
            except Exception as e:
                print(f"\nERROR in {config_file.name}: {e}")
                continue

    elif args.config:
        config_path = Path(args.config)
        generate_edge_case_from_config(config_path, output_base_dir)

    else:
        parser.error("Specify either --config or --all")


if __name__ == '__main__':
    main()
