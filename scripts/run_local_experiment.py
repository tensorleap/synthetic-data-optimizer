"""
Local experiment runner with data generation.

This script runs the complete optimization loop using generated data.
For testing the framework end-to-end.

Usage:
    python -m scripts.run_local_experiment --config configs/experiment_config.yaml
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

from src.orchestration.experiment_runner import ExperimentRunner
from src.data_generation.parameter_sampler import ParameterSampler
from src.data_generation.void_generator import VoidGenerator
from src.embedding.dinov2_embedder import DinoV2Embedder
from src.embedding.pca_projector import PCAProjector


def get_param_bounds_from_config(config: Dict) -> Tuple[Dict[str, Dict], List[str]]:
    """
    Infer parameter bounds from config's param_distributions.

    For local experiments, derives bounds from the min/max across all distribution
    types (real, close, far) to ensure initial conditions are within bounds.

    Format matches get_param_bounds(): {group_name: {param_name: [min, max]}}
    where param_name includes _mean/_std suffixes.

    Args:
        config: Experiment config with param_distributions

    Returns:
        param_bounds: Dict mapping group names to their parameter bounds
        group_names: List of group names
    """
    param_distributions = config.get('param_distributions', {})
    group_names = ['circle', 'ellipse', 'irregular']

    # Collect all mean/std values across distribution types
    param_ranges = {}  # param_name -> list of values

    for dist_type, dist_params in param_distributions.items():
        for param_name, param_config in dist_params.items():
            if param_name == 'void_shape':
                continue
            if isinstance(param_config, dict) and 'mean' in param_config:
                mean_key = f'{param_name}_mean'
                std_key = f'{param_name}_std'

                if mean_key not in param_ranges:
                    param_ranges[mean_key] = []
                    param_ranges[std_key] = []

                param_ranges[mean_key].append(param_config['mean'])
                param_ranges[std_key].append(param_config['std'])

    # Create bounds with margin (min - 20%, max + 20%)
    def compute_bounds(values: List[float]) -> List[float]:
        min_val = min(values)
        max_val = max(values)
        margin = 0.2 * (max_val - min_val) if max_val != min_val else 0.2 * abs(max_val)
        return [float(max(0, min_val - margin)), float(max_val + margin)]

    # Build param_bounds per group
    # All groups share the same params except ellipse has rotation
    base_params = ['void_count', 'base_size', 'center_x', 'center_y', 'position_spread']
    ellipse_params = base_params + ['rotation']

    param_bounds = {}
    for group in group_names:
        group_params = ellipse_params if group == 'ellipse' else base_params
        param_bounds[group] = {}

        for param in group_params:
            mean_key = f'{param}_mean'
            std_key = f'{param}_std'
            if mean_key in param_ranges:
                param_bounds[group][mean_key] = compute_bounds(param_ranges[mean_key])
            if std_key in param_ranges:
                param_bounds[group][std_key] = compute_bounds(param_ranges[std_key])

    return param_bounds, group_names


def generate_initial_distributions(
    config: Dict,
    sampler: ParameterSampler,
    n_distributions: int,
    param_bounds: Dict[str, Dict],
    group_names: List[str]
) -> List[Tuple[str, Dict]]:
    """
    Generate initial distributions in joint format from config.

    Samples from the 'initial_condition' distribution type and converts
    to the joint format expected by the optimizer:
    - shape_logit_* keys for all groups
    - {group}__{param} keys for all params of all groups

    NOTE: This function modifies param_bounds in-place to expand bounds
    if initial condition values fall outside them.

    Args:
        config: Experiment config dict
        sampler: Parameter sampler
        n_distributions: Number of distributions to generate
        param_bounds: Dict mapping group names to their parameter bounds
        group_names: List of group names

    Returns:
        List of (dist_id, joint_params_dict) tuples
    """
    import math

    initial_condition = config.get('initial_condition', 'close')
    seed = config.get('random_seed', 42)

    # Get the std values from the initial condition's param_distributions
    initial_dist_config = config.get('param_distributions', {}).get(initial_condition, {})

    # Sample parameter sets
    param_sets = sampler.sample_parameter_sets(
        initial_condition,
        n_sets=n_distributions,
        seed=seed
    )

    # Convert to joint format: (dist_id, joint_params_dict)
    distributions = []
    for i, params in enumerate(param_sets):
        sampled_shape = params['void_shape']

        # Build joint params dict
        joint_params = {}

        # Add shape logits - higher logit for the sampled shape
        for group_name in group_names:
            if group_name == sampled_shape:
                # Give higher probability to sampled shape
                joint_params[f'shape_logit_{group_name}'] = 1.0
            else:
                joint_params[f'shape_logit_{group_name}'] = -1.0

        # Add all params for all shapes
        for group_name in group_names:
            group_bounds = param_bounds.get(group_name, {})

            for param_name, bounds in group_bounds.items():
                optuna_key = f'{group_name}__{param_name}'

                # Get base param name (remove _mean/_std suffix)
                if param_name.endswith('_mean'):
                    base_param = param_name[:-5]
                elif param_name.endswith('_std'):
                    base_param = param_name[:-4]
                else:
                    base_param = param_name

                if param_name.endswith('_mean'):
                    # Use sampled value if this is the sampled shape, else midpoint
                    if group_name == sampled_shape and base_param in params:
                        value = params[base_param]
                    else:
                        value = (bounds[0] + bounds[1]) / 2

                    # Expand bounds if value is outside
                    if value < bounds[0]:
                        bounds[0] = float(value)
                    if value > bounds[1]:
                        bounds[1] = float(value)

                    joint_params[optuna_key] = value

                elif param_name.endswith('_std'):
                    # Get std from config
                    std_value = initial_dist_config.get(base_param, {}).get('std', 0.1)

                    # Expand bounds if needed
                    if std_value < bounds[0]:
                        bounds[0] = float(std_value)
                    if std_value > bounds[1]:
                        bounds[1] = float(std_value)

                    joint_params[optuna_key] = std_value

        distributions.append((f"init_{i}", joint_params))

    return distributions


def generate_synthetic_data(
    distributions: List[Tuple[str, Dict]],
    sampler: ParameterSampler,
    generator: VoidGenerator,
    embedder: DinoV2Embedder,
    pca: PCAProjector,
    replications: int,
    seed_offset: int,
    group_names: List[str]
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Generate synthetic data for given distributions.

    Args:
        distributions: List of (dist_id, joint_params_dict) tuples in joint format
        sampler: Parameter sampler
        generator: Void generator
        embedder: DinoV2 embedder
        pca: Fitted PCA projector
        replications: Number of samples per distribution
        seed_offset: Seed offset for reproducibility
        group_names: List of group names

    Returns:
        embeddings_400d: Generated embeddings (N, 400)
        metadata: List of metadata dicts with param_set_id
    """
    all_params = []

    for dist_idx, (dist_id, joint_params) in enumerate(distributions):
        # Sample concrete parameters from joint distribution
        params = sampler.sample_from_joint_distribution(
            joint_params,
            group_names,
            n_samples=replications,
            seed=seed_offset + dist_idx
        )

        # Tag with distribution ID (generator uses this for param_set_id grouping)
        for p in params:
            p['distribution_id'] = dist_idx
        all_params.extend(params)

    # Generate images
    images, metadata = generator.generate_batch(
        all_params,
        replications=1,
        seed_offset=seed_offset
    )

    # Extract embeddings
    embeddings_768d = embedder.embed_batch(images)
    embeddings_400d = pca.transform(embeddings_768d)

    return embeddings_400d, metadata


def run_local_experiment(config_path: Path):
    """
    Run complete optimization loop with generated data.

    Args:
        config_path: Path to experiment config YAML
    """
    print("\n" + "=" * 60)
    print("LOCAL EXPERIMENT (with data generation)")
    print("=" * 60)

    # Load config first to derive bounds
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Derive param bounds from config's param_distributions (for local testing)
    param_bounds, group_names = get_param_bounds_from_config(config)

    # Initialize data generation components first (needed to sample initial distributions)
    sampler = ParameterSampler(config_path)
    generator = VoidGenerator(Path(config['base_image_dir']))
    embedder = DinoV2Embedder(model_name=config['dino_model'])

    # Generate initial distributions BEFORE creating runner
    # This may expand param_bounds in-place if initial values fall outside
    n_distributions = config.get('iteration_batch_size', 8)
    initial_distributions = generate_initial_distributions(
        config, sampler, n_distributions, param_bounds, group_names
    )
    print(f"\n[Setup] Generated {len(initial_distributions)} initial distributions")

    # Initialize experiment runner with (potentially expanded) bounds
    runner = ExperimentRunner(config_path, param_bounds=param_bounds, group_names=group_names)
    config = runner.config  # Use runner's config (has experiment_dir resolved)

    # Setup: Generate real embeddings
    print("\n[Setup] Generating real distribution...")
    seed = config.get('random_seed', 42)
    real_params = sampler.sample_parameter_sets(
        'real',
        n_sets=config['real']['param_sets'],
        seed=seed
    )
    real_images, _ = generator.generate_batch(
        real_params,
        replications=config['real']['replications'],
        seed_offset=0
    )
    print(f"  Generated {len(real_images)} real images")

    # Extract embeddings and fit PCA
    print("  Extracting embeddings and fitting PCA...")
    real_embeddings_768d = embedder.embed_batch(real_images)
    pca = PCAProjector(n_components=config['pca_embedding_dim'])
    real_embeddings_400d = pca.fit_transform(real_embeddings_768d, verbose=False)

    # Set real embeddings in runner
    runner.set_real_embeddings(real_embeddings_400d)

    # Use the initial distributions generated earlier
    current_distributions = initial_distributions

    # Run optimization loop
    max_iterations = config.get('max_iterations', 10)
    replications = config.get('replications_per_iteration', 10)

    print(f"\nStarting optimization loop (max {max_iterations} iterations)...")

    for iteration in range(max_iterations):
        # Generate synthetic data for current distributions
        embeddings_400d, metadata = generate_synthetic_data(
            current_distributions,
            sampler,
            generator,
            embedder,
            pca,
            replications=replications,
            seed_offset=(iteration + 1) * 10000,
            group_names=group_names
        )

        # Run iteration
        next_suggestions = runner.run_iteration(
            iteration=iteration,
            current_distributions=current_distributions,
            synthetic_embeddings_400d=embeddings_400d,
            synthetic_metadata=metadata
        )

        # Use suggestions for next iteration
        current_distributions = next_suggestions

    print("\n" + "=" * 60)
    print("LOCAL EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Total iterations: {max_iterations}")
    print(f"Results saved to: {config['experiment_dir']}")


def main():
    parser = argparse.ArgumentParser(description='Run local experiment with data generation')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment_config.yaml',
        help='Path to experiment config YAML'
    )
    args = parser.parse_args()

    run_local_experiment(Path(args.config))


if __name__ == '__main__':
    main()
