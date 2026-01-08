"""
Standalone script to visualize parameter distribution convergence.

Shows histograms comparing real vs synthetic parameter distributions over iterations,
demonstrating how optimization drives synthetic parameters toward the real distribution.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import optuna
import yaml
from pathlib import Path
from typing import Dict, List


def load_real_params_from_config(n_samples: int = 100) -> List[Dict]:
    """
    Generate real parameters using the same logic as the experiment.

    Args:
        n_samples: Number of parameter samples to generate for visualization

    Returns:
        List of real parameter dictionaries
    """
    from src.data_generation.parameter_sampler import ParameterSampler

    # Load config
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Generate real params with same seed
    sampler = ParameterSampler()
    real_params = sampler.sample_parameter_sets(
        'real',
        n_sets=n_samples,
        seed=config['random_seed']
    )

    return real_params


def circular_distance(angle1: float, angle2: float) -> float:
    """
    Compute circular distance between two angles in degrees.

    Args:
        angle1: First angle in degrees
        angle2: Second angle in degrees

    Returns:
        Minimum angular distance (0-180 degrees)
    """
    diff = abs(angle1 - angle2) % 360
    return min(diff, 360 - diff)


def _create_custom_distribution(trial_params: Dict, config: Dict) -> Dict:
    """
    Create a custom distribution from Optuna trial parameters.

    The trial parameters ARE the distribution parameters (means/stds),
    so we construct a distribution dict that ParameterSampler can use.

    Args:
        trial_params: Dictionary of optimized distribution parameters from Optuna
        config: Experiment config containing param_distributions structure

    Returns:
        Distribution dictionary in the format expected by ParameterSampler
    """
    # Get reference distribution structure (use 'real' as template)
    ref_dist = config['param_distributions']['real']

    custom_dist = {}

    # Handle void_shape (categorical) - use trial params directly
    if 'void_shape_circle' in trial_params:
        # Trial has logits for each shape category
        custom_dist['void_shape'] = {
            'probabilities': {
                'circle': trial_params.get('void_shape_circle', ref_dist['void_shape']['probabilities']['circle']),
                'ellipse': trial_params.get('void_shape_ellipse', ref_dist['void_shape']['probabilities']['ellipse']),
                'irregular': trial_params.get('void_shape_irregular', ref_dist['void_shape']['probabilities']['irregular'])
            }
        }
    else:
        # Fallback to reference distribution
        custom_dist['void_shape'] = ref_dist['void_shape']

    # Handle numeric parameters with mean/std
    numeric_params = ['void_count', 'base_size', 'rotation', 'center_x', 'center_y', 'position_spread']

    for param_name in numeric_params:
        mean_key = f"{param_name}_mean"
        std_key = f"{param_name}_std"

        if mean_key in trial_params and std_key in trial_params:
            custom_dist[param_name] = {
                'mean': trial_params[mean_key],
                'std': trial_params[std_key]
            }
        else:
            # Fallback to reference distribution
            custom_dist[param_name] = ref_dist[param_name]

    return custom_dist


def plot_param_distributions(
    real_params: List[Dict],
    synthetic_params_by_iter: Dict[int, List[Dict]],
    param_name: str,
    param_bounds: tuple,
    experiment_dir: Path
):
    """
    Plot histogram showing parameter distribution convergence over iterations.

    Args:
        real_params: List of real parameter dicts
        synthetic_params_by_iter: Dict mapping iteration -> list of synthetic param dicts
        param_name: Name of parameter to plot
        param_bounds: (min, max) bounds for the parameter
        experiment_dir: Experiment directory for saving
    """
    # Extract values for this parameter
    real_values = [p[param_name] for p in real_params]

    # Determine iterations to show (first and last only)
    iterations = sorted(synthetic_params_by_iter.keys())
    if len(iterations) == 0:
        print(f"No iterations to plot for {param_name}")
        return

    iterations_to_plot = [iterations[0]]
    if len(iterations) > 1:
        iterations_to_plot.append(iterations[-1])

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine bins based on parameter type and bounds
    if param_name == 'void_count':
        bins = np.arange(param_bounds[0], param_bounds[1] + 2) - 0.5  # Integer bins
    elif param_name == 'rotation':
        bins = np.linspace(0, 360, 37)  # 10-degree bins for rotation
    else:
        bins = np.linspace(param_bounds[0], param_bounds[1], 21)  # 20 bins for continuous

    # Plot real distribution
    ax.hist(real_values, bins=bins, alpha=0.6, label='Real',
            color='red', edgecolor='darkred', linewidth=1.5)

    # Plot synthetic distributions for selected iterations
    colors = ['orange', 'green']
    for idx, iteration in enumerate(iterations_to_plot):
        synth_params = synthetic_params_by_iter[iteration]
        synth_values = [p[param_name] for p in synth_params]

        label = f'Iteration {iteration}' if iteration > 0 else 'Iteration 0 (Initial)'
        ax.hist(synth_values, bins=bins, alpha=0.5, label=label,
                color=colors[idx], edgecolor='black', linewidth=1)

    # Formatting
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Parameter Distribution Convergence: {param_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Special handling for rotation (circular)
    if param_name == 'rotation':
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])

    plt.tight_layout()

    # Save
    viz_dir = experiment_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    save_path = viz_dir / f"param_distribution_{param_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved {param_name} distribution plot to {save_path}")


def plot_diversity_histograms(
    real_params: List[Dict],
    top_n_params_list: List[List[Dict]],
    top_n_trials: List,
    param_name: str,
    param_bounds: tuple,
    experiment_dir: Path
):
    """
    Plot overlapping histograms showing diversity of top N solutions.

    Args:
        real_params: List of real parameter dicts
        top_n_params_list: List of [list of sampled params] for each top N trial
        top_n_trials: List of top N trial objects (for getting objective values)
        param_name: Name of parameter to plot
        param_bounds: (min, max) bounds for the parameter
        experiment_dir: Experiment directory for saving
    """
    # Extract values for this parameter
    real_values = [p[param_name] for p in real_params]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Determine bins based on parameter type and bounds
    if param_name == 'void_count':
        bins = np.arange(param_bounds[0], param_bounds[1] + 2) - 0.5
    elif param_name == 'rotation':
        bins = np.linspace(0, 360, 37)
    else:
        bins = np.linspace(param_bounds[0], param_bounds[1], 21)

    # Plot real distribution
    ax.hist(real_values, bins=bins, alpha=0.7, label='Real',
            color='red', edgecolor='darkred', linewidth=2, histtype='step')

    # Plot top N synthetic distributions with different colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_n_params_list)))

    for idx, (synth_params, trial) in enumerate(zip(top_n_params_list, top_n_trials)):
        synth_values = [p[param_name] for p in synth_params]

        # Get objective values for label
        mmd_rbf = trial.values[0]
        label = f'Top {idx+1} (MMD={mmd_rbf:.4f})'

        ax.hist(synth_values, bins=bins, alpha=0.3, label=label,
                color=colors[idx], edgecolor=colors[idx], linewidth=1.5, histtype='stepfilled')

    # Formatting
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Solution Diversity: {param_name} (Top {len(top_n_params_list)} Best)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    # Special handling for rotation (circular)
    if param_name == 'rotation':
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])

    plt.tight_layout()

    # Save
    viz_dir = experiment_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    save_path = viz_dir / f"diversity_{param_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved {param_name} diversity plot to {save_path}")


def plot_shape_diversity(
    real_params: List[Dict],
    top_n_params_list: List[List[Dict]],
    top_n_trials: List,
    experiment_dir: Path
):
    """
    Plot stacked bar chart showing void_shape distribution diversity.

    Args:
        real_params: List of real parameter dicts
        top_n_params_list: List of [list of sampled params] for each top N trial
        top_n_trials: List of top N trial objects (for getting objective values)
        experiment_dir: Experiment directory for saving
    """
    # Shape categories
    shape_categories = ['circle', 'ellipse', 'irregular']

    # Count shapes for real distribution
    real_shapes = [p['void_shape'] for p in real_params]
    real_counts = {shape: real_shapes.count(shape) / len(real_shapes) for shape in shape_categories}

    # Count shapes for each top N distribution
    top_n_counts = []
    for synth_params in top_n_params_list:
        synth_shapes = [p['void_shape'] for p in synth_params]
        counts = {shape: synth_shapes.count(shape) / len(synth_shapes) for shape in shape_categories}
        top_n_counts.append(counts)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for grouped bar chart
    x = np.arange(len(shape_categories))
    width = 0.12  # Width of bars

    # Plot real distribution
    real_values = [real_counts[shape] for shape in shape_categories]
    ax.bar(x - width * (len(top_n_trials) / 2 + 0.5), real_values, width,
           label='Real', color='red', alpha=0.7, edgecolor='darkred', linewidth=1.5)

    # Plot top N distributions
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_n_trials)))

    for idx, (counts, trial, color) in enumerate(zip(top_n_counts, top_n_trials, colors)):
        values = [counts[shape] for shape in shape_categories]
        mmd_rbf = trial.values[0]
        label = f'Top {idx+1} (MMD={mmd_rbf:.4f})'

        offset = width * (idx - len(top_n_trials) / 2 + 0.5)
        ax.bar(x + offset, values, width, label=label,
               color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Formatting
    ax.set_xlabel('Void Shape', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_title(f'Solution Diversity: void_shape (Top {len(top_n_trials)} Best)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(shape_categories)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    # Save
    viz_dir = experiment_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    save_path = viz_dir / "diversity_void_shape.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved void_shape diversity plot to {save_path}")


def analyze_solution_diversity(experiment_dir: Path, top_n: int = 5, n_samples: int = 100):
    """
    Analyze and visualize diversity of top N best solutions.

    Args:
        experiment_dir: Path to experiment directory
        top_n: Number of best trials to analyze
        n_samples: Number of samples to generate for each distribution (for visualization)
    """
    from src.data_generation.parameter_sampler import ParameterSampler

    experiment_dir = Path(experiment_dir)

    # Load config for param bounds
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    distribution_param_bounds = config['distribution_param_bounds']

    # Load Optuna study
    study_path = experiment_dir / "optuna_study.db"
    if not study_path.exists():
        raise FileNotFoundError(f"Optuna study database not found: {study_path}")

    print("Loading Optuna study...")
    storage = f"sqlite:///{study_path}"
    experiment_name = experiment_dir.name
    study = optuna.load_study(study_name=experiment_name, storage=storage)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"  Total completed trials: {len(completed_trials)}")

    if len(completed_trials) == 0:
        print("\nNo completed trials found.")
        return

    # Get top N trials by primary objective (MMD RBF)
    sorted_trials = sorted(completed_trials, key=lambda t: t.values[0] if t.values else float('inf'))
    top_n_trials = sorted_trials[:top_n]

    print(f"\nTop {top_n} trials:")
    for idx, trial in enumerate(top_n_trials):
        print(f"  {idx+1}. Trial {trial.number}: MMD={trial.values[0]:.6f}, NN_dist={trial.values[1]:.6f}")

    # Load real parameters
    print(f"\nGenerating {n_samples} real parameter samples for visualization...")
    sampler = ParameterSampler()
    real_params = sampler.sample_parameter_sets(
        'real',
        n_sets=n_samples,
        seed=config['random_seed']
    )
    print(f"  Real parameter sets: {len(real_params)}")

    # Sample from each top N distribution
    print(f"\nSampling {n_samples} from each of top {top_n} distributions...")
    top_n_params_list = []

    for idx, trial in enumerate(top_n_trials):
        print(f"  Sampling from trial {trial.number}...")

        # Create custom distribution from trial params
        custom_dist = _create_custom_distribution(trial.params, config)

        # Temporarily override sampler's distributions
        original_dists = sampler.distributions
        sampler.distributions = {'custom': custom_dist}

        synthetic_samples = sampler.sample_parameter_sets(
            'custom',
            n_sets=n_samples,
            seed=None
        )

        # Restore original distributions
        sampler.distributions = original_dists

        top_n_params_list.append(synthetic_samples)

    # Plot diversity for void_shape (categorical)
    print("\nGenerating diversity plots...")
    print("\n  Processing void_shape...")
    plot_shape_diversity(
        real_params,
        top_n_params_list,
        top_n_trials,
        experiment_dir
    )

    # Plot diversity for each sampled parameter (iterate over real_params keys)
    sample_param_names = list(real_params[0].keys()) if real_params else []

    for param_name in sample_param_names:
        # Skip categorical parameters (void_shape) for histogram plots
        sample_value = real_params[0][param_name]
        if isinstance(sample_value, str):
            continue

        # Get bounds from distribution_param_bounds using _mean suffix
        bounds_key = f"{param_name}_mean"
        bounds = distribution_param_bounds.get(bounds_key)

        print(f"\n  Processing {param_name}...")
        plot_diversity_histograms(
            real_params,
            top_n_params_list,
            top_n_trials,
            param_name,
            bounds,
            experiment_dir
        )

    print("\n" + "=" * 60)
    print("Diversity analysis complete!")
    print("=" * 60)


def analyze_param_distributions(experiment_dir: Path, top_n: int = 3, n_samples: int = 100):
    """
    Analyze and visualize parameter distribution convergence.

    Args:
        experiment_dir: Path to experiment directory
        top_n: Number of best trials to use per iteration
        n_samples: Number of samples to generate for each distribution (for visualization)
    """
    from src.data_generation.parameter_sampler import ParameterSampler

    experiment_dir = Path(experiment_dir)

    # Load config for param bounds
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    distribution_param_bounds = config['distribution_param_bounds']

    # Load Optuna study
    study_path = experiment_dir / "optuna_study.db"
    if not study_path.exists():
        raise FileNotFoundError(f"Optuna study database not found: {study_path}")

    print("Loading Optuna study...")
    storage = f"sqlite:///{study_path}"
    experiment_name = experiment_dir.name
    study = optuna.load_study(study_name=experiment_name, storage=storage)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"  Total completed trials: {len(completed_trials)}")

    if len(completed_trials) == 0:
        print("\nNo completed trials found.")
        return

    # Load real parameters (generate n_samples for rich histograms)
    print(f"\nGenerating {n_samples} real parameter samples for visualization...")
    real_params = load_real_params_from_config(n_samples=n_samples)
    print(f"  Real parameter sets: {len(real_params)}")

    # Get iteration info
    iter_0_data = experiment_dir / "iterations" / "iter_000" / "metadata.json"
    if iter_0_data.exists():
        with open(iter_0_data, 'r') as f:
            metadata = json.load(f)
            n_param_sets = metadata.get('n_param_sets', 10)
    else:
        n_param_sets = 10

    print(f"  Parameter sets per iteration: {n_param_sets}")
    print(f"  Using top {top_n} best trials per iteration")
    print(f"  Generating {n_samples} samples per synthetic distribution")

    # Initialize ParameterSampler for synthetic distributions
    sampler = ParameterSampler()

    # Group trials by iteration and get top N
    synthetic_params_by_iter = {}

    for i, trial in enumerate(completed_trials):
        iteration = i // n_param_sets

        if i % n_param_sets == 0:
            current_iter_trials = []

        current_iter_trials.append(trial)

        # Process when we have all trials for this iteration
        if (i + 1) % n_param_sets == 0 or i == len(completed_trials) - 1:
            # Sort by first objective (MMD RBF)
            sorted_trials = sorted(current_iter_trials,
                                 key=lambda t: t.values[0] if t.values else float('inf'))

            # Take best trial
            best_trial = sorted_trials[0]

            # The trial params ARE distribution parameters (means/stds)
            # We need to create a custom distribution and sample from it
            # For iteration 0, use initial_condition distribution
            if iteration == 0:
                # Sample from initial condition (e.g., 'far')
                synthetic_samples = sampler.sample_parameter_sets(
                    config['initial_condition'],
                    n_sets=n_samples,
                    seed=None
                )
            else:
                # For other iterations: create custom distribution from best trial's params
                # and sample from it
                # Note: This requires modifying the sampler's internal distributions temporarily
                custom_dist = _create_custom_distribution(best_trial.params, config)

                # Temporarily override sampler's distributions
                original_dists = sampler.distributions
                sampler.distributions = {'custom': custom_dist}

                synthetic_samples = sampler.sample_parameter_sets(
                    'custom',
                    n_sets=n_samples,
                    seed=None
                )

                # Restore original distributions
                sampler.distributions = original_dists

            synthetic_params_by_iter[iteration] = synthetic_samples

    print(f"\nFound {len(synthetic_params_by_iter)} iterations")

    # Plot distribution for each sampled parameter (iterate over real_params keys)
    print("\nGenerating distribution plots...")

    # Get parameter names from first real sample
    sample_param_names = list(real_params[0].keys()) if real_params else []

    for param_name in sample_param_names:
        # Skip categorical parameters (void_shape) for histogram plots
        sample_value = real_params[0][param_name]
        if isinstance(sample_value, str):
            continue

        # Get bounds from distribution_param_bounds using _mean suffix
        bounds_key = f"{param_name}_mean"
        bounds = distribution_param_bounds.get(bounds_key)

        print(f"\n  Processing {param_name}...")
        plot_param_distributions(
            real_params,
            synthetic_params_by_iter,
            param_name,
            bounds,
            experiment_dir
        )

    print("\n" + "=" * 60)
    print("Distribution analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    # Configuration
    MODE = 'diversity'
    EXPERIMENT_DIR = Path("data/experiments/infineon_void_poc")
    N_SAMPLES = 100  # Generate 100 samples per distribution for rich histograms

    # Check for mode argument
    mode = sys.argv[1] if len(sys.argv) > 1 else MODE

    if mode == "diversity":
        # Diversity analysis mode
        TOP_N = 5  # Analyze top 5 best solutions

        print("=" * 60)
        print("SOLUTION DIVERSITY ANALYSIS")
        print("=" * 60)
        print(f"Experiment directory: {EXPERIMENT_DIR}")
        print(f"Top N best trials: {TOP_N}")
        print(f"Samples per distribution: {N_SAMPLES}")
        print("=" * 60)

        # Run diversity analysis
        analyze_solution_diversity(EXPERIMENT_DIR, top_n=TOP_N, n_samples=N_SAMPLES)

    else:
        # Convergence analysis mode (default)
        TOP_N = 5  # Use top 3 best trials per iteration

        print("=" * 60)
        print("PARAMETER DISTRIBUTION CONVERGENCE ANALYSIS")
        print("=" * 60)
        print(f"Experiment directory: {EXPERIMENT_DIR}")
        print(f"Top N trials per iteration: {TOP_N}")
        print(f"Samples per distribution: {N_SAMPLES}")
        print("=" * 60)

        # Run convergence analysis
        analyze_param_distributions(EXPERIMENT_DIR, top_n=TOP_N, n_samples=N_SAMPLES)
