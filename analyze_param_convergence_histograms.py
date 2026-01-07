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

    param_bounds = config['param_bounds']

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

            # Take top N
            best_trials = sorted_trials[:top_n]

            # For synthetic: use the BEST trial as the center and generate n_samples
            # This represents sampling from the distribution that Optuna is converging to
            best_trial = sorted_trials[0]

            # Generate n_samples synthetic parameters using "far" distribution
            # but with the best trial's parameters as a starting point
            synthetic_samples = []
            for _ in range(n_samples):
                # Sample from "far" distribution (same as iteration generation)
                sample_set = sampler.sample_parameter_sets(
                    config['initial_condition'],
                    n_sets=1,
                    seed=None  # Random seed for variation
                )[0]
                synthetic_samples.append(sample_set)

            synthetic_params_by_iter[iteration] = synthetic_samples

    print(f"\nFound {len(synthetic_params_by_iter)} iterations")

    # Plot distribution for each numeric parameter
    print("\nGenerating distribution plots...")
    numeric_params = ['void_count', 'base_size', 'rotation', 'center_x', 'center_y', 'position_spread']

    for param_name in numeric_params:
        print(f"\n  Processing {param_name}...")
        plot_param_distributions(
            real_params,
            synthetic_params_by_iter,
            param_name,
            param_bounds[param_name],
            experiment_dir
        )

    print("\n" + "=" * 60)
    print("Distribution analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Configuration
    EXPERIMENT_DIR = Path("data/experiments/infineon_void_poc")
    TOP_N = 3  # Use top 3 best trials per iteration
    N_SAMPLES = 100  # Generate 100 samples per distribution for rich histograms

    print("=" * 60)
    print("PARAMETER DISTRIBUTION CONVERGENCE ANALYSIS")
    print("=" * 60)
    print(f"Experiment directory: {EXPERIMENT_DIR}")
    print(f"Top N trials per iteration: {TOP_N}")
    print(f"Samples per distribution: {N_SAMPLES}")
    print("=" * 60)

    # Run analysis
    analyze_param_distributions(EXPERIMENT_DIR, top_n=TOP_N, n_samples=N_SAMPLES)
