"""
Experiment Runner for orchestrating the optimization loop.

Agnostic to data source - receives embeddings and distributions from external source.
Data generation/loading happens outside this class.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from ..optimization.metrics import compute_per_param_set_metrics
from ..optimization.optuna_optimizer import OptunaOptimizer
from ..utils.bounds_inference import get_param_bounds
from ..visualization.experiment_reporter import ExperimentReporter
from .iteration_manager import IterationManager


class ExperimentRunner:
    """
    Orchestrates the optimization loop.

    Receives data from external source, computes metrics, updates optimizer.
    """

    def __init__(
        self,
        config_path: Path,
        param_bounds: Dict[str, Dict] = None,
        group_names: List[str] = None
    ):
        """
        Initialize experiment runner.

        Args:
            config_path: Path to experiment configuration YAML file
            param_bounds: Optional parameter bounds dict. If None, inferred from data.
            group_names: Optional list of group names. Required if param_bounds is provided.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._setup_experiment_dir()

        # Iteration manager
        self.iteration_manager = IterationManager(Path(self.config['experiment_dir']))

        # Get parameter bounds from data (or use provided)
        if param_bounds is None:
            param_bounds, group_names = get_param_bounds()

        # Optuna optimizer
        self.optimizer = OptunaOptimizer(
            experiment_dir=Path(self.config['experiment_dir']),
            config=self.config,
            param_bounds=param_bounds,
            group_names=group_names
        )

        # Experiment reporter for visualizations
        self.reporter = ExperimentReporter(Path(self.config['experiment_dir']), config=self.config)

        # Real embeddings reference (set via set_real_embeddings)
        self.real_embeddings_400d = None

        # Track embeddings by iteration for visualization
        self.synthetic_embeddings_by_iter = {}

        print(f"Initialized ExperimentRunner")
        print(f"Experiment directory: {self.config['experiment_dir']}")

    def _setup_experiment_dir(self):
        """
        Setup experiment directory based on experiment name.

        Creates directory as data/experiments/{experiment_name}.
        If directory exists, appends _1, _2, etc. until finding unused name.
        Updates self.config['experiment_dir'] with the resolved path.
        """
        base_dir = Path(self.config.get('experiments_base_dir', 'data/experiments'))
        exp_name = self.config['experiment_name']

        # Try base name first
        exp_dir = base_dir / exp_name
        if not exp_dir.exists():
            self.config['experiment_dir'] = str(exp_dir)
            return

        # Directory exists, find next available suffix
        suffix = 1
        while True:
            exp_dir = base_dir / f"{exp_name}_{suffix}"
            if not exp_dir.exists():
                self.config['experiment_dir'] = str(exp_dir)
                print(f"Experiment '{exp_name}' exists, using '{exp_name}_{suffix}'")
                return
            suffix += 1

    def set_real_embeddings(self, embeddings_400d: np.ndarray):
        """
        Set reference distribution (one-time setup).

        Args:
            embeddings_400d: Real embeddings (400D) to use as reference
        """
        self.real_embeddings_400d = embeddings_400d
        print(f"Set real embeddings: {embeddings_400d.shape}")

    def run_iteration(
        self,
        iteration: int,
        current_distributions: List[Tuple[str, Dict]],
        synthetic_embeddings_400d: np.ndarray,
        synthetic_metadata: List[Dict]
    ) -> List[Tuple[str, Dict]]:
        """
        Process external data and get next suggestions.

        Every iteration is the same:
        1. Receive external data (distributions + embeddings)
        2. Compute metrics
        3. Tell optimizer and get next suggestions

        Args:
            iteration: Iteration number (0, 1, 2, ...)
            current_distributions: What was evaluated [(group_name, params_dict), ...]
            synthetic_embeddings_400d: Embeddings from external source (400D)
            synthetic_metadata: Metadata with param_set_id for grouping

        Returns:
            next_suggestions: What to try next [(group_name, params_dict), ...]
        """
        if self.real_embeddings_400d is None:
            raise ValueError("Real embeddings not set. Call set_real_embeddings() first.")

        print("\n" + "=" * 60)
        print(f"ITERATION {iteration}")
        print("=" * 60)

        n_distributions = len(current_distributions)
        print(f"Received {n_distributions} distributions with {len(synthetic_embeddings_400d)} embeddings")

        # Compute per-distribution metrics
        print("\n[1/2] Computing per-distribution metrics...")
        metrics_list = compute_per_param_set_metrics(
            synthetic_embeddings_400d,
            synthetic_metadata,
            self.real_embeddings_400d,
            n_param_sets=n_distributions
        )

        # Print metrics
        print(f"\n  Per-distribution metrics:")
        for idx, metrics in enumerate(metrics_list):
            metrics_str = ', '.join([f"{name}={metrics[name]:.4f}"
                                    for name in self.config['optimization_metrics']])
            group_name = current_distributions[idx][0]
            print(f"    Distribution {idx} ({group_name}): {metrics_str}")

        # Compute average for display
        avg_metrics = {
            'mmd_rbf': np.mean([m['mmd_rbf'] for m in metrics_list]),
            'mean_nn_distance': np.mean([m['mean_nn_distance'] for m in metrics_list]),
            'coverage': np.mean([m['coverage'] for m in metrics_list])
        }
        print(f"\n  Average: MMD={avg_metrics['mmd_rbf']:.4f}, NN={avg_metrics['mean_nn_distance']:.4f}")

        # Tell optimizer and get next suggestions
        print("\n[2/2] Updating optimizer and getting next suggestions...")
        next_suggestions = self.optimizer.suggest_next_distributions(
            current_distributions=current_distributions,
            metrics_list=metrics_list,
            config=self.config
        )
        print(f"Optimizer suggests {len(next_suggestions)} distributions for next iteration")

        # Save iteration data
        self._save_iteration_data(
            iteration, current_distributions, synthetic_embeddings_400d,
            metrics_list, avg_metrics
        )

        print(f"\nIteration {iteration} complete!")
        return next_suggestions

    def _save_iteration_data(
        self,
        iteration: int,
        distributions: List[Tuple[str, Dict]],
        embeddings: np.ndarray,
        metrics_list: List[Dict],
        avg_metrics: Dict
    ):
        """Save all iteration artifacts."""
        # Save to iteration manager
        self.iteration_manager.save_iteration(
            iteration=iteration,
            params=[{'group': g, **p} for g, p in distributions],
            embeddings=embeddings,
            metrics=metrics_list,
            metadata={
                'n_embeddings': len(embeddings),
                'n_distributions': len(distributions)
            }
        )

        # Save distribution outputs (top distributions from optimizer)
        self._save_distribution_outputs(iteration, distributions, metrics_list)

        # Track for visualization
        self.reporter.update_metrics_history(iteration, avg_metrics)
        self.synthetic_embeddings_by_iter[iteration] = embeddings

    def _save_distribution_outputs(
        self,
        iteration: int,
        distributions: List[Tuple[str, Dict]],
        metrics_list: List[Dict]
    ):
        """
        Save distribution parameters and top N distributions at end of iteration.

        Saves:
        - distributions.json: The distribution params suggested this iteration
        - top_distributions.json: Top N best distributions so far (by primary metric)
        """
        import json
        import optuna

        iter_dir = self.iteration_manager.iterations_dir / f"iter_{iteration:03d}"

        # Save this iteration's distribution parameters
        current_distributions = []
        for idx, (group_name, params) in enumerate(distributions):
            current_distributions.append({
                'distribution_id': idx,
                'group': group_name,
                'params': params,
                'metrics': metrics_list[idx] if idx < len(metrics_list) else None
            })

        distributions_path = iter_dir / "distributions.json"
        with open(distributions_path, 'w') as f:
            json.dump(current_distributions, f, indent=2)
        print(f"Saved distribution parameters to {distributions_path}")

        # Save top N distributions from all completed trials
        top_n = self.config.get('top_n_distributions', 10)
        completed_trials = [
            t for t in self.optimizer.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if completed_trials:
            # Sort by primary metric (first optimization metric, minimize)
            sorted_trials = sorted(completed_trials, key=lambda t: t.values[0])
            top_trials = sorted_trials[:top_n]

            top_distributions = []
            for trial in top_trials:
                group = trial.params.get('group', 'unknown')
                prefix = f"{group}__"
                params = {
                    k.replace(prefix, ''): v
                    for k, v in trial.params.items()
                    if k.startswith(prefix)
                }
                metrics = {
                    self.optimizer.optimization_metrics[i]: trial.values[i]
                    for i in range(len(self.optimizer.optimization_metrics))
                }
                top_distributions.append({
                    'trial_number': trial.number,
                    'group': group,
                    'params': params,
                    'metrics': metrics
                })

            top_path = iter_dir / "top_distributions.json"
            with open(top_path, 'w') as f:
                json.dump(top_distributions, f, indent=2)
            print(f"Saved top {len(top_distributions)} distributions to {top_path}")

