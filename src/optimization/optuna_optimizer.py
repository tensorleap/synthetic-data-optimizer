"""
Optuna-based Bayesian optimizer for synthetic data parameter optimization.
"""

import optuna
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from ..utils.bounds_inference import infer_bounds_from_csv


class OptunaOptimizer:
    """
    Optuna-based optimizer using TPE (Tree-structured Parzen Estimator) sampler.

    Features:
    - Multi-objective optimization (configurable metrics)
    - Proper ask/tell pattern with pending trials tracking
    - Per-parameter-set metrics evaluation
    - Pareto front tracking for trade-off analysis
    - SQLite persistence for study state
    - Respects parameter bounds and precision from config
    - Simple convergence based on max_iterations
    """

    def __init__(self, experiment_dir: Path, config: Dict, bounds_csv_path: Union[str, Path, None] = None):
        """
        Initialize Optuna optimizer.

        Args:
            experiment_dir: Path to experiment directory for SQLite storage
            config: Experiment configuration dict with optimization_metrics, etc.
            bounds_csv_path: Optional path to CSV for inferring parameter bounds.
                           If provided, overrides config['distribution_param_bounds']
        """
        self.experiment_dir = Path(experiment_dir)
        self.config = config
        self.study_path = self.experiment_dir / "optuna_study.db"

        # Infer bounds from CSV if provided
        if bounds_csv_path is not None:
            print(f"Inferring parameter bounds from CSV: {bounds_csv_path}")
            self.param_bounds = infer_bounds_from_csv(bounds_csv_path)
            print(f"  Inferred bounds for {len(self.param_bounds)} parameters")
        else:
            # Use bounds from config (legacy support)
            self.param_bounds = config.get('distribution_param_bounds', {})

        if not self.param_bounds:
            raise ValueError("No parameter bounds available. Provide bounds_csv_path or config['distribution_param_bounds']")

        # Get optimization metrics from config
        self.optimization_metrics = config.get('optimization_metrics', ['mmd_rbf', 'mean_nn_distance'])
        n_objectives = len(self.optimization_metrics)

        # Ensure experiment directory exists
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create or load Optuna study with SQLite persistence
        storage = f"sqlite:///{self.study_path}"
        study_name = config.get('experiment_name', 'optuna_study')

        # Get optimizer config
        optimizer_config = config.get('optimizer', {})
        n_startup_trials = optimizer_config.get('n_startup_trials', 10)
        multivariate = optimizer_config.get('multivariate', True)

        # Multi-objective optimization with configurable metrics
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            directions=['minimize'] * n_objectives,  # minimize all metrics
            sampler=optuna.samplers.TPESampler(
                seed=config.get('random_seed', 42),
                n_startup_trials=n_startup_trials,
                multivariate=multivariate
            )
        )

        # Track pending trials by iteration
        self.pending_trials = {}  # {iteration: [trial1, trial2, ...]}

        print(f"Initialized OptunaOptimizer")
        print(f"  Study storage: {self.study_path}")
        print(f"  Study name: {study_name}")
        print(f"  Objectives: {n_objectives} ({', '.join(self.optimization_metrics)})")
        print(f"  TPE startup trials: {n_startup_trials}")
        print(f"  TPE multivariate: {multivariate}")
        print(f"  Parameters: {', '.join(self.param_bounds.keys())}")

    def _define_search_space(self, trial: optuna.Trial) -> Dict:
        """
        Define search space generically based on inferred parameter bounds.

        The bounds come from CSV columns which are the actual optimization parameters.
        No assumptions about parameter names or meanings.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested parameter values
        """
        suggested_params = {}

        for param_name, bounds in self.param_bounds.items():
            # Check if nested parameter (dict) or flat parameter (list)
            if isinstance(bounds, dict):
                # Nested/conditional parameters not yet implemented
                raise NotImplementedError(
                    f"Conditional parameter '{param_name}' detected. "
                    "Conditional parameters are not yet supported."
                )
            else:
                # Flat parameter - suggest directly
                suggested_params[param_name] = self._suggest_single_param(trial, param_name, bounds)

        return suggested_params

    def _suggest_single_param(self, trial: optuna.Trial, param_name: str, bounds):
        """
        Suggest a single parameter value based on its bounds.

        Args:
            trial: Optuna trial object
            param_name: Name of the parameter
            bounds: Either [min, max] for numerical or list of categories for categorical

        Returns:
            Suggested parameter value
        """
        if isinstance(bounds, list):
            # Check if numerical (2-element list with numbers) or categorical (list of strings/values)
            if len(bounds) == 2 and all(isinstance(b, (int, float)) for b in bounds):
                # Numerical parameter: [min, max]
                min_val, max_val = bounds[0], bounds[1]

                # Check if integer type
                is_int = all(isinstance(b, int) or (isinstance(b, float) and b.is_integer()) for b in bounds)

                if is_int:
                    return trial.suggest_int(param_name, int(min_val), int(max_val))
                else:
                    return trial.suggest_float(param_name, min_val, max_val)
            else:
                # Categorical parameter: list of categories
                return trial.suggest_categorical(param_name, bounds)

        raise ValueError(f"Invalid bounds format for parameter '{param_name}': {bounds}")

    def get_pareto_front(self) -> List[optuna.trial.FrozenTrial]:
        """
        Get non-dominated trials from Pareto front.

        Returns:
            List of trials on the Pareto front (non-dominated solutions)
        """
        return self.study.best_trials

    def suggest_next_distributions(
        self,
        synthetic_embeddings: np.ndarray,
        synthetic_params: List[Dict],
        real_embeddings: np.ndarray,
        metrics_list: List[Dict[str, float]],
        iteration: int,
        config: Dict
    ) -> Tuple[List[Dict], bool]:
        """
        Suggest next distribution specifications for synthetic data generation.

        Args:
            synthetic_embeddings: (N, 400) embeddings of current synthetic samples
            synthetic_params: List of parameter dicts used to generate synthetic_embeddings
            real_embeddings: (M, 400) embeddings of real samples (fixed reference)
            metrics_list: List of metric dicts, one per distribution from previous iteration
            iteration: Current iteration number
            config: Experiment configuration dict

        Returns:
            next_distributions: List of distribution specifications for next iteration
            converged: Whether optimization should stop
        """
        # Report results from PREVIOUS iteration
        if iteration > 0 and (iteration - 1) in self.pending_trials:
            prev_trials = self.pending_trials[iteration - 1]

            # Verify we have metrics for each trial (each trial = one distribution)
            if len(metrics_list) != len(prev_trials):
                raise ValueError(
                    f"Mismatch: {len(metrics_list)} metric dicts provided but "
                    f"{len(prev_trials)} trials pending for iteration {iteration-1}"
                )

            print(f"  Reporting results for iteration {iteration-1}:")
            print(f"    Completing {len(prev_trials)} distributions from iteration {iteration-1}")

            # Tell Optuna about each trial's individual results
            for trial, metrics in zip(prev_trials, metrics_list):
                # Extract metric values based on configured optimization metrics
                trial_values = [metrics[metric_name] for metric_name in self.optimization_metrics]
                self.study.tell(trial, trial_values)

                # Print individual distribution results
                dist_id = metrics.get('param_set_id', 'unknown')
                metrics_str = ', '.join([f"{name}={metrics[name]:.4f}"
                                        for name in self.optimization_metrics])
                print(f"      Distribution {dist_id}: {metrics_str}")

            # Clean up completed trials
            del self.pending_trials[iteration - 1]

        # Ask for next batch of distributions
        n_distributions = config.get('iteration_batch_size', 8)
        next_distributions = []
        trials = []

        print(f"\n  Suggesting {n_distributions} distributions for iteration {iteration}...")

        for i in range(n_distributions):
            trial = self.study.ask()
            dist_spec = self._define_search_space(trial)
            next_distributions.append(dist_spec)
            trials.append(trial)

        # Store pending trials for this iteration
        self.pending_trials[iteration] = trials
        print(f"  Stored {len(trials)} pending trials for iteration {iteration}")

        # Simple convergence check: max iterations reached
        max_iterations = config.get('max_iterations', 10)
        converged = iteration >= max_iterations

        if converged:
            print(f"  Convergence: Reached max iterations ({max_iterations})")

        # Print Pareto front size
        pareto_size = len(self.get_pareto_front())
        print(f"  Current Pareto front size: {pareto_size}")

        return next_distributions, converged
