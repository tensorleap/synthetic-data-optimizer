"""
Optuna-based Bayesian optimizer for synthetic data parameter optimization.
"""

import optuna
from pathlib import Path
from typing import Dict, List, Tuple


class OptunaOptimizer:
    """
    Optuna-based optimizer using TPE (Tree-structured Parzen Estimator) sampler.

    Supports grouped/conditional parameters where different groups (e.g., shapes)
    have different parameter sets. Optuna's TPE learns which groups produce
    better outcomes through native categorical parameter modeling.

    Features:
    - Multi-objective optimization (configurable metrics)
    - Grouped parameter bounds with conditional search space
    - Proper ask/tell pattern with pending trials tracking
    - Pareto front tracking for trade-off analysis
    - SQLite persistence for study state
    """

    def __init__(
        self,
        experiment_dir: Path,
        config: Dict,
        param_bounds: Dict[str, Dict],
        group_names: List[str]
    ):
        """
        Initialize Optuna optimizer.

        Args:
            experiment_dir: Path to experiment directory for SQLite storage
            config: Experiment configuration dict with optimization_metrics, etc.
            param_bounds: Dict mapping group names to their parameter bounds
                          e.g., {'circle': {'void_count_mean': [1.0, 10.0], ...}}
            group_names: Names for each group
        """
        self.experiment_dir = Path(experiment_dir)
        self.config = config
        self.study_path = self.experiment_dir / "optuna_study.db"

        # Validate inputs
        if not param_bounds or not group_names:
            raise ValueError("param_bounds and group_names are required")
        if set(param_bounds.keys()) != set(group_names):
            raise ValueError(
                f"param_bounds keys {set(param_bounds.keys())} must match "
                f"group_names {set(group_names)}"
            )

        self.group_names = group_names
        self.param_bounds = param_bounds

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
        multivariate = optimizer_config.get('multivariate', True)

        # Set n_startup_trials: ensure all groups get explored
        # Default: 20 trials per group
        if 'n_startup_trials' in optimizer_config:
            n_startup_trials = optimizer_config['n_startup_trials']
        else:
            n_startup_trials = 20 * len(self.group_names)

        # Multi-objective optimization with configurable metrics
        # For grouped mode with conditional params, suppress independent sampling warnings
        # (expected behavior when using dynamic search space with multivariate TPE)
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            directions=['minimize'] * n_objectives,  # minimize all metrics
            sampler=optuna.samplers.TPESampler(
                seed=config.get('random_seed', 42),
                n_startup_trials=n_startup_trials,
                multivariate=multivariate,
                warn_independent_sampling=False
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
        print(f"  Groups: {', '.join(self.group_names)}")
        for group_name in self.group_names:
            params = list(self.param_bounds[group_name].keys())
            print(f"    {group_name}: {len(params)} params")

    def _define_grouped_search_space(self, trial: optuna.Trial) -> Tuple[str, Dict]:
        """
        Define search space for grouped bounds using Optuna's native categorical handling.

        Optuna's TPE naturally learns which group produces better outcomes through
        its categorical parameter modeling. Only the selected group's parameters
        are suggested (conditional parameters).

        Note: Parameters are prefixed with group name for Optuna's internal tracking
        (e.g., 'circle__void_count_mean') but returned without prefix in the dict.

        Returns:
            selected_group: The group name selected by Optuna
            group_params: Dict of parameters for the selected group (keys without prefix)
        """
        # Let Optuna learn which group is best via categorical sampling
        selected_group = trial.suggest_categorical("group", self.group_names)

        # Suggest parameters only for the selected group (conditional params)
        # Prefix with group name to avoid Optuna conflicts between groups
        group_bounds = self.param_bounds[selected_group]
        params = {}
        for param_name, bounds in group_bounds.items():
            optuna_param_name = f"{selected_group}__{param_name}"
            params[param_name] = self._suggest_single_param(trial, optuna_param_name, bounds)

        return selected_group, params

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
        metrics_list: List[Dict[str, float]],
        iteration: int,
        config: Dict
    ) -> Tuple[List[Tuple[str, Dict]], bool]:
        """
        Suggest next distribution specifications.

        Each trial selects one group and suggests parameters for that group only.
        Optuna's TPE learns which groups produce better outcomes over time.

        Args:
            metrics_list: List of metric dicts from previous iteration
            iteration: Current iteration number
            config: Experiment configuration dict

        Returns:
            suggestions: List of (group_name, params_dict) tuples
            converged: Whether optimization should stop
        """
        # Report results from PREVIOUS iteration
        if iteration > 0 and (iteration - 1) in self.pending_trials:
            prev_trials = self.pending_trials[iteration - 1]

            if len(metrics_list) != len(prev_trials):
                raise ValueError(
                    f"Mismatch: {len(metrics_list)} metric dicts provided but "
                    f"{len(prev_trials)} trials pending for iteration {iteration-1}"
                )

            print(f"  Reporting results for iteration {iteration-1}:")
            print(f"    Completing {len(prev_trials)} trials from iteration {iteration-1}")

            for trial, metrics in zip(prev_trials, metrics_list):
                trial_values = [metrics[metric_name] for metric_name in self.optimization_metrics]
                self.study.tell(trial, trial_values)

                dist_id = metrics.get('param_set_id', 'unknown')
                metrics_str = ', '.join([f"{name}={metrics[name]:.4f}"
                                        for name in self.optimization_metrics])
                print(f"      Trial {dist_id}: {metrics_str}")

            del self.pending_trials[iteration - 1]

        # Ask for next batch of distributions
        n_distributions = config.get('iteration_batch_size', 8)
        suggestions = []
        trials = []

        print(f"\n  Suggesting {n_distributions} distributions for iteration {iteration}...")

        for i in range(n_distributions):
            trial = self.study.ask()
            group_name, params = self._define_grouped_search_space(trial)
            suggestions.append((group_name, params))
            trials.append(trial)
            print(f"    Trial {i}: group={group_name}")

        self.pending_trials[iteration] = trials
        print(f"  Stored {len(trials)} pending trials for iteration {iteration}")

        # Convergence check
        max_iterations = config.get('max_iterations', 10)
        converged = iteration >= max_iterations

        if converged:
            print(f"  Convergence: Reached max iterations ({max_iterations})")

        pareto_size = len(self.get_pareto_front())
        print(f"  Current Pareto front size: {pareto_size}")

        return suggestions, converged
