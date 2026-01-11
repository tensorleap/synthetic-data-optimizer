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

        # Track pending trials from last ask() for potential future completion
        self.pending_trials = []

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

    def _build_distributions_for_group(self, group_name: str) -> Dict:
        """
        Build Optuna distributions dict for a specific group's parameters.

        Required for add_trial() to tell Optuna the type and range of each parameter.

        Args:
            group_name: The group to build distributions for

        Returns:
            Dict mapping param names to optuna.distributions objects
        """
        distributions = {
            "group": optuna.distributions.CategoricalDistribution(self.group_names)
        }

        group_bounds = self.param_bounds[group_name]
        for param_name, bounds in group_bounds.items():
            optuna_param_name = f"{group_name}__{param_name}"

            if isinstance(bounds, list) and len(bounds) == 2:
                if all(isinstance(b, (int, float)) for b in bounds):
                    is_int = all(isinstance(b, int) or (isinstance(b, float) and b.is_integer()) for b in bounds)
                    if is_int:
                        distributions[optuna_param_name] = optuna.distributions.IntDistribution(
                            int(bounds[0]), int(bounds[1])
                        )
                    else:
                        distributions[optuna_param_name] = optuna.distributions.FloatDistribution(
                            bounds[0], bounds[1]
                        )
                else:
                    distributions[optuna_param_name] = optuna.distributions.CategoricalDistribution(bounds)

        return distributions

    def suggest_next_distributions(
        self,
        current_distributions: List[Tuple[str, Dict]],
        metrics_list: List[Dict[str, float]],
        config: Dict
    ) -> List[Tuple[str, Dict]]:
        """
        Register current results and suggest next distribution specifications.

        This is the main optimization interface. Each call:
        1. Registers current distributions and their metrics with Optuna via add_trial()
        2. Asks Optuna for the next batch of suggestions

        The optimizer is agnostic to data source - it only sees distributions and metrics.
        Works uniformly for all iterations (initial external data or optimizer-suggested).

        Args:
            current_distributions: List of (group_name, params_dict) tuples that were evaluated
            metrics_list: List of metric dicts corresponding to current_distributions
            config: Experiment configuration dict

        Returns:
            suggestions: List of (group_name, params_dict) tuples for next iteration
        """
        if len(current_distributions) != len(metrics_list):
            raise ValueError(
                f"Mismatch: {len(current_distributions)} distributions but "
                f"{len(metrics_list)} metric dicts"
            )

        # Tell: Register current results with Optuna using add_trial()
        print(f"  Registering {len(current_distributions)} results with Optuna...")

        for idx, ((group_name, params), metrics) in enumerate(zip(current_distributions, metrics_list)):
            # Build the full params dict with prefixed names
            trial_params = {"group": group_name}
            for param_name, value in params.items():
                trial_params[f"{group_name}__{param_name}"] = value

            # Build distributions for this group
            distributions = self._build_distributions_for_group(group_name)

            # Get metric values
            trial_values = [metrics[metric_name] for metric_name in self.optimization_metrics]

            # Create and add the completed trial
            trial = optuna.trial.create_trial(
                params=trial_params,
                distributions=distributions,
                values=trial_values,
                state=optuna.trial.TrialState.COMPLETE
            )
            self.study.add_trial(trial)

            metrics_str = ', '.join([f"{name}={metrics[name]:.4f}"
                                    for name in self.optimization_metrics])
            print(f"    Trial {idx} ({group_name}): {metrics_str}")

        # Ask: Get next batch of suggestions
        n_distributions = config.get('iteration_batch_size', 8)
        suggestions = []
        self.pending_trials = []

        print(f"\n  Suggesting {n_distributions} distributions for next iteration...")

        for i in range(n_distributions):
            trial = self.study.ask()
            group_name, params = self._define_grouped_search_space(trial)
            suggestions.append((group_name, params))
            self.pending_trials.append(trial)
            print(f"    Suggestion {i}: group={group_name}")

        completed_count = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pareto_size = len(self.get_pareto_front())
        print(f"  Total completed trials: {completed_count}")
        print(f"  Current Pareto front size: {pareto_size}")

        return suggestions
