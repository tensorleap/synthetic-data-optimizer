"""
Optuna-based Bayesian optimizer for synthetic data parameter optimization.
"""

import math
import optuna
from pathlib import Path
from typing import Dict, List, Tuple


class OptunaOptimizer:
    """
    Optuna-based optimizer using TPE (Tree-structured Parzen Estimator) sampler.

    Jointly optimizes shape probabilities and all shape-specific parameters.
    Each trial suggests:
    1. Shape logits (converted to probabilities via softmax downstream)
    2. All parameters for all shapes simultaneously

    Features:
    - Multi-objective optimization (configurable metrics)
    - Joint shape probability + parameter optimization
    - Proper ask/tell pattern with pending trials tracking
    - Pareto front tracking for trade-off analysis
    - SQLite persistence for study state
    """

    def __init__(
        self,
        experiment_dir: Path,
        config: Dict,
        param_bounds: Dict[str, Dict],
        logit_bounds: Tuple[float, float] = (-5.0, 5.0)
    ):
        """
        Initialize Optuna optimizer.

        Args:
            experiment_dir: Path to experiment directory for SQLite storage
            config: Experiment configuration dict with optimization_metrics, etc.
            param_bounds: Dict mapping simulation names to their parameter bounds
                          e.g., {'simulation_1': {'void_count_mean': [1.0, 10.0], ...}}
                          Simulation names (group_names) are inferred from the keys
            logit_bounds: Min/max bounds for shape logits (default: -5.0 to 5.0)
        """
        self.experiment_dir = Path(experiment_dir)
        self.config = config
        self.study_path = self.experiment_dir / "optuna_study.db"
        self.logit_bounds = logit_bounds

        # Validate inputs
        if not param_bounds:
            raise ValueError("param_bounds is required and cannot be empty")

        # Infer group names from param_bounds keys (sorted for deterministic order)
        self.group_names = sorted(param_bounds.keys())
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

        # Set n_startup_trials: higher for joint optimization due to larger search space
        # Default: 50 trials (more than per-group mode due to ~18 params)
        if 'n_startup_trials' in optimizer_config:
            n_startup_trials = optimizer_config['n_startup_trials']
        else:
            # Count total params: logits + all shape params
            total_params = len(self.group_names)  # logits
            for group_bounds in self.param_bounds.values():
                total_params += len(group_bounds)
            n_startup_trials = max(50, 3 * total_params)

        # Multi-objective optimization with configurable metrics
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

        print(f"Initialized OptunaOptimizer (joint mode)")
        print(f"  Study storage: {self.study_path}")
        print(f"  Study name: {study_name}")
        print(f"  Objectives: {n_objectives} ({', '.join(self.optimization_metrics)})")
        print(f"  Logit bounds: {logit_bounds}")
        print(f"  TPE startup trials: {n_startup_trials}")
        print(f"  TPE multivariate: {multivariate}")
        print(f"  Groups: {', '.join(self.group_names)}")
        for group_name in self.group_names:
            params = list(self.param_bounds[group_name].keys())
            print(f"    {group_name}: {len(params)} params")

    def _define_joint_search_space(self, trial: optuna.Trial) -> Dict:
        """
        Define joint search space: shape logits + all shape params.

        Each trial suggests:
        1. Shape logits for all groups (converted to probabilities downstream)
        2. All parameters for all shapes

        Returns:
            Dict with shape_logit_* keys and {shape}__{param} keys
        """
        params = {}

        # 1. Shape logits (converted to probs downstream via softmax)
        for group_name in self.group_names:
            logit = trial.suggest_float(
                f'shape_logit_{group_name}',
                self.logit_bounds[0],
                self.logit_bounds[1]
            )
            params[f'shape_logit_{group_name}'] = logit

        # 2. All params for all shapes
        for group_name in self.group_names:
            group_bounds = self.param_bounds[group_name]
            handled_max = set()

            for param_name, bounds in group_bounds.items():
                if param_name in handled_max:
                    continue

                if param_name.endswith('_min'):
                    base = param_name[:-4]
                    max_name = f'{base}_max'
                    if max_name in group_bounds:
                        min_key = f'{group_name}__{param_name}'
                        max_key = f'{group_name}__{max_name}'
                        min_val, max_val = self._suggest_min_max_pair(
                            trial=trial,
                            min_key=min_key,
                            max_key=max_key,
                            min_bounds=bounds,
                            max_bounds=group_bounds[max_name]
                        )

                        params[min_key] = min_val
                        params[max_key] = max_val
                        handled_max.add(max_name)
                        continue

                optuna_key = f'{group_name}__{param_name}'
                params[optuna_key] = self._suggest_single_param(trial, optuna_key, bounds)

        self._validate_min_max_pairs(params)
        return params

    def _validate_min_max_pairs(self, params: Dict) -> None:
        """
        Validate that every *_min/*_max pair in params satisfies min < max.
        """
        min_keys = [k for k in params if k.endswith('_min')]
        for min_key in min_keys:
            base = min_key[:-4]
            max_key = f'{base}_max'
            if max_key not in params:
                continue
            min_val = params[min_key]
            max_val = params[max_key]
            assert min_val < max_val, (
                f"Invalid min/max pair: {min_key}={min_val} must be < {max_key}={max_val}"
            )

    def _suggest_min_max_pair(
        self,
        trial: optuna.Trial,
        min_key: str,
        max_key: str,
        min_bounds,
        max_bounds
    ) -> tuple:
        """
        Suggest a min/max pair with the constraint min < max.

        Returns:
            (min_val, max_val)
        """

        min_val = self._suggest_single_param(trial, min_key, min_bounds)

        is_int = all(
            isinstance(b, int) or (isinstance(b, float) and b.is_integer())
            for b in min_bounds + max_bounds
        )
        if is_int:
            lower = max(int(max_bounds[0]), int(min_val) + 1)
            upper = int(max_bounds[1])
            assert lower <= upper, (
                f"Invalid bounds for {max_key}: {lower}..{upper} from {max_bounds} "
                f"with {min_key}={min_val}"
            )
            max_val = trial.suggest_int(max_key, lower, upper)
        else:
            lower = max(max_bounds[0], float(min_val) + 1e-6)
            upper = float(max_bounds[1])
            assert lower < upper, (
                f"Invalid bounds for {max_key}: {lower}..{upper} from {max_bounds} "
                f"with {min_key}={min_val}"
            )
            max_val = trial.suggest_float(max_key, lower, upper)

        return min_val, max_val

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

    def _bounds_to_distribution(self, bounds) -> optuna.distributions.BaseDistribution:
        """
        Convert bounds to an Optuna distribution object.

        Args:
            bounds: Either [min, max] for numerical or list of categories for categorical

        Returns:
            Optuna distribution object
        """
        if isinstance(bounds, list) and len(bounds) == 2:
            if all(isinstance(b, (int, float)) for b in bounds):
                is_int = all(isinstance(b, int) or (isinstance(b, float) and b.is_integer()) for b in bounds)
                if is_int:
                    return optuna.distributions.IntDistribution(int(bounds[0]), int(bounds[1]))
                else:
                    return optuna.distributions.FloatDistribution(bounds[0], bounds[1])
            else:
                return optuna.distributions.CategoricalDistribution(bounds)
        elif isinstance(bounds, list):
            return optuna.distributions.CategoricalDistribution(bounds)

        raise ValueError(f"Invalid bounds format: {bounds}")

    def _build_full_distributions(self) -> Dict:
        """
        Build Optuna distributions for all params (logits + all shape params).

        Required for add_trial() to tell Optuna the type and range of each parameter.

        Returns:
            Dict mapping param names to optuna.distributions objects
        """
        distributions = {}

        # Logit distributions
        for group_name in self.group_names:
            distributions[f'shape_logit_{group_name}'] = optuna.distributions.FloatDistribution(
                self.logit_bounds[0], self.logit_bounds[1]
            )

        # All shape param distributions
        for group_name in self.group_names:
            group_bounds = self.param_bounds[group_name]
            for param_name, bounds in group_bounds.items():
                optuna_key = f'{group_name}__{param_name}'
                distributions[optuna_key] = self._bounds_to_distribution(bounds)

        return distributions

    def get_pareto_front(self) -> List[optuna.trial.FrozenTrial]:
        """
        Get non-dominated trials from Pareto front.

        Returns:
            List of trials on the Pareto front (non-dominated solutions)
        """
        return self.study.best_trials

    def get_best_trials_as_distributions(
        self,
        top_n: int = None
    ) -> List[Tuple[str, Dict]]:
        """
        Get the best trials seen so far as distribution specifications.

        For single-objective: returns trials sorted by metric (best first)
        For multi-objective: returns Pareto front trials

        Args:
            top_n: Number of best trials to return. If None, returns all best trials.
                   For single-objective, this limits the sorted list.
                   For multi-objective, this limits the Pareto front.

        Returns:
            List of (dist_id, params_dict) tuples with probabilities (not logits)
        """
        # Get best trials
        if len(self.optimization_metrics) == 1:
            # Single objective: sort all completed trials by metric
            completed = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            sorted_trials = sorted(completed, key=lambda t: t.values[0])
            best_trials = sorted_trials[:top_n] if top_n else sorted_trials
        else:
            # Multi-objective: use Pareto front
            pareto_trials = self.get_pareto_front()
            best_trials = pareto_trials[:top_n] if top_n else pareto_trials

        # Convert to distribution format with probabilities
        distributions = []
        for trial in best_trials:
            dist_id = f"trial_{trial.number}"

            # Convert logits to probabilities for output
            params_with_probs = self.convert_logits_to_probs_in_params(trial.params)

            distributions.append((dist_id, params_with_probs))

        return distributions

    @staticmethod
    def sample_counts_to_logits(sample_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Convert sample counts to logits (inverse softmax).

        Used to infer initial shape probabilities from data where sample counts
        across shape CSVs determine the distribution.

        Args:
            sample_counts: Dict mapping group names to sample counts
                          e.g., {'circle': 100, 'ellipse': 80, 'irregular': 70}

        Returns:
            Dict with shape_logit_* keys
            e.g., {'shape_logit_circle': -0.22, 'shape_logit_ellipse': -0.44, ...}
        """
        total = sum(sample_counts.values())
        if total == 0:
            raise ValueError("Total sample count cannot be zero")

        logits = {}
        for shape, count in sample_counts.items():
            # Compute probability, clamp to avoid log(0)
            prob = max(count / total, 1e-6)
            # Inverse softmax: logit = log(prob)
            # (constant offset cancels out in softmax)
            logits[f'shape_logit_{shape}'] = math.log(prob)

        return logits

    @staticmethod
    def logits_to_probabilities(params: Dict) -> Dict[str, float]:
        """
        Convert shape logits in params dict to probabilities via softmax.

        Args:
            params: Dict containing shape_logit_* keys

        Returns:
            Dict mapping shape names to probabilities (sum to 1.0)
        """
        # Extract logits
        logit_prefix = 'shape_logit_'
        logits = {}
        for key, value in params.items():
            if key.startswith(logit_prefix):
                shape = key[len(logit_prefix):]
                logits[shape] = value

        if not logits:
            return {}

        # Softmax: exp(z_i) / sum(exp(z_j))
        # Subtract max for numerical stability
        max_logit = max(logits.values())
        exp_logits = {k: math.exp(v - max_logit) for k, v in logits.items()}
        total = sum(exp_logits.values())

        return {k: v / total for k, v in exp_logits.items()}

    @staticmethod
    def convert_logits_to_probs_in_params(params: Dict) -> Dict:
        """
        Convert shape_logit_* keys to shape_prob_* keys with softmax probabilities.

        This is for output formatting only - internally the optimizer still uses logits.

        Args:
            params: Dict with shape_logit_* and other parameter keys

        Returns:
            New dict with shape_prob_* instead of shape_logit_*, plus all other params
        """
        # Extract logits and compute probabilities
        logit_prefix = 'shape_logit_'
        logits = {}
        other_params = {}

        for key, value in params.items():
            if key.startswith(logit_prefix):
                shape = key[len(logit_prefix):]
                logits[shape] = value
            else:
                other_params[key] = value

        # Compute softmax probabilities
        if logits:
            max_logit = max(logits.values())
            exp_logits = {k: math.exp(v - max_logit) for k, v in logits.items()}
            total = sum(exp_logits.values())
            probs = {f'shape_prob_{k}': v / total for k, v in exp_logits.items()}
        else:
            probs = {}

        # Return combined dict with probabilities + other params
        return {**probs, **other_params}

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
            current_distributions: List of (dist_id, params_dict) where params_dict
                                   contains shape_logit_* and {shape}__{param} keys
            metrics_list: List of metric dicts corresponding to current_distributions
            config: Experiment configuration dict

        Returns:
            suggestions: List of (dist_id, params_dict) tuples for next iteration
        """
        if len(current_distributions) != len(metrics_list):
            raise ValueError(
                f"Mismatch: {len(current_distributions)} distributions but "
                f"{len(metrics_list)} metric dicts"
            )

        # Build full distributions once (same for all trials in joint mode)
        distributions = self._build_full_distributions()

        # Tell: Register current results with Optuna using add_trial()
        print(f"  Registering {len(current_distributions)} results with Optuna...")

        for idx, ((dist_id, params), metrics) in enumerate(zip(current_distributions, metrics_list)):
            # Get metric values
            trial_values = [metrics[metric_name] for metric_name in self.optimization_metrics]

            # Create and add the completed trial
            trial = optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                values=trial_values,
                state=optuna.trial.TrialState.COMPLETE
            )
            self.study.add_trial(trial)

            # Log probabilities for readability
            probs = self.logits_to_probabilities(params)
            probs_str = ', '.join([f"{k}={v:.2%}" for k, v in sorted(probs.items())])
            metrics_str = ', '.join([f"{name}={metrics[name]:.4f}"
                                    for name in self.optimization_metrics])

        # Ask: Get next batch of suggestions
        n_distributions = config.get('iteration_batch_size', 8)
        suggestions = []
        self.pending_trials = []

        # Get current trial count for generating dist_ids
        completed_count = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])

        print(f"\n  Suggesting {n_distributions} distributions for next iteration...")

        for i in range(n_distributions):
            trial = self.study.ask()
            params_with_logits = self._define_joint_search_space(trial)
            dist_id = f"dist_{completed_count + i}"

            # Convert logits to probabilities for output only
            params_with_probs = self.convert_logits_to_probs_in_params(params_with_logits)
            suggestions.append((dist_id, params_with_probs))
            self.pending_trials.append(trial)

            # Log suggestion with probabilities
            probs = self.logits_to_probabilities(params_with_logits)
            probs_str = ', '.join([f"{k}={v:.2%}" for k, v in sorted(probs.items())])

        pareto_size = len(self.get_pareto_front())
        print(f"  Total completed trials: {completed_count}")
        print(f"  Current Pareto front size: {pareto_size}")

        return suggestions
