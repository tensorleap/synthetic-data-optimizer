"""
Optuna-based Bayesian optimizer for synthetic data parameter optimization.

Stage 3: Proper ask/tell pattern with pending trials tracking
"""

import optuna
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class OptunaOptimizer:
    """
    Optuna-based optimizer using TPE (Tree-structured Parzen Estimator) sampler.

    Stage 3 features:
    - Multi-objective optimization (minimize mmd_rbf, wasserstein, mean_nn_distance)
    - Proper ask/tell pattern with pending trials tracking
    - Pareto front tracking for trade-off analysis
    - SQLite persistence for study state
    - Respects parameter bounds and precision from config
    - Simple convergence based on max_iterations
    """

    def __init__(self, experiment_dir: Path, config: Dict):
        """
        Initialize Optuna optimizer.

        Args:
            experiment_dir: Path to experiment directory for SQLite storage
            config: Experiment configuration dict with param_bounds, param_precision, etc.
        """
        self.experiment_dir = Path(experiment_dir)
        self.config = config
        self.study_path = self.experiment_dir / "optuna_study.db"

        # Ensure experiment directory exists
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create or load Optuna study with SQLite persistence
        storage = f"sqlite:///{self.study_path}"
        study_name = config.get('experiment_name', 'optuna_study')

        # Get optimizer config
        optimizer_config = config.get('optimizer', {})
        n_startup_trials = optimizer_config.get('n_startup_trials', 10)
        multivariate = optimizer_config.get('multivariate', True)

        # Stage 2: Multi-objective optimization
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            directions=['minimize', 'minimize', 'minimize'],  # mmd_rbf, wasserstein, mean_nn_distance
            sampler=optuna.samplers.TPESampler(
                seed=config.get('random_seed', 42),
                n_startup_trials=n_startup_trials,
                multivariate=multivariate
            )
        )

        # Stage 3: Track pending trials by iteration
        self.pending_trials = {}  # {iteration: [trial1, trial2, ...]}

        print(f"Initialized OptunaOptimizer (Stage 3)")
        print(f"  Study storage: {self.study_path}")
        print(f"  Study name: {study_name}")
        print(f"  Objectives: 3 (mmd_rbf, wasserstein, mean_nn_distance)")
        print(f"  TPE startup trials: {n_startup_trials}")
        print(f"  TPE multivariate: {multivariate}")

    def _define_search_space(self, trial: optuna.Trial) -> Dict:
        """
        Define parameter search space with bounds and precision.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of parameters sampled from the search space
        """
        bounds = self.config['param_bounds']
        precision = self.config.get('param_precision', {})

        params = {}

        # Categorical parameter: void_shape
        params['void_shape'] = trial.suggest_categorical(
            'void_shape',
            bounds['void_shape']
        )

        # Integer parameter: void_count
        params['void_count'] = trial.suggest_int(
            'void_count',
            bounds['void_count'][0],
            bounds['void_count'][1]
        )

        # Continuous parameters with precision
        # base_size: precision 1 decimal (step 0.1)
        if 'base_size' in precision:
            step = 10 ** (-precision['base_size'])
        else:
            step = 0.1
        params['base_size'] = trial.suggest_float(
            'base_size',
            bounds['base_size'][0],
            bounds['base_size'][1],
            step=step
        )

        # rotation: precision 1 decimal (step 0.1)
        if 'rotation' in precision:
            step = 10 ** (-precision['rotation'])
        else:
            step = 0.1
        params['rotation'] = trial.suggest_float(
            'rotation',
            bounds['rotation'][0],
            bounds['rotation'][1],
            step=step
        )

        # center_x: precision 2 decimals (step 0.01)
        if 'center_x' in precision:
            step = 10 ** (-precision['center_x'])
        else:
            step = 0.01
        params['center_x'] = trial.suggest_float(
            'center_x',
            bounds['center_x'][0],
            bounds['center_x'][1],
            step=step
        )

        # center_y: precision 2 decimals (step 0.01)
        if 'center_y' in precision:
            step = 10 ** (-precision['center_y'])
        else:
            step = 0.01
        params['center_y'] = trial.suggest_float(
            'center_y',
            bounds['center_y'][0],
            bounds['center_y'][1],
            step=step
        )

        # position_spread: precision 2 decimals (step 0.01)
        if 'position_spread' in precision:
            step = 10 ** (-precision['position_spread'])
        else:
            step = 0.01
        params['position_spread'] = trial.suggest_float(
            'position_spread',
            bounds['position_spread'][0],
            bounds['position_spread'][1],
            step=step
        )

        return params

    def get_pareto_front(self) -> List[optuna.trial.FrozenTrial]:
        """
        Get non-dominated trials from Pareto front.

        Returns:
            List of trials on the Pareto front (non-dominated solutions)
        """
        return self.study.best_trials

    def suggest_next_parameters(
        self,
        synthetic_embeddings: np.ndarray,
        synthetic_params: List[Dict],
        real_embeddings: np.ndarray,
        metrics: Dict[str, float],
        iteration: int,
        config: Dict
    ) -> Tuple[List[Dict], bool]:
        """
        Suggest next parameter sets for synthetic data generation.

        Stage 3: Proper ask/tell pattern with pending trials tracking.

        Args:
            synthetic_embeddings: (N, 400) embeddings of current synthetic samples
            synthetic_params: List of parameter dicts used to generate synthetic_embeddings
            real_embeddings: (M, 400) embeddings of real samples (fixed reference)
            metrics: Distance metrics for current iteration (mmd_rbf, wasserstein, etc.)
            iteration: Current iteration number
            config: Experiment configuration dict

        Returns:
            next_params: List of parameter dicts for next iteration
            converged: Whether optimization should stop
        """
        # Stage 3: Step 1 - Report results from PREVIOUS iteration
        if iteration > 0 and (iteration - 1) in self.pending_trials:
            prev_trials = self.pending_trials[iteration - 1]

            # All parameter sets from same iteration share the same metrics
            trial_values = [
                metrics['mmd_rbf'],
                metrics['wasserstein'],
                metrics['mean_nn_distance']
            ]

            print(f"  Reporting results for iteration {iteration-1}:")
            print(f"    mmd_rbf: {trial_values[0]:.4f}")
            print(f"    wasserstein: {trial_values[1]:.4f}")
            print(f"    mean_nn_distance: {trial_values[2]:.4f}")
            print(f"    Completing {len(prev_trials)} trials from iteration {iteration-1}")

            # Tell Optuna about each trial's results
            for trial in prev_trials:
                self.study.tell(trial, trial_values)

            # Clean up completed trials
            del self.pending_trials[iteration - 1]

        # Stage 3: Step 2 - Ask for next batch of parameter sets
        n_sets = config.get('iteration_batch_size', 8)
        next_params = []
        trials = []

        print(f"\n  Suggesting {n_sets} parameter sets for iteration {iteration}...")

        for i in range(n_sets):
            trial = self.study.ask()
            params = self._define_search_space(trial)
            next_params.append(params)
            trials.append(trial)

        # Stage 3: Step 3 - Store pending trials for this iteration
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

        return next_params, converged
