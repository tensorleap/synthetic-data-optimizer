"""
Optuna-based Bayesian optimizer for synthetic data parameter optimization.

Stage 1: Single-objective optimization (minimize mmd_rbf)
"""

import optuna
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class OptunaOptimizer:
    """
    Optuna-based optimizer using TPE (Tree-structured Parzen Estimator) sampler.

    Stage 1 features:
    - Single-objective optimization (minimize mmd_rbf)
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

        # Stage 1: Single-objective optimization
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction='minimize',  # Minimize mmd_rbf
            sampler=optuna.samplers.TPESampler(
                seed=config.get('random_seed', 42)
            )
        )

        print(f"Initialized OptunaOptimizer")
        print(f"  Study storage: {self.study_path}")
        print(f"  Study name: {study_name}")
        print(f"  Direction: minimize (single-objective)")

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

        Stage 1: Simple ask/tell pattern with single-objective optimization.

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
        # Stage 1: Report results from previous iteration if exists
        # For now, we report the single objective value (mmd_rbf)
        if iteration > 0 and len(synthetic_params) > 0:
            # In Stage 1, we use simple approach: report mmd_rbf as trial value
            # All parameter sets from previous iteration share the same metrics
            trial_value = metrics['mmd_rbf']

            # Note: In Stage 1, we're not properly tracking trials yet
            # This will be fixed in Stage 3 with proper ask/tell pattern
            print(f"  Iteration {iteration-1} result: mmd_rbf = {trial_value:.4f}")

        # Ask Optuna for next parameter sets
        n_sets = config.get('iteration_batch_size', 8)
        next_params = []

        print(f"\n  Suggesting {n_sets} parameter sets for iteration {iteration}...")

        for i in range(n_sets):
            trial = self.study.ask()
            params = self._define_search_space(trial)
            next_params.append(params)

            # For Stage 1, we immediately tell with a dummy value
            # This will be properly fixed in Stage 3
            if iteration > 0:
                self.study.tell(trial, metrics.get('mmd_rbf', float('inf')))

        # Simple convergence check: max iterations reached
        max_iterations = config.get('max_iterations', 10)
        converged = iteration >= max_iterations

        if converged:
            print(f"  Convergence: Reached max iterations ({max_iterations})")

        return next_params, converged
