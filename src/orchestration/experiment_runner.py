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
            current_distributions: What was evaluated [(dist_id, params_dict), ...]
                                   params_dict contains shape_logit_* and {shape}__{param} keys
            synthetic_embeddings_400d: Embeddings from external source (400D)
            synthetic_metadata: Metadata with distribution_id for grouping

        Returns:
            next_suggestions: What to try next [(dist_id, params_dict), ...]
        """
        if self.real_embeddings_400d is None:
            raise ValueError("Real embeddings not set. Call set_real_embeddings() first.")

        n_distributions = len(current_distributions)

        # Compute per-distribution metrics
        metrics_list = compute_per_param_set_metrics(
            synthetic_embeddings_400d,
            synthetic_metadata,
            self.real_embeddings_400d,
            n_param_sets=n_distributions
        )

        # Tell optimizer and get next suggestions
        next_suggestions = self.optimizer.suggest_next_distributions(
            current_distributions=current_distributions,
            metrics_list=metrics_list,
            config=self.config
        )

        return next_suggestions

