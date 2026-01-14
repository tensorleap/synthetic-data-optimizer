"""
Production configuration module.

All configuration values are hardcoded (no YAML dependency).
Based on configs/experiment_config.yaml.
"""

from typing import Dict

# Default experiment configuration
DEFAULT_CONFIG: Dict = {
    'experiment_name': 'void_optimization',
    'experiments_base_dir': 'data/experiments',
    'random_seed': 42,
    'iteration_batch_size': 10,
    'replications_per_iteration': 20,
    'max_iterations': 10,
    'param_precision': {
        'base_size': 1,
        'rotation': 1,
        'center_x': 2,
        'center_y': 2,
        'position_spread': 2,
    },
    'optimization_metrics': ['mmd_rbf'],
    'convergence_threshold': 0.05,
    'early_stop_patience': 3,
    'optimizer': {
        'n_startup_trials': 60,
        'multivariate': True,
    },
    'top_n_distributions': 10,
}
