"""
Parameter sampler for generating void parameter sets.

Samples from real/close/far distributions defined in config YAML.
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Literal, Optional


class ParameterSampler:
    """Sample void generation parameters from predefined distributions"""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize sampler with parameter distributions from config.

        Args:
            config_path: Path to experiment_config.yaml. If None, uses default location.
        """
        if config_path is None:
            # Default to configs/experiment_config.yaml relative to project root
            config_path = Path(__file__).parent.parent.parent / "configs" / "experiment_config.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.distributions = config['param_distributions']
        self.distribution_param_bounds = config['distribution_param_bounds']
        self.param_precision = config.get('param_precision', {})

    def sample_parameter_sets(
        self,
        distribution_type: Literal['real', 'close', 'far'],
        n_sets: int,
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Sample N parameter sets from specified distribution.

        Args:
            distribution_type: Which distribution to sample from ('real', 'close', 'far')
            n_sets: Number of parameter sets to generate
            seed: Random seed for reproducibility

        Returns:
            List of parameter dictionaries, each containing:
                - void_shape: str
                - void_count: int
                - base_size: float
                - rotation: float
                - center_x: float
                - center_y: float
                - position_spread: float
        """
        if seed is not None:
            np.random.seed(seed)

        if distribution_type not in self.distributions:
            raise ValueError(f"Unknown distribution type: {distribution_type}. Must be one of: {list(self.distributions.keys())}")

        dist = self.distributions[distribution_type]
        param_sets = []

        for i in range(n_sets):
            params = {
                'void_shape': self._sample_categorical(dist['void_shape']),
                'void_count': self._sample_integer(dist['void_count'], 'void_count'),
                'base_size': self._sample_continuous(dist['base_size'], 'base_size'),
                'rotation': self._sample_continuous(dist['rotation'], 'rotation'),
                'center_x': self._sample_continuous(dist['center_x'], 'center_x'),
                'center_y': self._sample_continuous(dist['center_y'], 'center_y'),
                'position_spread': self._sample_continuous(dist['position_spread'], 'position_spread'),
            }
            param_sets.append(params)

        return param_sets

    def sample_from_distribution_spec(
        self,
        dist_spec: Dict,
        n_samples: int,
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Sample N parameter sets from a given distribution specification.

        Args:
            dist_spec: Distribution specification dict with same structure as param_distributions
            n_samples: Number of parameter sets to sample
            seed: Random seed for reproducibility

        Returns:
            List of parameter dictionaries sampled from the distribution
        """
        if seed is not None:
            np.random.seed(seed)

        param_sets = []

        for i in range(n_samples):
            params = {
                'void_shape': self._sample_categorical(dist_spec['void_shape']),
                'void_count': self._sample_integer(dist_spec['void_count'], 'void_count'),
                'base_size': self._sample_continuous(dist_spec['base_size'], 'base_size'),
                'rotation': self._sample_continuous(dist_spec['rotation'], 'rotation'),
                'center_x': self._sample_continuous(dist_spec['center_x'], 'center_x'),
                'center_y': self._sample_continuous(dist_spec['center_y'], 'center_y'),
                'position_spread': self._sample_continuous(dist_spec['position_spread'], 'position_spread'),
            }
            param_sets.append(params)

        return param_sets

    def _sample_categorical(self, spec: Dict) -> str:
        """Sample from categorical distribution (e.g., void_shape)"""
        probabilities = spec['probabilities']
        categories = list(probabilities.keys())
        probs = list(probabilities.values())
        return np.random.choice(categories, p=probs)

    def _sample_integer(self, spec: Dict, param_name: str) -> int:
        """Sample integer from normal distribution, clipped to distribution bounds"""
        value = np.random.normal(spec['mean'], spec['std'])

        # Apply bounds from distribution_param_bounds (using mean bounds as parameter range)
        mean_bounds = self.distribution_param_bounds[param_name]['mean']
        value = max(value, mean_bounds[0])
        value = min(value, mean_bounds[1])

        return int(round(value))

    def _sample_continuous(self, spec: Dict, param_name: str) -> float:
        """Sample continuous value from normal distribution, clipped to distribution bounds"""
        value = np.random.normal(spec['mean'], spec['std'])

        # Apply bounds from distribution_param_bounds (using mean bounds as parameter range)
        mean_bounds = self.distribution_param_bounds[param_name]['mean']
        value = max(value, mean_bounds[0])
        value = min(value, mean_bounds[1])

        # Apply precision rounding if specified
        if param_name in self.param_precision:
            precision = self.param_precision[param_name]
            value = round(value, precision)

        return float(value)
