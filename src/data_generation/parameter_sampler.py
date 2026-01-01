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
        self.param_bounds = config['param_bounds']

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
                - brightness_factor: float
                - size_std: float
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
                'void_count': self._sample_integer(dist['void_count']),
                'base_size': self._sample_continuous(dist['base_size']),
                'brightness_factor': self._sample_continuous(dist['brightness_factor']),
                'size_std': self._sample_continuous(dist['size_std']),
                'position_spread': self._sample_continuous(dist['position_spread']),
            }
            param_sets.append(params)

        return param_sets

    def _sample_categorical(self, spec: Dict) -> str:
        """Sample from categorical distribution (e.g., void_shape)"""
        probabilities = spec['probabilities']
        categories = list(probabilities.keys())
        probs = list(probabilities.values())
        return np.random.choice(categories, p=probs)

    def _sample_integer(self, spec: Dict) -> int:
        """Sample integer from normal distribution, clipped to bounds"""
        value = np.random.normal(spec['mean'], spec['std'])

        # Apply bounds
        if 'min' in spec:
            value = max(value, spec['min'])
        if 'max' in spec:
            value = min(value, spec['max'])

        return int(round(value))

    def _sample_continuous(self, spec: Dict) -> float:
        """Sample continuous value from normal distribution, clipped to bounds"""
        value = np.random.normal(spec['mean'], spec['std'])

        # Apply bounds
        if 'min' in spec:
            value = max(value, spec['min'])
        if 'max' in spec:
            value = min(value, spec['max'])

        return float(value)
