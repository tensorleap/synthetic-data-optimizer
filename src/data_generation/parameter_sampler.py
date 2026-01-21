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

            if 'irregularity_pattern' in dist:
                params['irregularity_pattern'] = self._sample_categorical(dist['irregularity_pattern'])

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

            if 'irregularity_pattern' in dist_spec:
                params['irregularity_pattern'] = self._sample_categorical(dist_spec['irregularity_pattern'])

            param_sets.append(params)

        return param_sets

    @staticmethod
    def grouped_to_nested_dist_spec(group_name: str, params: Dict) -> Dict:
        """
        Convert grouped optimizer output to nested distribution specification.

        DEPRECATED: Use joint_to_nested_dist_spec() for joint optimization format.

        Converts grouped format (group_name, params) where:
            group_name: 'circle', 'ellipse', or 'irregular'
            params: {'void_count_min': 5, 'void_count_max': 10, ...}

        To nested format like:
            {'void_shape': {'probabilities': {'circle': 1.0}},
             'void_count': {'min': 5, 'max': 10}, ...}

        Args:
            group_name: The shape group (becomes void_shape with probability 1.0)
            params: Flat parameter dict with _min/_max suffixes

        Returns:
            Nested distribution specification compatible with sample_from_distribution_spec
        """
        nested = {}

        # Group name becomes void_shape with probability 1.0
        nested['void_shape'] = {
            'probabilities': {group_name: 1.0}
        }

        # Handle continuous/integer parameters with min/max suffixes
        # Note: rotation only exists for ellipse, others get default
        param_bases = ['void_count', 'base_size', 'rotation', 'center_x', 'center_y', 'position_spread']

        for param_base in param_bases:
            min_key = f'{param_base}_min'
            max_key = f'{param_base}_max'

            if min_key in params and max_key in params:
                nested[param_base] = {
                    'min': params[min_key],
                    'max': params[max_key]
                }
            elif param_base == 'rotation':
                # Default rotation for non-ellipse shapes (circles/irregular are rotation-invariant)
                nested['rotation'] = {'min': 0.0, 'max': 0.0}

        return nested

    @staticmethod
    def joint_to_per_shape_dist_specs(
        joint_params: Dict,
        group_names: List[str]
    ) -> Dict[str, Dict]:
        """
        Convert joint optimizer format to per-shape distribution specs.

        Joint format from optimizer:
            {
                'shape_logit_circle': 0.5,
                'shape_logit_ellipse': 0.3,
                'circle__void_count_min': 5,
                'circle__void_count_max': 10,
                'ellipse__void_count_min': 4,
                ...
            }

        Converts to per-shape specs:
            {
                'circle': {'void_count': {'min': 5, 'max': 10}, ...},
                'ellipse': {'void_count': {'min': 4, 'max': ...}, ...}
            }

        Args:
            joint_params: Joint optimizer output dict
            group_names: List of shape names

        Returns:
            Dict mapping shape names to nested distribution specs
        """
        per_shape_specs = {}
        param_bases = ['void_count', 'base_size', 'rotation', 'center_x', 'center_y', 'position_spread']

        for group_name in group_names:
            spec = {}

            for param_base in param_bases:
                min_key = f'{group_name}__{param_base}_min'
                max_key = f'{group_name}__{param_base}_max'

                if min_key in joint_params and max_key in joint_params:
                    spec[param_base] = {
                        'min': joint_params[min_key],
                        'max': joint_params[max_key]
                    }
                elif param_base == 'rotation' and group_name != 'ellipse':
                    # Default rotation for non-ellipse shapes
                    spec['rotation'] = {'min': 0.0, 'max': 0.0}

            per_shape_specs[group_name] = spec

        return per_shape_specs

    def sample_from_joint_distribution(
        self,
        joint_params: Dict,
        group_names: List[str],
        n_samples: int,
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Sample concrete parameters from joint optimizer output.

        Uses softmax on shape logits to get probabilities, then samples shapes
        according to those probabilities. For each sampled shape, samples params
        from that shape's distribution.

        Args:
            joint_params: Joint optimizer output dict with:
                         - shape_logit_* keys
                         - {shape}__{param}_mean and {shape}__{param}_std keys
            group_names: List of shape names
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility

        Returns:
            List of concrete parameter dicts ready for VoidGenerator
        """
        import math

        if seed is not None:
            np.random.seed(seed)

        # Convert logits to probabilities via softmax
        logits = {}
        for g in group_names:
            logit_key = f'shape_logit_{g}'
            logits[g] = joint_params.get(logit_key, 0.0)

        # Softmax with numerical stability
        max_logit = max(logits.values())
        exp_logits = {k: math.exp(v - max_logit) for k, v in logits.items()}
        total = sum(exp_logits.values())
        shape_probs = {k: v / total for k, v in exp_logits.items()}

        # Get per-shape distribution specs
        per_shape_specs = self.joint_to_per_shape_dist_specs(joint_params, group_names)

        # Sample
        samples = []
        for _ in range(n_samples):
            # Sample shape according to probabilities
            shape = np.random.choice(
                list(shape_probs.keys()),
                p=list(shape_probs.values())
            )

            # Get spec for this shape
            spec = per_shape_specs[shape]

            # Sample concrete params
            params = {'void_shape': shape}

            for param_base in ['void_count', 'base_size', 'rotation', 'center_x', 'center_y', 'position_spread']:
                if param_base in spec:
                    min_val = spec[param_base]['min']
                    max_val = spec[param_base]['max']

                    if min_val > max_val:
                        raise ValueError(f"Parameter '{param_base}': min ({min_val}) > max ({max_val})")

                    if param_base == 'void_count':
                        value = int(np.random.randint(min_val, max_val + 1))
                    else:
                        value = np.random.uniform(min_val, max_val)

                        # Apply precision rounding if specified
                        if param_base in self.param_precision:
                            value = round(value, self.param_precision[param_base])

                        value = float(value)

                    params[param_base] = value

            samples.append(params)

        return samples

    def _sample_categorical(self, spec: Dict) -> str:
        """Sample from categorical distribution (e.g., void_shape)"""
        probabilities = spec['probabilities']
        categories = list(probabilities.keys())
        probs = list(probabilities.values())
        return np.random.choice(categories, p=probs)

    def _sample_integer(self, spec: Dict, param_name: str) -> int:
        """Sample integer uniformly from [min, max] range"""
        min_val = spec['min']
        max_val = spec['max']

        if min_val > max_val:
            raise ValueError(f"Parameter '{param_name}': min ({min_val}) > max ({max_val})")

        return int(np.random.randint(min_val, max_val + 1))

    def _sample_continuous(self, spec: Dict, param_name: str) -> float:
        """Sample continuous value uniformly from [min, max] range"""
        min_val = spec['min']
        max_val = spec['max']

        if min_val > max_val:
            raise ValueError(f"Parameter '{param_name}': min ({min_val}) > max ({max_val})")

        value = np.random.uniform(min_val, max_val)

        # Apply precision rounding if specified
        if param_name in self.param_precision:
            precision = self.param_precision[param_name]
            value = round(value, precision)

        return float(value)
