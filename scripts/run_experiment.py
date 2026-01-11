"""
Production experiment runner - single iteration.

This script runs a single optimization iteration with data loaded from disk.
Called by external system after generating data for the current distributions.

Usage:
    python -m scripts.run_experiment \
        --config configs/experiment_config.yaml \
        --data-dir data/experiment_data \
        --iteration 0

The script expects:
    - data_dir/real_embeddings.npy: Real embeddings (400D)
    - data_dir/iter_{N}/embeddings.npy: Synthetic embeddings for iteration N
    - data_dir/iter_{N}/distributions.json: Distributions used for iteration N
    - data_dir/iter_{N}/metadata.json: Metadata for iteration N

After running, it saves:
    - data_dir/iter_{N}/suggestions.json: Next distributions to try
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

from src.orchestration.experiment_runner import ExperimentRunner


def load_distributions(path: Path) -> List[Tuple[str, Dict]]:
    """Load distributions from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    distributions = []
    for item in data:
        group = item['group']
        params = item['params']
        distributions.append((group, params))

    return distributions


def load_metadata(path: Path) -> List[Dict]:
    """Load metadata from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_suggestions(suggestions: List[Tuple[str, Dict]], path: Path):
    """Save suggestions to JSON file."""
    data = []
    for idx, (group, params) in enumerate(suggestions):
        data.append({
            'distribution_id': idx,
            'group': group,
            'params': params
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(suggestions)} suggestions to {path}")


def run_iteration(
    config_path: Path,
    data_dir: Path,
    iteration: int
):
    """
    Run a single optimization iteration with disk data.

    Args:
        config_path: Path to experiment config YAML
        data_dir: Directory containing experiment data
        iteration: Current iteration number
    """
    print("\n" + "=" * 60)
    print(f"PRODUCTION EXPERIMENT - Iteration {iteration}")
    print("=" * 60)

    # Initialize experiment runner
    runner = ExperimentRunner(config_path)

    # Load real embeddings
    real_path = data_dir / "real_embeddings.npy"
    if not real_path.exists():
        raise FileNotFoundError(f"Real embeddings not found: {real_path}")

    real_embeddings_400d = np.load(real_path)
    runner.set_real_embeddings(real_embeddings_400d)

    # Load iteration data
    iter_dir = data_dir / f"iter_{iteration:03d}"

    embeddings_path = iter_dir / "embeddings.npy"
    distributions_path = iter_dir / "distributions.json"
    metadata_path = iter_dir / "metadata.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    if not distributions_path.exists():
        raise FileNotFoundError(f"Distributions not found: {distributions_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    embeddings_400d = np.load(embeddings_path)
    current_distributions = load_distributions(distributions_path)
    metadata = load_metadata(metadata_path)

    print(f"Loaded {len(current_distributions)} distributions with {len(embeddings_400d)} embeddings")

    # Run iteration
    next_suggestions = runner.run_iteration(
        iteration=iteration,
        current_distributions=current_distributions,
        synthetic_embeddings_400d=embeddings_400d,
        synthetic_metadata=metadata
    )

    # Save suggestions for external system
    suggestions_path = iter_dir / "suggestions.json"
    save_suggestions(next_suggestions, suggestions_path)

    print("\n" + "=" * 60)
    print(f"ITERATION {iteration} COMPLETE")
    print("=" * 60)
    print(f"Next suggestions saved to: {suggestions_path}")
    print("External system should generate data for these distributions")


def main():
    parser = argparse.ArgumentParser(description='Run single iteration with disk data')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment_config.yaml',
        help='Path to experiment config YAML'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing experiment data'
    )
    parser.add_argument(
        '--iteration',
        type=int,
        required=True,
        help='Current iteration number'
    )
    args = parser.parse_args()

    run_iteration(
        config_path=Path(args.config),
        data_dir=Path(args.data_dir),
        iteration=args.iteration
    )


if __name__ == '__main__':
    main()
