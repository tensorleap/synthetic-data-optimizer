"""
Generate test data for production experiment script testing.

This script creates mock data files that can be used to test run_experiment.py.
Run this offline before testing the production script.

Usage:
    python -m scripts.generate_test_data --output-dir data/test_experiment

The script generates:
    - real_embeddings.npy: Mock real embeddings (400D)
    - iter_000/embeddings.npy: Mock synthetic embeddings for iteration 0
    - iter_000/distributions.json: Distribution parameters for iteration 0
    - iter_000/metadata.json: Sample metadata for iteration 0
    - bounds/: Directory with per-group parameter CSVs for bounds inference
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from src.data_generation.mock_data_generator import MockDataGenerator


def generate_test_data(
    output_dir: Path,
    n_real_samples: int = 100,
    n_distributions: int = 8,
    n_samples_per_distribution: int = 10,
    embedding_dim: int = 400,
    seed: int = 42
):
    """
    Generate test data for production experiment testing.

    Args:
        output_dir: Directory to save generated data
        n_real_samples: Number of real embeddings to generate
        n_distributions: Number of distributions per group (for bounds inference)
        n_samples_per_distribution: Samples per distribution
        embedding_dim: Embedding dimension
        seed: Random seed
    """
    print(f"\nGenerating test data in: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize mock data generator
    generator = MockDataGenerator(use_real_embeddings=False)

    # Generate mock data
    real_embeddings, synthetic_embeddings_groups, synthetic_params_groups = \
        generator.generate_conditional_groups(
            n_real_samples=n_real_samples,
            n_distributions_per_group=n_distributions,
            n_samples_per_distribution=n_samples_per_distribution,
            embedding_dim=embedding_dim,
            seed=seed
        )

    group_names = list(MockDataGenerator.DEFAULT_GROUP_SPECS.keys())

    # Save real embeddings
    real_path = output_dir / "real_embeddings.npy"
    np.save(real_path, real_embeddings)
    print(f"Saved real embeddings: {real_path} - shape {real_embeddings.shape}")

    # Save bounds data (per-group CSVs for bounds inference)
    # Round all values to 2 decimal places for consistency
    bounds_dir = output_dir / "bounds"
    bounds_dir.mkdir(exist_ok=True)

    for group_name, params_df in zip(group_names, synthetic_params_groups):
        # Round all numeric columns to 2 decimal places
        rounded_df = params_df.round(2)
        csv_path = bounds_dir / f"{group_name}_params.csv"
        rounded_df.to_csv(csv_path, index=False)
        print(f"Saved bounds data: {csv_path} - {len(rounded_df)} rows")

    # Generate iteration 0 data
    generate_iteration_data(
        output_dir=output_dir,
        iteration=0,
        synthetic_embeddings_groups=synthetic_embeddings_groups,
        synthetic_params_groups=synthetic_params_groups,
        group_names=group_names,
        n_samples_per_distribution=n_samples_per_distribution
    )

    print(f"\nTest data generation complete!")
    print(f"Run production experiment with: python -m scripts.run_experiment --data-dir {output_dir} --iteration 0")


def generate_iteration_data(
    output_dir: Path,
    iteration: int,
    synthetic_embeddings_groups: List[np.ndarray],
    synthetic_params_groups: List[pd.DataFrame],
    group_names: List[str],
    n_samples_per_distribution: int
):
    """
    Generate data for a specific iteration.

    Args:
        output_dir: Base output directory
        iteration: Iteration number
        synthetic_embeddings_groups: List of embedding arrays per group
        synthetic_params_groups: List of parameter DataFrames per group
        group_names: List of group names
        n_samples_per_distribution: Samples per distribution
    """
    iter_dir = output_dir / f"iter_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Combine embeddings from all groups
    all_embeddings = np.vstack(synthetic_embeddings_groups)
    embeddings_path = iter_dir / "embeddings.npy"
    np.save(embeddings_path, all_embeddings)
    print(f"Saved embeddings: {embeddings_path} - shape {all_embeddings.shape}")

    # Create distributions.json - unique distributions from params
    distributions = []
    metadata = []
    sample_idx = 0

    for group_idx, (group_name, params_df) in enumerate(zip(group_names, synthetic_params_groups)):
        # Round params and get unique distributions
        rounded_df = params_df.round(2)
        unique_params = rounded_df.drop_duplicates()

        for dist_local_idx, (_, row) in enumerate(unique_params.iterrows()):
            dist_id = len(distributions)
            params = row.to_dict()

            distributions.append({
                'distribution_id': dist_id,
                'group': group_name,
                'params': params
            })

            # Create metadata for samples from this distribution
            for _ in range(n_samples_per_distribution):
                metadata.append({
                    'sample_idx': sample_idx,
                    'distribution_id': dist_id,
                    'group': group_name
                })
                sample_idx += 1

    # Save distributions
    distributions_path = iter_dir / "distributions.json"
    with open(distributions_path, 'w') as f:
        json.dump(distributions, f, indent=2)
    print(f"Saved distributions: {distributions_path} - {len(distributions)} distributions")

    # Save metadata
    metadata_path = iter_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path} - {len(metadata)} samples")


def main():
    parser = argparse.ArgumentParser(description='Generate test data for production experiment')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/test_experiment',
        help='Directory to save generated test data'
    )
    parser.add_argument(
        '--n-real',
        type=int,
        default=100,
        help='Number of real embeddings to generate'
    )
    parser.add_argument(
        '--n-distributions',
        type=int,
        default=8,
        help='Number of distributions per group'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help='Number of samples per distribution'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    args = parser.parse_args()

    generate_test_data(
        output_dir=Path(args.output_dir),
        n_real_samples=args.n_real,
        n_distributions=args.n_distributions,
        n_samples_per_distribution=args.n_samples,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
