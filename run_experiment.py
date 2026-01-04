#!/usr/bin/env python3
"""
Main entry point for running optimization experiments.

Usage:
    python run_experiment.py [--config path/to/config.yaml]
"""

import argparse
from pathlib import Path
from src.orchestration import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description='Run synthetic data optimization experiment')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment_config.yaml',
        help='Path to experiment configuration file (default: configs/experiment_config.yaml)'
    )
    args = parser.parse_args()

    # Load and run experiment
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    print(f"Loading config from: {config_path}")

    runner = ExperimentRunner(config_path)
    summary = runner.run()

    print(f"\nExperiment summary saved to: {runner.iteration_manager.experiment_dir / 'summary.json'}")

    return 0


if __name__ == '__main__':
    exit(main())
