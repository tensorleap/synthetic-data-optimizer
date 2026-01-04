"""
Iteration Manager for tracking and persisting iteration state.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import pickle


class IterationManager:
    """Manages iteration state and data persistence"""

    def __init__(self, experiment_dir: Path):
        """
        Initialize iteration manager.

        Args:
            experiment_dir: Directory to save iteration results
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.iterations_dir = self.experiment_dir / "iterations"
        self.iterations_dir.mkdir(exist_ok=True)

        self.images_dir = self.experiment_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        self.models_dir = self.experiment_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

    def save_iteration(
        self,
        iteration: int,
        params: List[Dict],
        embeddings: np.ndarray,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save iteration data to disk.

        Args:
            iteration: Iteration number
            params: Parameter sets used
            embeddings: Generated embeddings (N, 400)
            metrics: Computed metrics
            metadata: Optional metadata dict
        """
        iter_dir = self.iterations_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(exist_ok=True)

        # Save parameters as JSON
        params_path = iter_dir / "params.json"
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)

        # Save embeddings as numpy
        embeddings_path = iter_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)

        # Save metrics as JSON
        metrics_path = iter_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save metadata if provided
        if metadata is not None:
            metadata_path = iter_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        print(f"Saved iteration {iteration} data to {iter_dir}")

    def load_iteration(self, iteration: int) -> Dict:
        """
        Load iteration data from disk.

        Args:
            iteration: Iteration number to load

        Returns:
            Dictionary with keys: params, embeddings, metrics, metadata
        """
        iter_dir = self.iterations_dir / f"iter_{iteration:03d}"

        if not iter_dir.exists():
            raise FileNotFoundError(f"Iteration {iteration} not found at {iter_dir}")

        # Load parameters
        with open(iter_dir / "params.json", 'r') as f:
            params = json.load(f)

        # Load embeddings
        embeddings = np.load(iter_dir / "embeddings.npy")

        # Load metrics
        with open(iter_dir / "metrics.json", 'r') as f:
            metrics = json.load(f)

        # Load metadata if exists
        metadata_path = iter_dir / "metadata.json"
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        return {
            'params': params,
            'embeddings': embeddings,
            'metrics': metrics,
            'metadata': metadata
        }

    def save_summary(self, summary: Dict) -> None:
        """
        Save experiment summary.

        Args:
            summary: Summary dictionary with all iterations' results
        """
        summary_path = self.experiment_dir / "summary.json"
        with open(summary_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_summary = {}
            for key, value in summary.items():
                if isinstance(value, np.ndarray):
                    serializable_summary[key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    serializable_summary[key] = [v.tolist() for v in value]
                else:
                    serializable_summary[key] = value

            json.dump(serializable_summary, f, indent=2)

        print(f"Saved experiment summary to {summary_path}")

    def get_latest_iteration(self) -> int:
        """
        Get the latest iteration number.

        Returns:
            Latest iteration number, or -1 if no iterations exist
        """
        iter_dirs = sorted(self.iterations_dir.glob("iter_*"))
        if not iter_dirs:
            return -1

        latest_dir = iter_dirs[-1]
        iter_num = int(latest_dir.name.split("_")[1])
        return iter_num
