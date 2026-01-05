"""
Experiment visualization and reporting utilities.

Generates visualizations for tracking optimization progress:
- Sample images from each iteration
- 2D projection plots (real vs synthetic over iterations)
- Metrics optimization plots
- Pareto front visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.decomposition import PCA
import json


class ExperimentReporter:
    """
    Generates visualizations and reports for optimization experiments.

    Saves outputs to experiment directory for analysis and debugging.
    """

    def __init__(self, experiment_dir: Path):
        """
        Initialize reporter.

        Args:
            experiment_dir: Path to experiment directory for saving outputs
        """
        self.experiment_dir = Path(experiment_dir)
        self.viz_dir = self.experiment_dir / "visualizations"
        self.samples_dir = self.experiment_dir / "sample_images"

        # Create directories
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        # Track metrics history
        self.metrics_history = []

    def save_sample_images(
        self,
        images: np.ndarray,
        iteration: int,
        label: str = "synthetic",
        n_samples: int = 6
    ):
        """
        Save sample images from an iteration.

        Args:
            images: (N, H, W, 3) array of images
            iteration: Iteration number
            label: Label for images ("synthetic", "real", etc.)
            n_samples: Number of samples to save
        """
        n_samples = min(n_samples, len(images))

        # Create grid of samples
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for i in range(n_samples):
            axes[i].imshow(images[i])
            axes[i].axis('off')
            axes[i].set_title(f"Sample {i+1}")

        # Hide unused subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"Iteration {iteration} - {label.title()} Samples", fontsize=14)
        plt.tight_layout()

        save_path = self.samples_dir / f"iter_{iteration:03d}_{label}_samples.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved {n_samples} sample images to {save_path}")

    def plot_2d_projection(
        self,
        real_embeddings: np.ndarray,
        synthetic_embeddings_by_iter: Dict[int, np.ndarray],
        initial_embeddings: Optional[np.ndarray] = None
    ):
        """
        Plot 2D projection of real vs synthetic embeddings over iterations.

        Args:
            real_embeddings: (M, D) real distribution embeddings
            synthetic_embeddings_by_iter: {iteration: (N, D) embeddings}
            initial_embeddings: Optional (N, D) initial synthetic embeddings (iteration 0)
        """
        # Combine all embeddings for PCA fitting
        all_embeddings = [real_embeddings]
        if initial_embeddings is not None:
            all_embeddings.append(initial_embeddings)
        all_embeddings.extend(synthetic_embeddings_by_iter.values())
        combined = np.vstack(all_embeddings)

        # Fit PCA for 2D projection
        pca_2d = PCA(n_components=2)
        pca_2d.fit(combined)

        # Project to 2D
        real_2d = pca_2d.transform(real_embeddings)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot real distribution
        ax.scatter(
            real_2d[:, 0], real_2d[:, 1],
            c='red', s=100, alpha=0.6, marker='*',
            label='Real', edgecolors='darkred', linewidth=0.5
        )

        # Plot initial synthetic (iteration 0) if provided
        if initial_embeddings is not None:
            initial_2d = pca_2d.transform(initial_embeddings)
            ax.scatter(
                initial_2d[:, 0], initial_2d[:, 1],
                c='lightgray', s=50, alpha=0.5,
                label='Iteration 0 (Initial)', edgecolors='gray', linewidth=0.5
            )

        # Plot synthetic from each iteration with color gradient
        iterations = sorted(synthetic_embeddings_by_iter.keys())
        cmap = plt.cm.viridis
        colors = [cmap(i / (len(iterations) - 1)) if len(iterations) > 1 else cmap(0.5)
                  for i in range(len(iterations))]

        for iteration, color in zip(iterations, colors):
            synthetic_2d = pca_2d.transform(synthetic_embeddings_by_iter[iteration])
            ax.scatter(
                synthetic_2d[:, 0], synthetic_2d[:, 1],
                c=[color], s=70, alpha=0.7,
                label=f'Iteration {iteration}', edgecolors='black', linewidth=0.3
            )

        ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        ax.set_title('Real vs Synthetic Distribution Evolution (2D Projection)', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.viz_dir / "distribution_evolution_2d.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved 2D projection plot to {save_path}")

    def update_metrics_history(self, iteration: int, metrics: Dict[str, float]):
        """
        Update metrics history for tracking optimization progress.

        Args:
            iteration: Iteration number
            metrics: Dictionary of metrics
        """
        self.metrics_history.append({
            'iteration': iteration,
            **metrics
        })

    def plot_metrics_optimization(self):
        """
        Plot metrics progression over iterations.

        Shows how the 3 optimization objectives evolve over time.
        """
        if len(self.metrics_history) == 0:
            print("  No metrics history to plot")
            return

        # Extract data
        iterations = [m['iteration'] for m in self.metrics_history]
        mmd_rbf = [m.get('mmd_rbf', np.nan) for m in self.metrics_history]
        wasserstein = [m.get('wasserstein', np.nan) for m in self.metrics_history]
        mean_nn_dist = [m.get('mean_nn_distance', np.nan) for m in self.metrics_history]

        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Plot MMD RBF
        axes[0].plot(iterations, mmd_rbf, marker='o', color='blue', linewidth=2, markersize=8)
        axes[0].set_ylabel('MMD RBF', fontsize=12)
        axes[0].set_title('Multi-Objective Optimization Progress', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(left=0)

        # Plot Wasserstein
        axes[1].plot(iterations, wasserstein, marker='s', color='green', linewidth=2, markersize=8)
        axes[1].set_ylabel('Wasserstein Distance', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(left=0)

        # Plot Mean NN Distance
        axes[2].plot(iterations, mean_nn_dist, marker='^', color='red', linewidth=2, markersize=8)
        axes[2].set_xlabel('Iteration', fontsize=12)
        axes[2].set_ylabel('Mean NN Distance', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(left=0)

        plt.tight_layout()
        save_path = self.viz_dir / "metrics_optimization.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved metrics optimization plot to {save_path}")

        # Also save metrics history as JSON
        json_path = self.viz_dir / "metrics_history.json"
        with open(json_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"  Saved metrics history to {json_path}")

    def plot_pareto_front(self, pareto_trials: List):
        """
        Visualize Pareto front with 2D projections of objective space.

        Args:
            pareto_trials: List of optuna.trial.FrozenTrial objects on Pareto front
        """
        if len(pareto_trials) == 0:
            print("  No Pareto front to plot")
            return

        # Extract objective values
        mmd_rbf = [trial.values[0] for trial in pareto_trials]
        wasserstein = [trial.values[1] for trial in pareto_trials]
        mean_nn_dist = [trial.values[2] for trial in pareto_trials]

        # Create 2D projections
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # MMD vs Wasserstein
        axes[0].scatter(mmd_rbf, wasserstein, s=100, alpha=0.7, c=range(len(pareto_trials)),
                       cmap='viridis', edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel('MMD RBF', fontsize=11)
        axes[0].set_ylabel('Wasserstein Distance', fontsize=11)
        axes[0].set_title('Pareto Front: MMD vs Wasserstein', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # MMD vs NN Distance
        axes[1].scatter(mmd_rbf, mean_nn_dist, s=100, alpha=0.7, c=range(len(pareto_trials)),
                       cmap='viridis', edgecolors='black', linewidth=0.5)
        axes[1].set_xlabel('MMD RBF', fontsize=11)
        axes[1].set_ylabel('Mean NN Distance', fontsize=11)
        axes[1].set_title('Pareto Front: MMD vs NN Distance', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        # Wasserstein vs NN Distance
        axes[2].scatter(wasserstein, mean_nn_dist, s=100, alpha=0.7, c=range(len(pareto_trials)),
                       cmap='viridis', edgecolors='black', linewidth=0.5)
        axes[2].set_xlabel('Wasserstein Distance', fontsize=11)
        axes[2].set_ylabel('Mean NN Distance', fontsize=11)
        axes[2].set_title('Pareto Front: Wasserstein vs NN Distance', fontsize=12)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.viz_dir / "pareto_front_2d_projections.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved Pareto front 2D projections to {save_path}")

    def generate_summary_report(self):
        """
        Generate text summary of experiment results.
        """
        if len(self.metrics_history) == 0:
            print("  No metrics history for summary")
            return

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("EXPERIMENT SUMMARY")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Basic stats
        report_lines.append(f"Total iterations: {len(self.metrics_history)}")
        report_lines.append("")

        # Initial vs final metrics
        initial = self.metrics_history[0]
        final = self.metrics_history[-1]

        report_lines.append("Metrics Evolution:")
        report_lines.append("-" * 60)

        for metric in ['mmd_rbf', 'wasserstein', 'mean_nn_distance']:
            if metric in initial and metric in final:
                init_val = initial[metric]
                final_val = final[metric]

                # Handle zero initial value (avoid division by zero)
                if init_val == 0:
                    if final_val == 0:
                        improvement_str = "0.00%"
                    else:
                        improvement_str = "N/A (initial was 0)"
                else:
                    improvement = ((init_val - final_val) / init_val) * 100
                    improvement_str = f"{improvement:+.2f}%"

                report_lines.append(f"{metric}:")
                report_lines.append(f"  Initial:  {init_val:.6f}")
                report_lines.append(f"  Final:    {final_val:.6f}")
                report_lines.append(f"  Change:   {improvement_str}")
                report_lines.append("")

        # Best metrics achieved
        report_lines.append("Best Values Achieved:")
        report_lines.append("-" * 60)

        for metric in ['mmd_rbf', 'wasserstein', 'mean_nn_distance']:
            values = [m.get(metric, np.inf) for m in self.metrics_history]
            best_val = min(values)
            best_iter = values.index(best_val)

            report_lines.append(f"{metric}:")
            report_lines.append(f"  Best value: {best_val:.6f} (iteration {best_iter})")
            report_lines.append("")

        report_lines.append("=" * 60)

        # Save report
        report_path = self.viz_dir / "experiment_summary.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"  Saved experiment summary to {report_path}")

        # Also print to console
        print("\n" + '\n'.join(report_lines))
