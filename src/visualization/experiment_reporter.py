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

    def __init__(self, experiment_dir: Path, config: Dict = None):
        """
        Initialize reporter.

        Args:
            experiment_dir: Path to experiment directory for saving outputs
            config: Optional experiment config dict (for optimization_metrics)
        """
        self.experiment_dir = Path(experiment_dir)
        self.viz_dir = self.experiment_dir / "visualizations"
        self.samples_dir = self.experiment_dir / "sample_images"

        # Get optimization metrics from config
        if config:
            self.optimization_metrics = config.get('optimization_metrics', ['mmd_rbf', 'mean_nn_distance'])
        else:
            # Default fallback if no config provided
            self.optimization_metrics = ['mmd_rbf', 'mean_nn_distance']

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

        # Plot synthetic from first and last iteration only
        iterations = sorted(synthetic_embeddings_by_iter.keys())

        if len(iterations) > 0:
            # First iteration (iteration 0)
            first_iter = iterations[0]
            synthetic_first_2d = pca_2d.transform(synthetic_embeddings_by_iter[first_iter])
            ax.scatter(
                synthetic_first_2d[:, 0], synthetic_first_2d[:, 1],
                c='orange', s=70, alpha=0.7,
                label=f'Iteration {first_iter}', edgecolors='black', linewidth=0.3
            )

            # Last iteration (if different from first)
            if len(iterations) > 1:
                last_iter = iterations[-1]
                synthetic_last_2d = pca_2d.transform(synthetic_embeddings_by_iter[last_iter])
                ax.scatter(
                    synthetic_last_2d[:, 0], synthetic_last_2d[:, 1],
                    c='green', s=70, alpha=0.7,
                    label=f'Iteration {last_iter}', edgecolors='black', linewidth=0.3
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

        # Get number of objectives from configured metrics
        n_metrics = len(self.optimization_metrics)

        if n_metrics < 2:
            print("  Need at least 2 objectives to plot Pareto front")
            return

        # Extract objective values dynamically
        objective_values = {}
        for i, metric_name in enumerate(self.optimization_metrics):
            objective_values[metric_name] = [trial.values[i] for trial in pareto_trials]

        # Create pairwise 2D projections
        # Number of subplot pairs: n_metrics choose 2
        n_plots = (n_metrics * (n_metrics - 1)) // 2

        if n_plots == 1:
            fig, axes = plt.subplots(1, 1, figsize=(6, 5))
            axes = [axes]
        else:
            fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
            if n_plots == 1:
                axes = [axes]

        plot_idx = 0
        metric_names = self.optimization_metrics

        # Create all pairwise plots
        for i in range(n_metrics):
            for j in range(i + 1, n_metrics):
                x_metric = metric_names[i]
                y_metric = metric_names[j]

                axes[plot_idx].scatter(
                    objective_values[x_metric],
                    objective_values[y_metric],
                    s=100, alpha=0.7, c=range(len(pareto_trials)),
                    cmap='viridis', edgecolors='black', linewidth=0.5
                )
                axes[plot_idx].set_xlabel(x_metric.replace('_', ' ').title(), fontsize=11)
                axes[plot_idx].set_ylabel(y_metric.replace('_', ' ').title(), fontsize=11)
                axes[plot_idx].set_title(f'Pareto Front: {x_metric} vs {y_metric}', fontsize=12)
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1

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

        for metric in self.optimization_metrics:
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

        for metric in self.optimization_metrics:
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
