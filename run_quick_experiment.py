"""
Quick 2-iteration experiment to generate visualizations.
"""

from pathlib import Path
from src.orchestration.experiment_runner import ExperimentRunner

# Use existing config but override max_iterations
config_path = Path("configs/experiment_config.yaml")

# Run experiment
runner = ExperimentRunner(config_path)

# Override max iterations to just 2
runner.config['max_iterations'] = 2

# Run
print("\n" + "=" * 80)
print("RUNNING QUICK 2-ITERATION EXPERIMENT FOR VISUALIZATION")
print("=" * 80)

summary = runner.run()

print("\n" + "=" * 80)
print("VISUALIZATION OUTPUTS:")
print("=" * 80)
print(f"Experiment directory: {runner.config['experiment_dir']}")
print(f"Sample images: {runner.reporter.samples_dir}/")
print(f"Visualizations: {runner.reporter.viz_dir}/")
print("\nGenerated files:")
print("  - sample_images/iter_000_real_samples.png")
print("  - sample_images/iter_000_synthetic_samples.png")
print("  - sample_images/iter_001_synthetic_samples.png")
print("  - sample_images/iter_002_synthetic_samples.png")
print("  - visualizations/distribution_evolution_2d.png")
print("  - visualizations/metrics_optimization.png")
print("  - visualizations/metrics_history.json")
print("  - visualizations/experiment_summary.txt")
print("=" * 80)
