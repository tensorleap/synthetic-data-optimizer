"""
End-to-end integration test for OptunaOptimizer with full pipeline.

Tests the optimizer with actual:
- VoidGenerator
- DinoV2Embedder
- PCA projection
- Metrics computation
- ExperimentRunner integration
"""

import pytest
import optuna
import yaml
import tempfile
import shutil
from pathlib import Path

from src.optimization.optuna_optimizer import OptunaOptimizer
from src.data_generation.void_generator import VoidGenerator
from src.data_generation.parameter_sampler import ParameterSampler
from src.embedding.dinov2_embedder import DinoV2Embedder
from src.embedding.pca_projector import PCAProjector
from src.optimization.metrics import compute_all_metrics
from src.visualization.experiment_reporter import ExperimentReporter


@pytest.fixture
def temp_test_dir():
    """Create temporary test directory"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mini_config(temp_test_dir):
    """Minimal config for fast end-to-end test"""
    return {
        'experiment_name': 'test_e2e_optuna',
        'experiment_dir': str(temp_test_dir / 'experiment'),
        'base_image_dir': 'data/base_chips',
        'dino_model': 'dinov2_vits14',  # Smallest DinoV2 model for speed
        'pca_embedding_dim': 50,  # Reduced for speed
        'random_seed': 42,
        'iteration_batch_size': 2,  # Small batch for speed
        'replications_per_iteration': 1,  # Single replication for speed
        'max_iterations': 3,
        'param_bounds': {
            'void_shape': ['circle', 'ellipse', 'irregular'],
            'void_count': [1, 10],
            'base_size': [5.0, 15.0],
            'rotation': [0.0, 360.0],
            'center_x': [0.2, 0.8],
            'center_y': [0.2, 0.8],
            'position_spread': [0.1, 0.8]
        },
        'param_precision': {
            'base_size': 1,
            'rotation': 1,
            'center_x': 2,
            'center_y': 2,
            'position_spread': 2
        },
        'optimizer': {
            'n_startup_trials': 2,
            'multivariate': True
        }
    }


class TestOptunaE2EIntegration:
    """End-to-end integration tests with full pipeline"""

    def test_optimizer_with_real_components(self, mini_config, temp_test_dir):
        """Test OptunaOptimizer with real generator, embedder, and metrics"""
        # Initialize components
        experiment_dir = Path(mini_config['experiment_dir'])
        experiment_dir.mkdir(parents=True, exist_ok=True)

        optimizer = OptunaOptimizer(experiment_dir, mini_config)
        sampler = ParameterSampler()
        generator = VoidGenerator(Path(mini_config['base_image_dir']))
        embedder = DinoV2Embedder(model_name=mini_config['dino_model'])

        # Generate "real" distribution (small sample)
        print("\n[Test] Generating real distribution...")
        real_params = sampler.sample_parameter_sets('real', n_sets=2, seed=42)
        real_images, _ = generator.generate_batch(real_params, replications=1, seed_offset=0)

        # Extract embeddings and project to lower dimension
        print("[Test] Extracting embeddings...")
        real_embeddings_full = embedder.embed_batch(real_images)

        # Fit PCA
        pca = PCAProjector(n_components=mini_config['pca_embedding_dim'])
        real_embeddings = pca.fit_transform(real_embeddings_full)

        print(f"[Test] Real embeddings shape: {real_embeddings.shape}")

        # Run 2 optimization iterations
        iteration_results = []

        for iteration in range(1, 3):
            print(f"\n[Test] === Iteration {iteration} ===")

            # Get previous iteration data (or empty for iteration 1)
            if iteration == 1:
                prev_embeddings = real_embeddings  # Use real as baseline
                prev_params = real_params
                prev_metrics = compute_all_metrics(real_embeddings, real_embeddings)
            else:
                prev_embeddings = iteration_results[-1]['embeddings']
                prev_params = iteration_results[-1]['params']
                prev_metrics = iteration_results[-1]['metrics']

            # Ask optimizer for next parameters
            next_params, converged = optimizer.suggest_next_parameters(
                synthetic_embeddings=prev_embeddings,
                synthetic_params=prev_params,
                real_embeddings=real_embeddings,
                metrics=prev_metrics,
                iteration=iteration,
                config=mini_config
            )

            # Generate synthetic images with suggested parameters
            print(f"[Test] Generating {len(next_params)} synthetic samples...")
            synthetic_images, _ = generator.generate_batch(
                next_params,
                replications=mini_config['replications_per_iteration'],
                seed_offset=iteration * 1000
            )

            # Extract and project embeddings
            synthetic_embeddings_full = embedder.embed_batch(synthetic_images)
            synthetic_embeddings = pca.transform(synthetic_embeddings_full)

            # Compute metrics
            metrics = compute_all_metrics(synthetic_embeddings, real_embeddings)

            print(f"[Test] Metrics: mmd_rbf={metrics['mmd_rbf']:.4f}, "
                  f"wasserstein={metrics['wasserstein']:.4f}, "
                  f"mean_nn_distance={metrics['mean_nn_distance']:.4f}")

            # Store results
            iteration_results.append({
                'iteration': iteration,
                'params': next_params,
                'embeddings': synthetic_embeddings,
                'metrics': metrics,
                'converged': converged
            })

        # Verify basic integration
        assert len(iteration_results) == 2, "Should complete 2 iterations"

        # Verify optimizer produced valid parameters
        for result in iteration_results:
            assert len(result['params']) == mini_config['iteration_batch_size']
            for params in result['params']:
                assert 'void_shape' in params
                assert 'void_count' in params
                assert params['void_count'] >= 1 and params['void_count'] <= 10

        # Verify metrics were computed
        import numpy as np
        for result in iteration_results:
            metrics = result['metrics']
            assert 'mmd_rbf' in metrics
            assert 'wasserstein' in metrics
            assert 'mean_nn_distance' in metrics
            # Check core metrics are numeric
            assert isinstance(metrics['mmd_rbf'], (float, np.floating, np.integer))
            assert isinstance(metrics['wasserstein'], (float, np.floating, np.integer))
            assert isinstance(metrics['mean_nn_distance'], (float, np.floating, np.integer))

        # Verify Pareto front exists
        pareto_front = optimizer.get_pareto_front()
        assert len(pareto_front) > 0, "Pareto front should have solutions"
        print(f"\n[Test] Final Pareto front size: {len(pareto_front)}")

        # Verify each trial has 3 objectives
        for trial in pareto_front:
            assert len(trial.values) == 3, "Each trial should have 3 objective values"

        # Verify SQLite persistence
        assert optimizer.study_path.exists(), "SQLite study file should exist"

        print("[Test] ✓ End-to-end integration test passed!")

    def test_experiment_reporter_visualization(self, mini_config, temp_test_dir):
        """Test ExperimentReporter generates all visualizations"""
        import numpy as np

        # Initialize components
        experiment_dir = Path(mini_config['experiment_dir'])
        experiment_dir.mkdir(parents=True, exist_ok=True)

        optimizer = OptunaOptimizer(experiment_dir, mini_config)
        sampler = ParameterSampler()
        generator = VoidGenerator(Path(mini_config['base_image_dir']))
        embedder = DinoV2Embedder(model_name=mini_config['dino_model'])
        reporter = ExperimentReporter(experiment_dir)

        # Generate real distribution
        print("\n[Test] Generating real distribution...")
        real_params = sampler.sample_parameter_sets('real', n_sets=2, seed=42)
        real_images, _ = generator.generate_batch(real_params, replications=1, seed_offset=0)
        real_embeddings_full = embedder.embed_batch(real_images)

        pca = PCAProjector(n_components=mini_config['pca_embedding_dim'])
        real_embeddings = pca.fit_transform(real_embeddings_full)

        # Save real sample images
        print("[Test] Saving real sample images...")
        reporter.save_sample_images(real_images, iteration=0, label="real", n_samples=2)

        # Track embeddings by iteration
        synthetic_embeddings_by_iter = {}

        # Run 2 optimization iterations
        prev_embeddings = real_embeddings
        prev_params = real_params
        prev_metrics = compute_all_metrics(real_embeddings, real_embeddings)

        for iteration in range(1, 3):
            print(f"\n[Test] === Iteration {iteration} ===")

            # Update metrics history
            reporter.update_metrics_history(iteration - 1, prev_metrics)

            # Get next parameters
            next_params, converged = optimizer.suggest_next_parameters(
                synthetic_embeddings=prev_embeddings,
                synthetic_params=prev_params,
                real_embeddings=real_embeddings,
                metrics=prev_metrics,
                iteration=iteration,
                config=mini_config
            )

            # Generate synthetic images
            synthetic_images, _ = generator.generate_batch(
                next_params,
                replications=mini_config['replications_per_iteration'],
                seed_offset=iteration * 1000
            )

            # Save sample images
            reporter.save_sample_images(synthetic_images, iteration=iteration, label="synthetic", n_samples=2)

            # Extract embeddings
            synthetic_embeddings_full = embedder.embed_batch(synthetic_images)
            synthetic_embeddings = pca.transform(synthetic_embeddings_full)

            # Store for 2D projection
            synthetic_embeddings_by_iter[iteration] = synthetic_embeddings

            # Compute metrics
            metrics = compute_all_metrics(synthetic_embeddings, real_embeddings)

            print(f"[Test] Iteration {iteration} metrics:")
            print(f"  mmd_rbf: {metrics['mmd_rbf']:.4f}")
            print(f"  wasserstein: {metrics['wasserstein']:.4f}")
            print(f"  mean_nn_distance: {metrics['mean_nn_distance']:.4f}")

            prev_embeddings = synthetic_embeddings
            prev_params = next_params
            prev_metrics = metrics

        # Update final metrics
        reporter.update_metrics_history(2, prev_metrics)

        # Generate visualizations
        print("\n[Test] Generating visualizations...")

        # 2D projection plot
        reporter.plot_2d_projection(
            real_embeddings,
            synthetic_embeddings_by_iter,
            initial_embeddings=real_embeddings  # Use real as initial
        )

        # Metrics optimization plot
        reporter.plot_metrics_optimization()

        # Pareto front plot
        pareto_front = optimizer.get_pareto_front()
        reporter.plot_pareto_front(pareto_front)

        # Summary report
        reporter.generate_summary_report()

        # Verify all outputs were created
        print("\n[Test] Verifying visualization outputs...")

        # Check sample images
        assert (reporter.samples_dir / "iter_000_real_samples.png").exists()
        assert (reporter.samples_dir / "iter_001_synthetic_samples.png").exists()
        assert (reporter.samples_dir / "iter_002_synthetic_samples.png").exists()

        # Check visualizations
        assert (reporter.viz_dir / "distribution_evolution_2d.png").exists()
        assert (reporter.viz_dir / "metrics_optimization.png").exists()
        assert (reporter.viz_dir / "pareto_front_2d_projections.png").exists()

        # Check JSON and text outputs
        assert (reporter.viz_dir / "metrics_history.json").exists()
        assert (reporter.viz_dir / "experiment_summary.txt").exists()

        print("[Test] ✓ All visualizations generated successfully!")

    def test_optimizer_persistence_across_runs(self, mini_config, temp_test_dir):
        """Test that optimizer state persists and can be resumed"""
        experiment_dir = Path(mini_config['experiment_dir'])
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # First run: Create optimizer and run 1 iteration
        optimizer1 = OptunaOptimizer(experiment_dir, mini_config)

        # Mock a simple iteration
        sampler = ParameterSampler()
        real_params = sampler.sample_parameter_sets('real', n_sets=2, seed=42)

        import numpy as np
        mock_embeddings = np.random.randn(4, mini_config['pca_embedding_dim'])
        mock_metrics = {'mmd_rbf': 0.15, 'wasserstein': 0.08, 'mean_nn_distance': 3.5}

        next_params1, _ = optimizer1.suggest_next_parameters(
            synthetic_embeddings=mock_embeddings,
            synthetic_params=real_params,
            real_embeddings=mock_embeddings,
            metrics=mock_metrics,
            iteration=1,
            config=mini_config
        )

        n_trials_1 = len(optimizer1.study.trials)
        pareto_size_1 = len(optimizer1.get_pareto_front())

        # Second run: Load existing optimizer
        optimizer2 = OptunaOptimizer(experiment_dir, mini_config)

        # Should have same number of trials (loaded from SQLite)
        n_trials_2 = len(optimizer2.study.trials)
        assert n_trials_2 == n_trials_1, "Trials should persist across optimizer instances"

        # Pareto front should also be preserved
        pareto_size_2 = len(optimizer2.get_pareto_front())
        assert pareto_size_2 == pareto_size_1, "Pareto front should persist"

        print(f"[Test] ✓ Persistence test passed! {n_trials_1} trials persisted across runs")

    @pytest.mark.slow
    def test_full_mini_experiment(self, mini_config, temp_test_dir):
        """
        Run a complete mini experiment (3 iterations).
        Marked as slow - only run with pytest -m slow or pytest --run-slow
        """
        experiment_dir = Path(mini_config['experiment_dir'])
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all components
        optimizer = OptunaOptimizer(experiment_dir, mini_config)
        sampler = ParameterSampler()
        generator = VoidGenerator(Path(mini_config['base_image_dir']))
        embedder = DinoV2Embedder(model_name=mini_config['dino_model'])

        # Generate real distribution
        print("\n[Full Test] Generating real distribution...")
        real_params = sampler.sample_parameter_sets('real', n_sets=3, seed=42)
        real_images, _ = generator.generate_batch(real_params, replications=2, seed_offset=0)

        real_embeddings_full = embedder.embed_batch(real_images)
        pca = PCAProjector(n_components=mini_config['pca_embedding_dim'])
        real_embeddings = pca.fit_transform(real_embeddings_full)

        # Track metrics over iterations
        all_metrics = []

        # Start with far distribution
        print("[Full Test] Generating initial synthetic distribution (far)...")
        initial_params = sampler.sample_parameter_sets('far', n_sets=3, seed=100)
        initial_images, _ = generator.generate_batch(initial_params, replications=2, seed_offset=1000)
        initial_embeddings = pca.transform(embedder.embed_batch(initial_images))
        initial_metrics = compute_all_metrics(initial_embeddings, real_embeddings)

        all_metrics.append(initial_metrics)
        print(f"[Full Test] Initial metrics: mmd_rbf={initial_metrics['mmd_rbf']:.4f}")

        # Run optimization iterations
        prev_embeddings = initial_embeddings
        prev_params = initial_params
        prev_metrics = initial_metrics

        for iteration in range(1, mini_config['max_iterations'] + 1):
            print(f"\n[Full Test] === Iteration {iteration} ===")

            # Optimizer suggests next parameters
            next_params, converged = optimizer.suggest_next_parameters(
                synthetic_embeddings=prev_embeddings,
                synthetic_params=prev_params,
                real_embeddings=real_embeddings,
                metrics=prev_metrics,
                iteration=iteration,
                config=mini_config
            )

            # Generate and evaluate
            synthetic_images, _ = generator.generate_batch(
                next_params,
                replications=mini_config['replications_per_iteration'],
                seed_offset=iteration * 10000
            )

            synthetic_embeddings = pca.transform(embedder.embed_batch(synthetic_images))
            metrics = compute_all_metrics(synthetic_embeddings, real_embeddings)

            all_metrics.append(metrics)

            print(f"[Full Test] Iteration {iteration} metrics:")
            print(f"  mmd_rbf: {metrics['mmd_rbf']:.4f}")
            print(f"  wasserstein: {metrics['wasserstein']:.4f}")
            print(f"  mean_nn_distance: {metrics['mean_nn_distance']:.4f}")

            prev_embeddings = synthetic_embeddings
            prev_params = next_params
            prev_metrics = metrics

            if converged:
                print(f"[Full Test] Converged at iteration {iteration}")
                break

        # Verify experiment completed
        assert len(all_metrics) > 1, "Should have multiple iterations of metrics"

        # Verify Pareto front grew
        final_pareto = optimizer.get_pareto_front()
        print(f"\n[Full Test] Final Pareto front: {len(final_pareto)} solutions")
        assert len(final_pareto) > 0

        print("[Full Test] ✓ Full mini experiment completed successfully!")

    def test_ask_tell_pattern_e2e(self, mini_config, temp_test_dir):
        """
        Test proper ask/tell pattern in end-to-end workflow (Stage 3).
        Verifies that pending trials are properly tracked and completed.
        """
        import numpy as np

        # Initialize components
        experiment_dir = Path(mini_config['experiment_dir'])
        experiment_dir.mkdir(parents=True, exist_ok=True)

        optimizer = OptunaOptimizer(experiment_dir, mini_config)
        sampler = ParameterSampler()
        generator = VoidGenerator(Path(mini_config['base_image_dir']))
        embedder = DinoV2Embedder(model_name=mini_config['dino_model'])

        # Generate real distribution
        print("\n[E2E Test] Generating real distribution...")
        real_params = sampler.sample_parameter_sets('real', n_sets=2, seed=42)
        real_images, _ = generator.generate_batch(real_params, replications=1, seed_offset=0)
        real_embeddings_full = embedder.embed_batch(real_images)

        pca = PCAProjector(n_components=mini_config['pca_embedding_dim'])
        real_embeddings = pca.fit_transform(real_embeddings_full)

        # Track pending trials state throughout experiment
        pending_trials_history = []

        # Run 3 iterations and verify ask/tell pattern
        prev_embeddings = real_embeddings
        prev_params = real_params
        prev_metrics = compute_all_metrics(real_embeddings, real_embeddings)

        for iteration in range(3):
            print(f"\n[E2E Test] === Iteration {iteration} ===")

            # Get next parameters (triggers ask/tell)
            next_params, converged = optimizer.suggest_next_parameters(
                synthetic_embeddings=prev_embeddings,
                synthetic_params=prev_params,
                real_embeddings=real_embeddings,
                metrics=prev_metrics,
                iteration=iteration,
                config=mini_config
            )

            # Verify pending trials state
            print(f"[E2E Test] Pending trials after iteration {iteration}: {list(optimizer.pending_trials.keys())}")
            pending_trials_history.append({
                'iteration': iteration,
                'pending_iterations': list(optimizer.pending_trials.keys()),
                'n_pending_trials': {k: len(v) for k, v in optimizer.pending_trials.items()}
            })

            # Verify current iteration has pending trials
            assert iteration in optimizer.pending_trials, f"Iteration {iteration} should have pending trials"
            assert len(optimizer.pending_trials[iteration]) == mini_config['iteration_batch_size']

            # Verify previous iteration was completed (if exists)
            if iteration > 0:
                assert (iteration - 1) not in optimizer.pending_trials, \
                    f"Iteration {iteration-1} should have been completed"

            # Generate synthetic images with suggested parameters
            print(f"[E2E Test] Generating {len(next_params)} synthetic samples...")
            synthetic_images, _ = generator.generate_batch(
                next_params,
                replications=mini_config['replications_per_iteration'],
                seed_offset=iteration * 1000
            )

            # Extract embeddings
            synthetic_embeddings_full = embedder.embed_batch(synthetic_images)
            synthetic_embeddings = pca.transform(synthetic_embeddings_full)

            # Compute metrics for next iteration
            metrics = compute_all_metrics(synthetic_embeddings, real_embeddings)

            print(f"[E2E Test] Iteration {iteration} metrics:")
            print(f"  mmd_rbf: {metrics['mmd_rbf']:.4f}")
            print(f"  wasserstein: {metrics['wasserstein']:.4f}")
            print(f"  mean_nn_distance: {metrics['mean_nn_distance']:.4f}")

            # Update for next iteration
            prev_embeddings = synthetic_embeddings
            prev_params = next_params
            prev_metrics = metrics

        # Verify final state
        print("\n[E2E Test] Verifying final state...")

        # Only last iteration should have pending trials
        assert len(optimizer.pending_trials) == 1, "Only last iteration should have pending trials"
        assert 2 in optimizer.pending_trials, "Iteration 2 should have pending trials"

        # Verify completed trials count
        completed_trials = [t for t in optimizer.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        expected_completed = 2 * mini_config['iteration_batch_size']  # Iterations 0 and 1
        assert len(completed_trials) == expected_completed, \
            f"Expected {expected_completed} completed trials, got {len(completed_trials)}"

        # Verify each completed trial has 3 objectives
        for trial in completed_trials:
            assert len(trial.values) == 3, "Each trial should have 3 objective values"
            assert all(isinstance(v, (float, int)) for v in trial.values), "All values should be numeric"

        # Verify Pareto front exists
        pareto_front = optimizer.get_pareto_front()
        assert len(pareto_front) > 0, "Pareto front should have solutions"

        print(f"[E2E Test] Completed trials: {len(completed_trials)}")
        print(f"[E2E Test] Pareto front size: {len(pareto_front)}")
        print(f"[E2E Test] Pending trials state history:")
        for state in pending_trials_history:
            print(f"  Iteration {state['iteration']}: pending={state['pending_iterations']}, counts={state['n_pending_trials']}")

        print("[E2E Test] ✓ Ask/tell pattern E2E test passed!")
