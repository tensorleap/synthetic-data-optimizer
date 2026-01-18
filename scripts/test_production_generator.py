from pathlib import Path
from src.data_generation.production_data_generator import ProductionDataGenerator
from scripts.run_experiment import run_optimizer_iteration

if __name__ == '__main__':
    base_image_dir = Path(__file__).parent.parent / "data" / "base_chips"
    config_path = Path(__file__).parent.parent / "configs" / "local_experiment_config.yaml"

    generator = ProductionDataGenerator(
        base_image_dir=base_image_dir,
        unet_model_path=None,
        config_path=config_path
    )

    real_embeddings, embeddings_per_sim, metadata_per_sim = generator.generate_data_for_optimizer(
        real_distribution_type='real',
        synthetic_distribution_type='far',
        n_real_samples=100,
        n_distributions=2,
        n_samples_per_distribution=15,
        target_embedding_dim=400,
        seed=42
    )

    print("\n" + "="*60)
    print("TESTING WITH OPTIMIZER")
    print("="*60)

    suggestions_df, best_trials_df = run_optimizer_iteration(
        real_embeddings=real_embeddings,
        embeddings_per_simulation=embeddings_per_sim,
        metadata_per_simulation=metadata_per_sim
    )

    print("\n[SUGGESTIONS]")
    print(suggestions_df.head(10))

    print("\n[BEST TRIALS]")
    print(best_trials_df.head(10))

    print("\nSUCCESS! Production generator works with run_optimizer_iteration")
