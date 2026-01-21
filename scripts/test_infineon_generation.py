from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.infineon_void_generator import InfineonVoidGenerator

if __name__ == '__main__':
    base_image_dir = Path(__file__).parent.parent / "data" / "base_chips"
    output_dir = Path(__file__).parent.parent / "data" / "test_infineon_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = InfineonVoidGenerator(base_image_dir=base_image_dir)

    test_params = {
        'void_type': 'circular_shadow',
        'package_type': '53440',
        'num_splatters_min': 1,
        'num_splatters_max': 2,
        'radius_min': 25,
        'radius_max': 50,
        'irregularity_min': 0.05,
        'irregularity_max': 0.10,
        'shadow_width_min': 0.15,
        'shadow_width_max': 0.20,
        'shadow_opacity_min': 0.80,
        'shadow_opacity_max': 0.95,
    }

    print("Testing circular_shadow generation...")
    try:
        img, metadata = generator.generate_single(test_params, seed=42)
        print(f"✓ Success! Generated image shape: {img.shape}, mask shape: {metadata['mask'].shape}")

        # Save test output
        import cv2
        cv2.imwrite(str(output_dir / "test_circular_shadow.png"), img)
        cv2.imwrite(str(output_dir / "test_circular_shadow_mask.png"), metadata['mask'])
        print(f"✓ Saved to {output_dir}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
