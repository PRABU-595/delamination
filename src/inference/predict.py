"""
Delamination Prediction Inference Script.

Load a trained model and make predictions on new MAT files.
User provides the path to their experimental data.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import scipy.io
import numpy as np
from PIL import Image
import torchvision.transforms as T


def load_trained_model(checkpoint_path: str = None):
    """Load the trained delamination prediction model."""
    from src.models.integrated.framework import IntegratedDelaminationFramework
    
    config = {
        'snpi_net': {'adaptive_kernel': {'input_dim': 6}},
        'cad_former': {'d_model': 128, 'n_layers': 2},
        'al_vtfd': {}
    }
    
    model = IntegratedDelaminationFramework(config)
    
    # Load trained weights
    if checkpoint_path is None:
        checkpoint_path = PROJECT_ROOT / "src" / "training" / "checkpoints" / "best_model.pt"
    
    checkpoint_path = Path(checkpoint_path)
    
    if checkpoint_path.exists():
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded (trained to epoch {checkpoint.get('epoch', 'N/A')})")
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}")
        print("Using untrained model!")
    
    model.eval()
    return model


def parse_mat_file(mat_path: str) -> dict:
    """
    Parse a NASA-format MAT file to extract features.
    
    Expected structure:
    coupon -> PZT_data -> signal_sensor (features)
    coupon -> straingage_data -> stiffness_degradation (target if available)
    """
    mat_path = Path(mat_path)
    
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    
    print(f"\nParsing: {mat_path.name}")
    
    mat = scipy.io.loadmat(str(mat_path))
    
    features = np.zeros(2048, dtype=np.float32)
    target = None
    
    if 'coupon' in mat:
        coupon = mat['coupon'][0, 0]
        
        # Extract PZT features
        if 'PZT_data' in coupon.dtype.names:
            pzt = coupon['PZT_data'][0, 0]
            if 'signal_sensor' in pzt.dtype.names:
                signals = pzt['signal_sensor']
                if signals.size > 0:
                    raw = signals[0]
                    if isinstance(raw, np.ndarray):
                        flat = raw.astype(np.float32).flatten()
                        length = min(len(flat), 2048)
                        features[:length] = flat[:length]
                        print(f"  PZT signals loaded: {length} samples")
        
        # Extract target if available
        if 'straingage_data' in coupon.dtype.names:
            sg = coupon['straingage_data'][0, 0]
            if 'stiffness_degradation' in sg.dtype.names:
                deg = sg['stiffness_degradation']
                if deg.size > 0:
                    target = float(deg.flat[0])
                    print(f"  Known stiffness degradation: {target:.4f}")
    else:
        print("  Warning: 'coupon' structure not found in MAT file")
    
    return {
        'features': torch.from_numpy(features),
        'target': target
    }


def load_xray_image(image_path: str) -> torch.Tensor:
    """Load and preprocess an X-ray image."""
    if image_path and Path(image_path).exists():
        img = Image.open(image_path).convert('L')
        transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        return transform(img).unsqueeze(0).expand(1, 3, -1, -1)  # [1, 3, 64, 64]
    else:
        return torch.zeros(1, 3, 64, 64)


def predict_delamination(model, mat_path: str, xray_path: str = None):
    """
    Make delamination prediction on a single sample.
    
    Args:
        model: Trained IntegratedDelaminationFramework
        mat_path: Path to MAT file with PZT data
        xray_path: Optional path to X-ray image
    
    Returns:
        Dictionary with predictions
    """
    # Parse MAT file
    data = parse_mat_file(mat_path)
    features = data['features'].unsqueeze(0)  # Add batch dim
    
    # Load X-ray if provided
    xray = load_xray_image(xray_path)
    
    # Prepare model inputs
    laminate_config = features[:, :256].view(1, 4, 64)
    loading_history = features[:, 256:356]
    physics_inputs = features[:, :6]
    
    # Run prediction
    with torch.no_grad():
        outputs = model.predict_delamination(
            laminate_config,
            loading_history,
            physics_inputs=physics_inputs,
            meso_data=xray
        )
    
    # Extract results
    results = {
        'delamination_area': outputs['delamination_area'].item(),
        'growth_rate': outputs['growth_rate'].item(),
        'uncertainty': outputs['uncertainty'].mean().item(),
        'known_target': data['target']
    }
    
    return results


def print_results(results: dict):
    """Pretty print prediction results."""
    print("\n" + "=" * 50)
    print("DELAMINATION PREDICTION RESULTS")
    print("=" * 50)
    
    print(f"\n  Predicted Delamination Area: {results['delamination_area']:.6f}")
    print(f"  Predicted Growth Rate:       {results['growth_rate']:.6f}")
    print(f"  Prediction Uncertainty:      {abs(results['uncertainty']):.6f}")
    
    if results['known_target'] is not None:
        error = abs(results['delamination_area'] - results['known_target'])
        print(f"\n  Known Target Value:          {results['known_target']:.6f}")
        print(f"  Prediction Error:            {error:.6f}")
    
    # Interpretation
    print("\n" + "-" * 50)
    print("INTERPRETATION:")
    
    area = results['delamination_area']
    if area < 0.01:
        print("  ✓ Delamination level: MINIMAL")
    elif area < 0.1:
        print("  ⚠ Delamination level: MODERATE - monitoring recommended")
    else:
        print("  ✗ Delamination level: SIGNIFICANT - inspection required")
    
    unc = abs(results['uncertainty'])
    if unc < 0.1:
        print("  ✓ Prediction confidence: HIGH")
    elif unc < 0.5:
        print("  ⚠ Prediction confidence: MODERATE")
    else:
        print("  ⚠ Prediction confidence: LOW - more data recommended")
    
    print("=" * 50)


def main():
    """Interactive inference script."""
    print("=" * 60)
    print("  DELAMINATION PREDICTION SYSTEM")
    print("  Using Trained SNPI-Net + CAD-Former Model")
    print("=" * 60)
    
    # Load model
    print("\n[1] Loading trained model...")
    model = load_trained_model()
    
    while True:
        print("\n" + "-" * 60)
        print("Enter path to your MAT file (or 'quit' to exit):")
        mat_path = input("> ").strip()
        
        if mat_path.lower() in ['quit', 'exit', 'q']:
            print("\nExiting. Goodbye!")
            break
        
        if not mat_path:
            print("Error: Please enter a file path")
            continue
        
        # Optional X-ray image
        print("\nEnter path to X-ray image (press Enter to skip):")
        xray_path = input("> ").strip()
        if not xray_path:
            xray_path = None
        
        try:
            # Run prediction
            print("\n[2] Running prediction...")
            results = predict_delamination(model, mat_path, xray_path)
            
            # Display results
            print_results(results)
            
        except FileNotFoundError as e:
            print(f"\nError: {e}")
        except Exception as e:
            print(f"\nError during prediction: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
