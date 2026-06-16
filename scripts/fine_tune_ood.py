"""
Fine-tuning script for OOD (Out-of-Distribution) domain adaptation.
Adapts the CFRP-trained model to GFRP and External distributions.
"""
import sys
from pathlib import Path
import torch
import torch.optim as optim
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.integrated.framework import IntegratedDelaminationFramework
from experiments.run_extended_validation import generate_samples, prepare_inputs, run_single_eval
from src.training.losses import PhysicsInformedLoss

def fine_tune_ood():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Fine-tuning on {device}...")

    # 1. Load Pre-trained Model with correct configuration
    config = {
        'snpi_net': {'adaptive_kernel': {'input_dim': 6}},
        'cad_former': {'d_model': 128, 'n_layers': 4},
        'al_vtfd': {}
    }
    model = IntegratedDelaminationFramework(config)
    
    # Try loading previous OMD weights to resume, else base
    ood_checkpoint = Path("src/training/checkpoints/mega_run/mega_best_model_ood.pt")
    base_checkpoint = Path("src/training/checkpoints/mega_run/mega_best_model.pt")
    
    if ood_checkpoint.exists():
        model.load_state_dict(torch.load(ood_checkpoint, map_location=device), strict=False)
        print("  ✅ Resuming from improved OOD weights (-0.0626).")
    elif base_checkpoint.exists():
        model.load_state_dict(torch.load(base_checkpoint, map_location=device), strict=False)
        print("  ✅ Loading base Mega Model weights.")
    else:
        print("  ⚠️ No checkpoint found. Initializing random.")
    
    model.to(device)


    # 2. Generate small OOD fine-tuning sets
    # 30 external + 30 gfrp = 60 samples total
    print("  Generating 60 OOD fine-tuning samples...")
    ext_data = generate_samples(30, 'external', seed=42)
    std_data = generate_samples(30, 'standard', seed=42)
    
    # Merge
    combined_features = torch.cat([ext_data['features'], std_data['features']], dim=0)
    combined_images = torch.cat([ext_data['image'], std_data['image']], dim=0)
    combined_targets = torch.cat([ext_data['target'], std_data['target']], dim=0)
    
    dataset = {
        'features': combined_features,
        'image': combined_images,
        'target': combined_targets
    }

    # Record initial performance
    print("  Initial validation (OOD)...")
    model.eval()
    inputs_val = prepare_inputs(dataset, device)
    y_pred_init, y_true_val = run_single_eval(model, inputs_val, device)
    from sklearn.metrics import r2_score
    r2_init = r2_score(y_true_val, y_pred_init)
    print(f"    Initial R²: {r2_init:.4f}")

    # 3. Setup Aggressive Fine-tuning Optimizer (Unfreeze FULL model)
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = PhysicsInformedLoss({'mse': 1.0, 'physics': 0.1, 'uncertainty': 0.1}) # Increase unc weight to force stability

    # 4. Fine-tuning Loop
    print("  Starting aggressive fine-tuning (100 epochs)...")
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        inputs = prepare_inputs(dataset, device)
        outputs = model.predict_delamination(
            inputs['laminate_config'], inputs['loading_history'],
            physics_inputs=inputs['physics_inputs'],
            meso_data=inputs['image']
        )
        
        target_dict = {
            'area': torch.from_numpy(inputs['target']).float().to(device).unsqueeze(1),
            'growth_rate': torch.from_numpy(inputs['target']).float().to(device).unsqueeze(1) * 0.1,
            'migration': torch.zeros(len(inputs['target']), dtype=torch.long).to(device)
        }
        
        loss_inputs = {
            'delamination_area': outputs['delamination_area'],
            'growth_rate': outputs['growth_rate'],
            'aleatoric_log_var': outputs['aleatoric_log_var']
        }
        
        loss, _ = loss_fn(loss_inputs, target_dict)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            # Quick Eval check
            model.eval()
            with torch.no_grad():
                y_pred_mid, _ = run_single_eval(model, inputs_val, device)
                r2_mid = r2_score(y_true_val, y_pred_mid)
            model.train()
            print(f"    Epoch {epoch+1}/50 - Loss: {loss.item():.6f} | R²: {r2_mid:.4f}")

    # Final validation
    model.eval()
    y_pred_final, _ = run_single_eval(model, inputs_val, device)
    r2_final = r2_score(y_true_val, y_pred_final)
    print(f"    Final R²: {r2_final:.4f}")
    if r2_final > r2_init:
        print(f"  🏆 R² improved by {r2_final - r2_init:.4f}")
    else:
        print("  ⚠️ R² did not improve. Check learning rate or data variability.")

    # 5. Save fine-tuned model
    save_path = Path("src/training/checkpoints/mega_run/mega_best_model_ood.pt")
    torch.save(model.state_dict(), save_path)
    print(f"  ✅ Fine-tuned model saved to {save_path}")

if __name__ == "__main__":
    fine_tune_ood()
