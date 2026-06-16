"""
Fine-tuning script for Migration Tracking.
"Teaches" the CAD-Former migration head to recognize interlaminar delamination migration
using supervised synthetic data.
"""
import sys
import os
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.integrated.framework import IntegratedDelaminationFramework
from experiments.run_extended_validation import generate_samples, prepare_inputs
from src.training.losses import PhysicsInformedLoss

def fine_tune_migration():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 MIGRATION FINE-TUNER INITIALIZED (Device: {device})")

    # 1. Load Pre-trained Model
    config = {
        'snpi_net': {'adaptive_kernel': {'input_dim': 6}},
        'cad_former': {'d_model': 128, 'n_layers': 4},
        'al_vtfd': {}
    }
    model = IntegratedDelaminationFramework(config)
    
    # Load best previous weights (OOD checkpoint)
    checkpoint_path = Path("src/training/checkpoints/mega_run/mega_best_model_ood.pt")
    if not checkpoint_path.exists():
        checkpoint_path = Path("src/training/checkpoints/mega_run/mega_best_model.pt")
        
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        print(f"  ✅ Model loaded from {checkpoint_path}")
    else:
        print("  ⚠️ No checkpoint found. Initializing random weights.")
    
    model.to(device)

    # 2. Generate Supervised Dataset (2,000 samples)
    print("  Generating 2,000 supervised synthetic samples...")
    # Use different seeds for train/val
    train_data = generate_samples(1600, 'standard', seed=123)
    val_data = generate_samples(400, 'standard', seed=456)

    # 3. Setup Migration-Only Optimizer
    # Freeze everything except migration heads and spatial attention (for 3D context)
    for param in model.parameters():
        param.requires_grad = False
    
    migration_params = []
    # CAD-Former migration heads
    for name, param in model.cad_former.named_parameters():
        if 'migration' in name:
            param.requires_grad = True
            migration_params.append(param)
            print(f"    Unfreezing: {name}")
    
    # Also unfreeze spatial attention to allow the model to 'look' at the right interfaces
    for name, param in model.cad_former.named_parameters():
        if 'spatial' in name:
            param.requires_grad = True
            migration_params.append(param)
    
    optimizer = optim.Adam(migration_params, lr=5e-4) # Slightly higher LR for quick head convergence
    
    # Weighted Loss: focus heavily on migration
    loss_fn = PhysicsInformedLoss({
        'mse': 0.1,      # Low priority for area/growth during this phase
        'migration': 5.0, # High priority!
        'physics': 0.1
    })

    # 4. Fine-tuning Loop
    print("\n  Starting Supervised Migration Learning (30 epochs)...")
    batch_size = 32
    n_batches = 1600 // batch_size
    
    best_val_acc = 0.0
    
    for epoch in range(30):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        # Simple slicing for batches
        indices = np.random.permutation(1600)
        
        for i in range(n_batches):
            idx = indices[i*batch_size : (i+1)*batch_size]
            batch = {
                'features': train_data['features'][idx],
                'image': train_data['image'][idx],
                'target': train_data['target'][idx],
                'migration_gt': train_data['migration_gt'][idx]
            }
            
            optimizer.zero_grad()
            inputs = prepare_inputs(batch, device)
            outputs = model.predict_delamination(
                inputs['laminate_config'], inputs['loading_history'],
                physics_inputs=inputs['physics_inputs'],
                meso_data=inputs['image']
            )
            
            target_indices = batch['migration_gt'].to(device)
            # Define probs first
            probs = outputs['migration_interface']
            target_one_hot = F.one_hot(target_indices, num_classes=probs.size(1)).float()
            
            target_dict = {
                'area': torch.from_numpy(inputs['target']).float().to(device).unsqueeze(1),
                'growth_rate': torch.from_numpy(inputs['target']).float().to(device).unsqueeze(1) * 0.1,
                'migration': target_one_hot
            }
            
            loss_inputs = {
                'delamination_area': outputs['delamination_area'],
                'growth_rate': outputs['growth_rate'],
                'migration_probs': outputs['migration_interface']
            }
            
            loss, _ = loss_fn(loss_inputs, target_dict)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Accuracy calc
            probs = outputs['migration_interface']
            preds = torch.argmax(probs, dim=1)
            correct += (preds == target_indices).sum().item()
            total += target_indices.size(0)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            v_idx = np.arange(400)
            v_batch = {
                'features': val_data['features'][v_idx],
                'image': val_data['image'][v_idx],
                'target': val_data['target'][v_idx],
                'migration_gt': val_data['migration_gt'][v_idx]
            }
            v_inputs = prepare_inputs(v_batch, device)
            v_outputs = model.predict_delamination(
                v_inputs['laminate_config'], v_inputs['loading_history'],
                physics_inputs=v_inputs['physics_inputs'],
                meso_data=v_inputs['image']
            )
            v_preds = torch.argmax(v_outputs['migration_interface'], dim=1)
            val_correct = (v_preds == v_batch['migration_gt'].to(device)).sum().item()
            val_total = 400
        
        val_acc = val_correct / val_total
        print(f"    Epoch {epoch+1}/30 - Loss: {epoch_loss/n_batches:.4f} | Train Acc: {correct/total*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save progress
            save_path = Path("src/training/checkpoints/mega_run/mega_best_model_migration.pt")
            torch.save(model.state_dict(), save_path)

    print(f"\n  ✅ Training Complete. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"  ✅ Model saved to src/training/checkpoints/mega_run/mega_best_model_migration.pt")

if __name__ == "__main__":
    fine_tune_migration()
