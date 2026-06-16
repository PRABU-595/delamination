import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import sys
import os
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.integrated.framework import IntegratedDelaminationFramework
from src.models.al_vtfd.virtual_testing import VirtualTestingFramework
from src.data.nasa_loader import get_nasa_loader
from src.training.losses import PhysicsInformedLoss

def train_integrated_system(train_config_path, model_config_path="config/model_config.yaml"):
    """
    Main training workflow for the Integrated Delamination Framework.
    """
    print("Loading Configuration...")
    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)
        
    with open(model_config_path, 'r') as f:
        model_config_full = yaml.safe_load(f)
        
    # Adapting config for Framework
    framework_config = {
        'snpi': model_config_full.get('snpi_net', {}),
        'cad_former': model_config_full.get('cad_former', {})
    }
        
    # 1. Initialize Integrated Model
    print("Initializing Integrated Framework...")
    model = IntegratedDelaminationFramework(framework_config)
    
    # 2. Setup Data
    print("Setting up Data Loaders...")
    # Using 'nasa_loader' from data config
    train_loader = get_nasa_loader(config_path='config/data_config.yaml', 
                                   batch_size=train_config['training']['batch_size'], 
                                   split='train')
    
    # 3. Setup AL Framework
    print("Initializing Active Learning Wrapper...")
    al_framework = VirtualTestingFramework(model)
    
    # 4. Pre-training (Physics-Informed Loop)
    print("Starting Pre-training Phase...")
    optimizer = optim.Adam(model.parameters(), lr=float(train_config['training'].get('learning_rate', 1e-4)))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Map config weights to loss function keys
    raw_weights = train_config['training'].get('loss_weights', {})
    loss_weights = {
        'mse': raw_weights.get('prediction', 1.0),
        'physics': raw_weights.get('physics', 0.1),
        'uncertainty': raw_weights.get('uncertainty', 0.05),
        'thermo': 0.05,
        'energy': 0.05
    }
    loss_fn = PhysicsInformedLoss(loss_weights)
    
    # Helper to generate standard laminate config (since loader has placeholders)
    # Using [0, 45, -45, 90]_s quasi-isotropic layup common in NASA/F-MOC
    from src.physics.laminate_theory import ABD_matrix
    
    def get_standard_laminate(batch_size, device):
        # [0, 45, -45, 90]_s -> 8 plies
        angles = [0, 45, -45, 90, 90, -45, 45, 0]
        # ABD computation on CPU for stability then move to device
        abd_res = ABD_matrix(angles) 
        
        return {
            'ply_angles': torch.tensor(angles).unsqueeze(0).repeat(batch_size, 1).to(device),
            'depths': abd_res['z_coords'].unsqueeze(0).repeat(batch_size, 1).to(device),
            'abd_matrix': abd_res['ABD'].unsqueeze(0).repeat(batch_size, 1, 1).to(device),
            'is_symmetric': torch.tensor(True).unsqueeze(0).repeat(batch_size, 1).to(device)
        }

    epochs = train_config['training'].get('epochs', 50)
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loss_history = {}
        
        for i, batch in enumerate(train_loader):
            features = batch['features'].to(device=model.device) # [Batch, 2048]
            targets = batch['target'].to(device=model.device)   # [Batch, 1]
            
            # --- Input Construction ---
            # 1. Laminate Config (Standardized for now)
            laminate_config = get_standard_laminate(features.shape[0], model.device)
            
            # 2. Extract History proxy from features
            # Loader features[262:362] are history proxy
            # Full signal is at [362:1386]
            loading_history = features[:, 362:1386] 
            
            # 3. Macro features (Load/Disp)
            macro_data = features[:, 262:326] # 64 dims
            
            # 4. Micro features (High freq)
            micro_data = features[:, -80:] # Last 80 dims
            
            # 5. Physics params (constants)
            physics_params = {
                'G_Ic': 280.0,
                'G_IIc': 790.0
            }
            
            # Forward
            # Note: Framework.forward signature:
            # (micro, meso, macro, laminate_config, loading_history, physics_params)
            
            # Meso data (Images)
            meso_data = None
            if 'image' in batch:
                meso_data = batch['image'].to(device=model.device)
                if meso_data.shape[1] == 1: meso_data = meso_data.repeat(1, 3, 1, 1)

            predictions = model(
                micro_data=micro_data,
                meso_data=meso_data,
                macro_data=macro_data,
                laminate_config=laminate_config,
                loading_history=loading_history,
                physics_params=physics_params
            )
            
            # Target Dictionary Construction
            target_dict = {
                'area': targets,
                # Heuristic: Growth rate proportional to target / time proxy
                'growth_rate': targets * 0.1, 
                'migration': torch.zeros(features.shape[0], dtype=torch.long).to(model.device) # Default index 0
            }
            
            # Compute Loss
            loss, components = loss_fn(predictions, target_dict)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Aggregate components for logging
            for k, v in components.items():
                loss_history[k] = loss_history.get(k, 0.0) + v
            
        # End of Epoch
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        # Log breakdown
        log_msg = f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | "
        log_msg += f"MSE: {loss_history['mse']/len(train_loader):.4f} | "
        log_msg += f"Phys: {loss_history['physics']/len(train_loader):.4f}"
        print(log_msg)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save Checkpoint
            # ...
        
    # 5. Active Learning Phase
    if train_config.get('active_learning', True):
        print("\nStarting Active Learning Phase (Virtual Testing)...")
        results = al_framework.run_optimization(
            material_system="NASA_CFRP_PCoE",
            budget_tests=train_config.get('al_budget', 20)
        )
        print(f"Final Active Learning Accuracy: {results.get('final_accuracy', 0.0):.4f}")
    
    # Save Model
    save_path = Path("results/models/integrated_framework.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Ensure config exists or pass dict
    if os.path.exists("config/training_config.yaml"):
        train_integrated_system("config/training_config.yaml")
    else:
        print("Config not found, creating dummy config...")
        # Create minimal config for test run
        dummy_train = {'training': {'batch_size': 4, 'epochs': 2, 'loss_weights': {'prediction': 1.0}}}
        dummy_model = {'snpi_net': {}, 'cad_former': {}}
        with open("config/training_config.yaml", 'w') as f: yaml.dump(dummy_train, f)
        with open("config/model_config.yaml", 'w') as f: yaml.dump(dummy_model, f)
        train_integrated_system("config/training_config.yaml")
