"""
Champion Multi-Modal Training Script (train_mega.py)
--------------------------------------------------
Goal: Train the best possible delamination model using 100GB of fused multi-modal data.
"""
print("DEBUG: [1/5] Process Started. Importing System modules...")
import sys
import os
from pathlib import Path

# EMERGENCY: FORCE CPU to bypass GPU Driver Lock
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Force unbuffered output just in case
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("DEBUG: [2/5] Importing Torch (This may take a moment)...")
import torch
print(f"DEBUG: [3/5] Torch Imported. CUDA Available: {torch.cuda.is_available()}")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import json
import logging
from src.data.multimodal_loader import get_mega_loader
from src.models.integrated.framework import IntegratedDelaminationFramework

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'mega_training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_mega():
    logger.info("=" * 80)
    logger.info("CHAMPIONSHIP MULTI-MODAL TRAINING RUN (100GB TARGET)")
    logger.info("=" * 80)
    
    # Config
    CONFIG = {
        'epochs': 200,
        'batch_size': 32,
        'grad_accum': 4,
        'learning_rate': 2e-4,
        'weight_decay': 0.05,
        'checkpoint_dir': PROJECT_ROOT / "src" / "training" / "checkpoints" / "mega_run"
    }
    CONFIG['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
    
    # 1. Load Mega Data
    logger.info("[1] Initializing Mega DataLoader...")
    data_root = PROJECT_ROOT / "data" / "raw"
    loader = get_mega_loader(data_root, batch_size=CONFIG['batch_size'])
    logger.info(f"    Unified Training Samples: {len(loader.dataset)}")
    
    # 2. Initialize Framework
    logger.info("[2] Initializing Integrated Framework...")
    model_config = {
        'snpi_net': {'adaptive_kernel': {'input_dim': 6}},
        'cad_former': {'d_model': 128, 'n_layers': 4},
        'al_vtfd': {}
    }
    model = IntegratedDelaminationFramework(model_config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"    Model moved to {device}. Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Training Components
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    criterion = nn.MSELoss()
    
    # 4. Training Loop
    logger.info("[3] Starting Multi-Modal Gradient Descent...")
    best_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(loader):
            features = batch['features'].to(device) # [B, 2048]
            image = batch['image'].to(device)       # [B, 3, 224, 224]
            target = batch['target'].to(device)     # [B, 1]
            
            optimizer.zero_grad()
            
            # Robust Input Handling for Multi-Modal Data
            # 1. Primary Composite Data (NASA/F-MOC)
            if features.shape[1] >= 262: 
                laminate_config = features[:, 6:262].view(-1, 4, 64)
                loading_history = features[:, 262:362]
                physics_inputs = features[:, :6]
            else:
                # 2. Transfer Learning Data (SDNET/Ultrasonic)
                # Fallback: Zero-pad or use simplified inputs for pre-training
                physics_inputs = features[:, :6] if features.shape[1] >= 6 else torch.zeros(features.size(0), 6, device=device)
                laminate_config = torch.zeros(features.size(0), 4, 64, device=device) # Dummy config
                loading_history = torch.zeros(features.size(0), 100, device=device)   # Dummy history
                
            # Mixed Precision Forward
            with torch.amp.autocast('cuda') if scaler else torch.autocast('cpu', enabled=False):
                outputs = model.predict_delamination(
                    laminate_config, loading_history,
                    physics_inputs=physics_inputs,
                    meso_data=image
                )
                loss = criterion(outputs['delamination_area'], target)
            
            # Backprop
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            epoch_losses.append(loss.item())
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.6f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        scheduler.step()
        
        # Checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CONFIG['checkpoint_dir'] / "mega_best_model.pt")
            logger.info(f"--- NEW BEST | Epoch {epoch} | Avg Loss: {avg_loss:.6f} ---")
            
        if epoch % 20 == 0:
            torch.save(model.state_dict(), CONFIG['checkpoint_dir'] / f"mega_checkpoint_e{epoch}.pt")

    logger.info("Training Complete. Best Model Saved.")

if __name__ == "__main__":
    train_mega()
