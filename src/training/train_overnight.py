"""
Overnight Extended Training Script.

Run this script overnight to train on the full NASA dataset with extended epochs.
Usage: python src/training/train_overnight.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_overnight():
    """Extended training for overnight run."""
    from src.training.train_real_data import NASARealDataset
    from src.models.integrated.framework import IntegratedDelaminationFramework
    
    logger.info("=" * 70)
    logger.info("OVERNIGHT EXTENDED TRAINING")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now()}")
    
    # Configuration for extended training
    CONFIG = {
        'epochs': 200,           # Extended epochs for overnight
        'batch_size': 8,         # Larger batch if memory allows
        'learning_rate': 5e-5,   # Lower LR for fine-tuning
        'weight_decay': 1e-4,
        'warmup_epochs': 10,
        'checkpoint_every': 20,  # Save every N epochs
    }
    
    logger.info(f"Config: {CONFIG}")
    
    # Data root
    DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "NASA_CFRP"
    
    # Load datasets
    logger.info("\n[1] Loading datasets...")
    train_dataset = NASARealDataset(str(DATA_ROOT), split='train')
    val_dataset = NASARealDataset(str(DATA_ROOT), split='val')
    
    if len(train_dataset) == 0:
        logger.error("No training samples found!")
        return None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=0,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val samples: {len(val_dataset)}")
    logger.info(f"  Batches per epoch: {len(train_loader)}")
    
    # Initialize model
    logger.info("\n[2] Initializing model...")
    
    model_config = {
        'snpi_net': {'adaptive_kernel': {'input_dim': 6}},
        'cad_former': {'d_model': 128, 'n_layers': 2},
        'al_vtfd': {}
    }
    
    model = IntegratedDelaminationFramework(model_config)
    
    # Load previous best model if exists
    checkpoint_dir = PROJECT_ROOT / "src" / "training" / "checkpoints"
    best_checkpoint = checkpoint_dir / "best_model.pt"
    
    if best_checkpoint.exists():
        logger.info(f"  Loading previous best model from: {best_checkpoint}")
        checkpoint = torch.load(best_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"  Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
        start_val_loss = checkpoint.get('val_loss', float('inf'))
    else:
        start_val_loss = float('inf')
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total_params:,}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"  Device: {device}")
    model = model.to(device)
    
    # Optimizer with warm restarts
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG['learning_rate'], 
        weight_decay=CONFIG['weight_decay']
    )
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=50,  # Restart every 50 epochs
        T_mult=2
    )
    
    criterion = nn.MSELoss()
    
    # History
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': start_val_loss,
        'best_epoch': 0,
        'learning_rates': []
    }
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    logger.info("\n[3] Starting training...")
    logger.info("-" * 70)
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(device)
            xray = batch['xray'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            
            batch_size = features.shape[0]
            laminate_config = features[:, :256].view(batch_size, 4, 64)
            loading_history = features[:, 256:356]
            physics_inputs = features[:, :6]
            
            outputs = model.predict_delamination(
                laminate_config, loading_history,
                physics_inputs=physics_inputs,
                meso_data=xray.expand(-1, 3, -1, -1)
            )
            
            pred = outputs['delamination_area']
            loss = criterion(pred, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        scheduler.step()
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['learning_rates'].append(scheduler.get_last_lr()[0])
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                xray = batch['xray'].to(device)
                target = batch['target'].to(device)
                
                batch_size = features.shape[0]
                laminate_config = features[:, :256].view(batch_size, 4, 64)
                loading_history = features[:, 256:356]
                physics_inputs = features[:, :6]
                
                outputs = model.predict_delamination(
                    laminate_config, loading_history,
                    physics_inputs=physics_inputs,
                    meso_data=xray.expand(-1, 3, -1, -1)
                )
                
                pred = outputs['delamination_area']
                val_losses.append(criterion(pred, target).item())
        
        avg_val_loss = np.mean(val_losses) if val_losses else avg_train_loss
        history['val_loss'].append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < history['best_val_loss']:
            history['best_val_loss'] = avg_val_loss
            history['best_epoch'] = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': model_config
            }, checkpoint_dir / 'best_model.pt')
            logger.info(f"Epoch {epoch+1:3d} | NEW BEST | Val Loss: {avg_val_loss:.8f}")
        
        # Periodic checkpoint
        if (epoch + 1) % CONFIG['checkpoint_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
                f"Train: {avg_train_loss:.8f} | "
                f"Val: {avg_val_loss:.8f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )
    
    # Final save
    logger.info("-" * 70)
    logger.info(f"Training complete!")
    logger.info(f"Best val loss: {history['best_val_loss']:.8f} at epoch {history['best_epoch']+1}")
    
    torch.save({
        'epoch': CONFIG['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, checkpoint_dir / 'final_overnight_model.pt')
    
    with open(checkpoint_dir / 'overnight_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Models saved to: {checkpoint_dir}")
    logger.info(f"Finished: {datetime.now()}")
    
    return model, history


if __name__ == "__main__":
    model, history = train_overnight()
