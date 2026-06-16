"""
Full Real-Data Training Pipeline.

Trains the Integrated Delamination Framework on ALL available real data:
- NASA PCoE CFRP (2,180 samples: PZT lamb wave + X-ray)
- F-MOC (6,219 samples: Acoustic Emission + DIC images)
- Total: ~8,399 real experimental samples

Features:
- 80/20 Train/Validation split with reproducible seeding
- Physics-informed loss (MSE + Heteroscedastic NLL + Thermodynamics)
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping (max_norm=1.0)
- Best model checkpointing
- TensorBoard-compatible logging
- Early stopping (patience=10)

Usage:
    .venv\\Scripts\\python.exe scripts/train_full_real.py
    .venv\\Scripts\\python.exe scripts/train_full_real.py --epochs 100 --batch_size 32
"""
import sys
import os
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from src.data.multimodal_loader import get_mega_loader
from src.models.integrated.framework import IntegratedDelaminationFramework
from src.training.losses import PhysicsInformedLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train on full real dataset")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='experiments/checkpoints/full_real',
                        help='Checkpoint save directory')
    return parser.parse_args()


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"  CPU mode (no GPU detected)")
    return device


def build_model(device):
    """Initialize the Integrated Delamination Framework."""
    config = {
        'snpi_net': {'adaptive_kernel': {'input_dim': 6}},
        'cad_former': {'d_model': 128, 'n_layers': 4},
        'al_vtfd': {}
    }
    model = IntegratedDelaminationFramework(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def load_real_data(batch_size, val_split, seed):
    """
    Load ALL real datasets via MegaLoader and split into train/val.
    Returns (train_loader, val_loader, dataset_stats).
    """
    data_root = Path("data/raw")
    
    # Load full dataset (batch_size here is just for initial loading)
    full_loader = get_mega_loader(data_root, batch_size=1)
    full_dataset = full_loader.dataset
    
    total_samples = len(full_dataset)
    n_val = int(total_samples * val_split)
    n_train = total_samples - n_val
    
    # Reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, drop_last=False)
    
    stats = {
        'total_samples': total_samples,
        'train_samples': n_train,
        'val_samples': n_val,
        'train_batches': len(train_loader),
        'val_batches': len(val_loader)
    }
    
    return train_loader, val_loader, stats


def prepare_batch(batch, device):
    """Convert a data batch into model-compatible inputs."""
    features = batch['features'].to(device)
    targets = batch['target'].to(device)
    image = batch['image'].to(device) if 'image' in batch else None
    
    # Extract structured inputs from the flat feature vector
    physics_inputs = features[:, :6]
    
    # Laminate config as tensor (framework converts internally)
    if features.shape[1] >= 262:
        laminate_config = features[:, 6:262].view(-1, 4, 64)
    else:
        laminate_config = torch.zeros(features.size(0), 4, 64, device=device)
    
    # Loading history
    if features.shape[1] >= 362:
        loading_history = features[:, 262:362]
    else:
        loading_history = torch.zeros(features.size(0), 100, device=device)
    
    return {
        'laminate_config': laminate_config,
        'loading_history': loading_history,
        'physics_inputs': physics_inputs,
        'meso_data': image,
        'targets': targets
    }


def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch):
    """Train for one epoch, return average loss and components."""
    model.train()
    total_loss = 0.0
    total_components = {}
    n_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        inputs = prepare_batch(batch, device)
        
        # Forward pass
        predictions = model(
            inputs['laminate_config'],
            inputs['loading_history'],
            physics_inputs=inputs['physics_inputs'],
            meso_data=inputs['meso_data']
        )
        
        # Build target dict
        target_dict = {
            'area': inputs['targets'],
            'growth_rate': inputs['targets'] * 0.1,  # Proxy growth rate
        }
        
        # Compute loss
        loss, components = loss_fn(predictions, target_dict)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in components.items():
            total_components[k] = total_components.get(k, 0.0) + v
        n_batches += 1
        
        # Progress indicator every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx+1}/{len(train_loader)} — Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in total_components.items()}
    
    return avg_loss, avg_components


@torch.no_grad()
def validate(model, val_loader, loss_fn, device):
    """Validate on held-out data, return average loss and metrics."""
    model.eval()
    total_loss = 0.0
    total_components = {}
    all_preds = []
    all_targets = []
    n_batches = 0
    
    for batch in val_loader:
        inputs = prepare_batch(batch, device)
        
        predictions = model(
            inputs['laminate_config'],
            inputs['loading_history'],
            physics_inputs=inputs['physics_inputs'],
            meso_data=inputs['meso_data']
        )
        
        target_dict = {
            'area': inputs['targets'],
            'growth_rate': inputs['targets'] * 0.1,
        }
        
        loss, components = loss_fn(predictions, target_dict)
        total_loss += loss.item()
        for k, v in components.items():
            total_components[k] = total_components.get(k, 0.0) + v
        n_batches += 1
        
        all_preds.append(predictions['delamination_area'].cpu())
        all_targets.append(inputs['targets'].cpu())
    
    avg_loss = total_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in total_components.items()}
    
    # Compute regression metrics
    preds = torch.cat(all_preds).numpy().flatten()
    targets = torch.cat(all_targets).numpy().flatten()
    
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    
    metrics = {
        'val_loss': avg_loss,
        'val_rmse': float(rmse),
        'val_r2': float(r2),
        'components': avg_components
    }
    
    return metrics


def main():
    args = parse_args()
    
    print("\n" + "#" * 70)
    print("# FULL REAL-DATA TRAINING PIPELINE")
    print("#" * 70)
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    
    # Save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    print("\n[1/4] Loading REAL datasets...")
    train_loader, val_loader, data_stats = load_real_data(
        args.batch_size, args.val_split, args.seed
    )
    print(f"  Train: {data_stats['train_samples']} samples ({data_stats['train_batches']} batches)")
    print(f"  Val:   {data_stats['val_samples']} samples ({data_stats['val_batches']} batches)")
    
    # Build Model
    print("\n[2/4] Building model...")
    model = build_model(device)
    
    # Loss & Optimizer
    print("\n[3/4] Configuring training...")
    loss_weights = {
        'mse': 1.0,
        'nll': 0.5,
        'migration': 0.3,
        'rcurve': 0.1,
        'physics': 0.1,
        'energy': 0.05,
        'thermo': 0.05
    }
    loss_fn = PhysicsInformedLoss(loss_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    print(f"  Optimizer:  AdamW (lr={args.lr}, wd=1e-5)")
    print(f"  Scheduler:  ReduceLROnPlateau (factor=0.5, patience=5)")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Early stop: patience={args.patience}")
    
    # Training Loop
    print("\n[4/4] Training on REAL data...")
    print("=" * 70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_log = []
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_components = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)
        
        # Scheduler step
        scheduler.step(val_metrics['val_loss'])
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['val_loss'],
            'val_rmse': val_metrics['val_rmse'],
            'val_r2': val_metrics['val_r2'],
            'lr': current_lr,
            'epoch_time_s': round(epoch_time, 1)
        }
        training_log.append(log_entry)
        
        # Print progress
        improved = val_metrics['val_loss'] < best_val_loss
        marker = " ★ BEST" if improved else ""
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | "
              f"Val: {val_metrics['val_loss']:.4f} | "
              f"RMSE: {val_metrics['val_rmse']:.4f} | "
              f"R²: {val_metrics['val_r2']:+.4f} | "
              f"LR: {current_lr:.1e} | "
              f"{epoch_time:.1f}s{marker}")
        
        # Save best model
        if improved:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_rmse': val_metrics['val_rmse'],
                'val_r2': val_metrics['val_r2'],
                'config': {
                    'snpi_net': {'adaptive_kernel': {'input_dim': 6}},
                    'cad_former': {'d_model': 128, 'n_layers': 4},
                }
            }
            best_path = save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
            # Also save to the standard checkpoint location
            std_ckpt_dir = Path("src/training/checkpoints/mega_run")
            std_ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), std_ckpt_dir / "mega_best_model.pt")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n  ⚠️  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break
    
    total_time = time.time() - start_time
    
    # Final Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time:        {total_time/60:.1f} minutes")
    print(f"  Best epoch:        {training_log[np.argmin([l['val_loss'] for l in training_log])]['epoch']}")
    print(f"  Best val loss:     {best_val_loss:.4f}")
    
    best_entry = min(training_log, key=lambda x: x['val_loss'])
    print(f"  Best val RMSE:     {best_entry['val_rmse']:.4f}")
    print(f"  Best val R²:       {best_entry['val_r2']:+.4f}")
    print(f"  Data used:         {data_stats['total_samples']} real samples")
    print(f"  Checkpoint saved:  {save_dir / 'best_model.pt'}")
    
    # Save training log
    log_file = save_dir / "training_log.json"
    with open(log_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'data_stats': data_stats,
            'training_log': training_log,
            'final_metrics': {
                'best_val_loss': best_val_loss,
                'best_val_rmse': best_entry['val_rmse'],
                'best_val_r2': best_entry['val_r2'],
                'total_time_minutes': round(total_time / 60, 1),
                'total_epochs_run': len(training_log)
            }
        }, f, indent=2)
    print(f"  Training log:      {log_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
