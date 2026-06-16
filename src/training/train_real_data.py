"""
Train Delamination Model on NASA CFRP Real Data.

Uses only real experimental data from NASA PCoE Composites dataset.
No synthetic data generation.
"""
import sys
from pathlib import Path
# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io
from PIL import Image
import torchvision.transforms as T
from datetime import datetime
import json


class NASARealDataset(Dataset):
    """
    Load real NASA CFRP data only.
    
    Sources:
    - PZT signals from MAT files (features)
    - X-ray images (meso-scale visual features)
    - Strain gage data (target: stiffness degradation)
    """
    
    def __init__(self, root_dir: str, split: str = 'train'):
        self.root_dir = Path(root_dir)
        self.split = split
        self.samples = []
        
        # Find all sample directories (L2_S11_F, L2_S17_F, etc.)
        composites_dir = self.root_dir / "2. Composites"
        
        if not composites_dir.exists():
            print(f"WARNING: {composites_dir} not found")
            return
        
        sample_dirs = [d for d in composites_dir.iterdir() 
                      if d.is_dir() and d.name.startswith(('L2_', 'L3_'))]
        
        print(f"Found {len(sample_dirs)} sample directories")
        
        # Collect all MAT files from each sample
        for sample_dir in sorted(sample_dirs):
            pzt_dir = sample_dir / "PZT-data"
            xray_dir = sample_dir / "XRays"
            strain_dir = sample_dir / "StrainData"
            
            if pzt_dir.exists():
                mat_files = sorted(list(pzt_dir.glob("*.mat")))
                for mat_file in mat_files:
                    self.samples.append({
                        'mat_file': mat_file,
                        'xray_dir': xray_dir if xray_dir.exists() else None,
                        'strain_dir': strain_dir if strain_dir.exists() else None,
                        'sample_id': sample_dir.name
                    })
        
        print(f"Total MAT files found: {len(self.samples)}")
        
        # Split 80/20 train/val
        n = len(self.samples)
        split_idx = int(0.8 * n)
        
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        print(f"{split} split: {len(self.samples)} samples")
        
        # Image transforms
        self.img_transform = T.Compose([
            T.Resize((64, 64)),  # Reduced for faster training
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def _parse_mat_file(self, mat_path: Path) -> dict:
        """Parse NASA MAT file structure."""
        try:
            mat = scipy.io.loadmat(str(mat_path))
            
            features = np.zeros(2048, dtype=np.float32)
            target = 0.0
            
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
                
                # Extract target (stiffness degradation)
                if 'straingage_data' in coupon.dtype.names:
                    sg = coupon['straingage_data'][0, 0]
                    if 'stiffness_degradation' in sg.dtype.names:
                        deg = sg['stiffness_degradation']
                        if deg.size > 0 and np.issubdtype(deg.dtype, np.number):
                            target = float(deg.flat[0])
            
            return {'features': features, 'target': target}
        
        except Exception as e:
            print(f"Error parsing {mat_path.name}: {e}")
            return {'features': np.zeros(2048, dtype=np.float32), 'target': 0.0}
    
    def _load_xray(self, xray_dir: Path, mat_name: str) -> torch.Tensor:
        """Load corresponding X-ray image."""
        if xray_dir is None:
            return torch.zeros(1, 64, 64)
        
        # Try to match image name from MAT filename
        # e.g., L2S11_10000.mat -> L2S11_10000.jpg
        base_name = mat_name.replace('.mat', '')
        
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            img_path = xray_dir / f"{base_name}{ext}"
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert('L')  # Grayscale
                    return self.img_transform(img)
                except Exception:
                    pass
        
        # Try any image in the directory
        for img_file in xray_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif']:
                try:
                    img = Image.open(img_file).convert('L')
                    return self.img_transform(img)
                except Exception:
                    continue
        
        return torch.zeros(1, 64, 64)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        mat_path = sample_info['mat_file']
        
        # Parse MAT file
        parsed = self._parse_mat_file(mat_path)
        
        # Load X-ray image
        xray = self._load_xray(sample_info['xray_dir'], mat_path.name)
        
        # Feature tensor
        features = torch.from_numpy(parsed['features'])
        
        # Target tensor (stiffness degradation or delamination area proxy)
        target = torch.tensor([parsed['target']], dtype=torch.float32)
        
        return {
            'features': features,       # [2048] PZT signals
            'xray': xray,               # [1, 64, 64] X-ray image
            'target': target,           # [1] stiffness degradation
            'sample_id': sample_info['sample_id']
        }


def train_on_real_data(data_root: str, epochs: int = 50, batch_size: int = 4, 
                       learning_rate: float = 1e-4, save_dir: str = None):
    """
    Train the delamination framework on real NASA data only.
    """
    print("=" * 60)
    print("TRAINING ON REAL NASA CFRP DATA")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    
    # Create datasets
    print("\n[1] Loading real data...")
    train_dataset = NASARealDataset(data_root, split='train')
    val_dataset = NASARealDataset(data_root, split='val')
    
    if len(train_dataset) == 0:
        print("ERROR: No training samples found!")
        return None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0)
    
    print(f"    Train samples: {len(train_dataset)}")
    print(f"    Val samples: {len(val_dataset)}")
    
    # Initialize model
    print("\n[2] Initializing model...")
    from src.models.integrated.framework import IntegratedDelaminationFramework
    
    config = {
        'snpi_net': {
            'adaptive_kernel': {'input_dim': 6}
        },
        'cad_former': {
            'd_model': 128,  # Smaller for faster training
            'n_layers': 2
        },
        'al_vtfd': {}
    }
    
    model = IntegratedDelaminationFramework(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    
    # Setup save directory
    if save_dir is None:
        save_dir = Path(__file__).parent / "checkpoints"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n[3] Training...")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(device)
            xray = batch['xray'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            
            # Prepare inputs for framework
            # Create dummy laminate config and loading history from features
            batch_size_actual = features.shape[0]
            laminate_config = features[:, :256].view(batch_size_actual, 4, 64)
            loading_history = features[:, 256:356]  # 100-dim history
            
            # Physics inputs from remaining features
            physics_inputs = features[:, :6]
            
            # Forward pass
            outputs = model.predict_delamination(
                laminate_config, 
                loading_history,
                physics_inputs=physics_inputs,
                meso_data=xray.expand(-1, 3, -1, -1)  # Convert grayscale to 3-channel
            )
            
            # Compute loss
            pred = outputs['delamination_area']
            loss = criterion(pred, target)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        scheduler.step()
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                xray = batch['xray'].to(device)
                target = batch['target'].to(device)
                
                batch_size_actual = features.shape[0]
                laminate_config = features[:, :256].view(batch_size_actual, 4, 64)
                loading_history = features[:, 256:356]
                physics_inputs = features[:, :6]
                
                outputs = model.predict_delamination(
                    laminate_config, 
                    loading_history,
                    physics_inputs=physics_inputs,
                    meso_data=xray.expand(-1, 3, -1, -1)
                )
                
                pred = outputs['delamination_area']
                loss = criterion(pred, target)
                val_losses.append(loss.item())
        
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
            }, save_dir / 'best_model.pt')
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
    
    print("-" * 60)
    print(f"\nTraining complete!")
    print(f"Best validation loss: {history['best_val_loss']:.6f} (epoch {history['best_epoch']+1})")
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, save_dir / 'final_model.pt')
    
    # Save history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Models saved to: {save_dir}")
    
    return model, history


if __name__ == "__main__":
    # Path to NASA CFRP data
    DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "raw" / "NASA_CFRP"
    
    print(f"Data root: {DATA_ROOT}")
    
    model, history = train_on_real_data(
        data_root=str(DATA_ROOT),
        epochs=50,
        batch_size=4,
        learning_rate=1e-4
    )
