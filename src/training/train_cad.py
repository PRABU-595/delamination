import torch
import yaml
import os
from torch.utils.data import DataLoader
from src.models.cad_former.cad_former import CADFormer
from src.data.nasa_loader import NASACompositesDataset
from src.training.losses import PhysicsInformedLoss
from src.utils.logger import setup_logger

def train_cad_former(config_path):
    # Load config
    if not os.path.exists(config_path):
        # Fallback to default if config missing
        config = {
            'training': {
                'batch_size': 16,
                'learning_rate': 1e-4,
                'epochs': 50,
                'loss_weights': {'mse': 1.0, 'physics': 0.1}
            },
            'cad_former': {
                'd_model': 256,
                'n_heads': 8,
                'n_layers': 6
            },
            'dataset': {
                'path': 'data/nasa_composites'
            }
        }
    else:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    # Setup
    logger = setup_logger('cad_training')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on {device}")
    
    # Data
    # Assuming config points to a valid dataset config or path
    # For demo purposes, we might use the NASA loader
    train_dataset = NASACompositesDataset('config/dataset_config.yaml', split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    # Model
    model = CADFormer(
        d_model=config['cad_former']['d_model'],
        n_heads=config['cad_former']['n_heads'],
        n_layers=config['cad_former']['n_layers']
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Prepare inputs
            micro = batch['micro_features'].to(device) if 'micro_features' in batch else torch.randn(len(batch), 80).to(device) # Placeholder mapping
            meso = batch['meso_features'].to(device) if 'meso_features' in batch else torch.randn(len(batch), 3, 32, 32).to(device)
            macro = batch['macro_features'].to(device) if 'macro_features' in batch else torch.randn(len(batch), 64).to(device)
            
            # Forward pass
            outputs = model(
                micro_data=micro,
                meso_data=meso,
                macro_data=macro,
                laminate_config=batch['laminate_config'], # Needs to be in batch
                loading_history=batch['loading_history'].to(device)
            )
            
            # Compute loss
            # Simplified loss calculation for CAD-Former specific
            loss_mse = torch.nn.functional.mse_loss(outputs['delamination_area'], batch['targets'].to(device))
            loss = loss_mse
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        logger.info(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'experiments/checkpoints/cad_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train_cad_former('config/cad_config.yaml')
