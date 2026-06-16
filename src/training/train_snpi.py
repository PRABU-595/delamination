import torch
import yaml
from torch.utils.data import DataLoader
from src.models.snpi_net.snpi_net import SNPINet
from src.data.data_loader import DelaminationDataset
from src.training.losses import PhysicsInformedLoss
from src.utils.logger import setup_logger

def train_snpi_net(config_path):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup
    logger = setup_logger('snpi_training')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    train_dataset = DelaminationDataset('data/processed', split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    # Model
    model = SNPINet(config['snpi_net']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    criterion = PhysicsInformedLoss(config['training']['loss_weights'])
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch['features'].to(device))
            
            # Compute loss
            loss = criterion(
                outputs,
                batch['targets'].to(device),
                batch['physics_params'].to(device)
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'experiments/checkpoints/snpi_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train_snpi_net('config/model_config.yaml')
