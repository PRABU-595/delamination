import sys
sys.path.append('.')

import torch
from src.data.nasa_loader import get_nasa_loader

def test_real_loader():
    config_path = 'config/data_config.yaml'
    print(f"Testing NASA Loader with config: {config_path}")
    
    loader = get_nasa_loader(config_path, batch_size=4, split='train')
    
    print(f"Dataset Size: {len(loader.dataset)}")
    
    if len(loader.dataset) == 0:
        print("Dataset is empty! Check path logic.")
        return

    # Try extracting a batch until we find non-zero features
    print("Searching for non-zero samples...")
    found = False
    for i, batch in enumerate(loader):
        features = batch['features']
        if torch.sum(features) != 0:
            print(f"Found VALID batch at index {i}")
            print(f"Feature Shape: {features.shape}")
            print(f"Sample Metadata: {batch['metadata']}")
            print(f"Feature Mean: {features.mean().item()}")
            found = True
            break
        if i > 20: # Check first 20 batches
            break
            
    if not found:
        print("Warning: First 20 batches are all zeros.")

if __name__ == "__main__":
    test_real_loader()
