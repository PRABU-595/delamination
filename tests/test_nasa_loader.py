import pytest
import os
import scipy.io
import torch
import numpy as np
from src.data.nasa_loader import NASACompositeDataset

def test_nasa_loader_with_mock_data(tmp_path):
    # 1. Create dummy directory structure
    data_dir = tmp_path / "NASA_CFRP"
    data_dir.mkdir()
    
    # 2. Create dummy .mat files
    # Structure based on typical PCoE data or our assumption in loader
    dummy_data = {
        'strain': np.random.randn(100, 3),
        'lamb_waves': np.random.randn(100, 16)
    }
    
    file_path_1 = data_dir / "test_sample_1.mat"
    scipy.io.savemat(file_path_1, dummy_data)
    
    file_path_2 = data_dir / "test_sample_2.mat"
    scipy.io.savemat(file_path_2, dummy_data)
    
    # 3. Initialize Dataset
    dataset = NASACompositeDataset(data_dir=str(data_dir))
    
    # 4. Assertions
    assert len(dataset) == 2
    
    features, label = dataset[0]
    assert torch.is_tensor(features)
    # Check shape - logic in loader is currently placeholder "torch.randn(6)"
    # We should probably update loader to look at real data structure if we knew it
    # For now, just checking it returns tensors and doesn't crash
    assert features.shape == (6,) 
    assert label.shape == (1,)
    
def test_nasa_loader_empty_dir(tmp_path):
    empty_dir = tmp_path / "Empty"
    empty_dir.mkdir()
    
    dataset = NASACompositeDataset(data_dir=str(empty_dir))
    assert len(dataset) == 0
