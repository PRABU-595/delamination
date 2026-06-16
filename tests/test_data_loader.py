import torch
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from src.data.nasa_loader import NASACompositesDataset

@pytest.fixture
def mock_config():
    return {
        'nasa_dataset': {
            'root_dir': 'dummy/path'
        }
    }

@patch('builtins.open', new_callable=mock_open, read_data="data: test")
@patch('src.data.nasa_loader.yaml.safe_load')
@patch('src.data.nasa_loader.Path')
def test_dataset_initialization(mock_path, mock_yaml, mock_file, mock_config):
    """Verify dataset finds files and splits them correctly."""
    mock_yaml.return_value = mock_config
    
    # Mock filesystem
    mock_files = []
    for i in range(5):
        m = MagicMock()
        m.__lt__ = lambda self, other: id(self) < id(other) # Simple sort
        mock_files.append(m)
        
    mock_path.return_value.rglob.return_value = mock_files
    mock_path.return_value.exists.return_value = True
    
    # Test Train Split (80%)
    dataset_train = NASACompositesDataset("config.yaml", split='train')
    assert len(dataset_train) == 4 # 5 * 0.8 = 4
    
    # Test Test Split (20%)
    dataset_test = NASACompositesDataset("config.yaml", split='test')
    assert len(dataset_test) == 1 # 5 * 0.2 = 1

@patch('builtins.open', new_callable=mock_open, read_data="data: test")
@patch('src.data.nasa_loader.scipy.io.loadmat')
@patch('src.data.nasa_loader.yaml.safe_load')
@patch('src.data.nasa_loader.Path')
def test_getitem_error_handling(mock_path, mock_yaml, mock_loadmat, mock_file, mock_config):
    """Verify robust error handling when MAT file is corrupt."""
    mock_yaml.return_value = mock_config
    
    # Mock multiple files to satisfy split logic
    mock_files = []
    for i in range(5):
        m = MagicMock()
        m.name = f"Corrupt_{i}.mat"
        m.__lt__ = lambda self, other: True
        mock_files.append(m)
    
    mock_path.return_value.rglob.return_value = mock_files
    mock_path.return_value.exists.return_value = True
    
    # Simulate extraction error
    mock_loadmat.side_effect = Exception("Corrupt File")
    
    dataset = NASACompositesDataset("config.yaml")
    sample = dataset[0]
    
    # Validation
    assert isinstance(sample['features'], torch.Tensor)
    assert torch.all(sample['features'] == 0)
    assert "Corrupt File" in sample['metadata']
    assert sample['targets'].item() == 0.0
