import torch
import pytest
import numpy as np
from src.models.integrated.framework import IntegratedDelaminationFramework
from src.training.losses import PhysicsInformedLoss

@pytest.fixture
def mock_config():
    return {
        'snpi': {
            'adaptive_kernel': {'input_dim': 6, 'hidden_dim': 32},
            'uncertainty': {'dropout_rate': 0.1}
        },
        'cad_former': {
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'dropout': 0.1
        }
    }

@pytest.fixture
def mock_data():
    batch_size = 2
    n_interfaces = 4
    return {
        'micro': torch.randn(batch_size, 80),
        'meso': torch.randn(batch_size, 3, 32, 32),
        'macro': torch.randn(batch_size, 64),
        'history': torch.randn(batch_size, 1024),
        'laminate': {
            'ply_angles': torch.zeros(batch_size, n_interfaces, 1),
            'depths': torch.zeros(batch_size, n_interfaces, 1),
            'abd_matrix': torch.zeros(batch_size, n_interfaces, 3), # simplified shape
            'is_symmetric': torch.zeros(batch_size, n_interfaces, 1)
        }
    }

def test_integrated_forward_pass(mock_config, mock_data):
    """Verify end-to-end forward pass of the integrated framework."""
    model = IntegratedDelaminationFramework(mock_config)
    
    # Run prediction
    outputs = model(
        micro_data=mock_data['micro'],
        meso_data=mock_data['meso'],
        macro_data=mock_data['macro'],
        laminate_config=mock_data['laminate'],
        loading_history=mock_data['history']
    )
    
    # Check outputs
    assert 'delamination_area' in outputs
    assert 'growth_rate' in outputs
    assert 'uncertainty' in outputs
    assert 'migration_interface' in outputs
    
    # Check physics constraint (non-negative growth)
    assert torch.all(outputs['growth_rate'] >= 0)
    
    # Check dimensions
    batch_size = mock_data['micro'].shape[0]
    assert outputs['delamination_area'].shape == (batch_size, 1)

def test_physics_loss_computation(mock_config, mock_data):
    """Verify loss calculation with physics constraints."""
    model = IntegratedDelaminationFramework(mock_config)
    loss_fn = PhysicsInformedLoss()
    
    outputs = model(
        micro_data=mock_data['micro'],
        meso_data=mock_data['meso'],
        macro_data=mock_data['macro'],
        laminate_config=mock_data['laminate'],
        loading_history=mock_data['history']
    )
    
    targets = {
        'area': torch.abs(torch.randn(2, 1)), # Positive area target
        'growth_rate': torch.abs(torch.randn(2, 1)),
        'migration': torch.randint(0, 4, (2,)) 
    }
    
    loss, components = loss_fn(outputs, targets)
    
    assert loss.item() > 0
    assert 'mse' in components
    assert 'physics' in components
    assert 'nll' in components
