import pytest
import torch
from src.models.cad_former.cad_former import CADFormer

def test_cad_former_structure():
    config = {
        'd_model': 32,
        'n_heads': 4,
        'n_layers': 2,
        'meso_cnn': {'hidden_channels': [8, 16]}, # Reduced for test
        'micro_gnn': {'node_feat_dim': 4, 'hidden_dims': [8, 16]}
    }
    model = CADFormer(config)
    
    # Mock Inputs
    batch_size = 4
    n_interfaces = 5
    
    # 1. Encoders inputs
    micro_data = torch.randn(batch_size, 4) # Fallback MLP expected format for simplicity
    meso_data = torch.randn(batch_size, 3, 32, 32)
    macro_data = torch.randn(batch_size, 64)
    
    # 2. Config inputs
    laminate_config = {
        'ply_angles': torch.randn(batch_size, n_interfaces, 1),
        'depths': torch.randn(batch_size, n_interfaces, 1),
        'abd_matrix': torch.randn(batch_size, n_interfaces, 3),
        'is_symmetric': torch.randn(batch_size, n_interfaces, 1)
    }
    
    # 3. History
    loading_history = torch.randn(batch_size, 10, 32) # [batch, seq, d_model]
    
    inputs = {
        'micro_data': micro_data,
        'meso_data': meso_data,
        'macro_data': macro_data,
        'laminate_config': laminate_config,
        'loading_history': loading_history
    }
    
    outputs = model(inputs)
    
    # assertions
    assert 'delamination_area' in outputs
    assert 'growth_rate' in outputs
    assert 'migration_probs' in outputs
    
    assert outputs['delamination_area'].shape == (batch_size, 1)
    # n_interfaces output for migration probs
    assert outputs['migration_probs'].shape == (batch_size, n_interfaces, 1)

def test_migration_predictor():
    from src.models.cad_former.migration_predictor import InterfaceMigrationPredictor
    predictor = InterfaceMigrationPredictor(feature_dim=16)
    
    batch = 5
    h_i = torch.randn(batch, 16)
    h_j = torch.randn(batch, 16)
    
    shear = torch.randn(batch, 1)
    mixity = torch.randn(batch, 1)
    density = torch.randn(batch, 1)
    angle = torch.randn(batch, 1)
    
    prob = predictor(h_i, h_j, shear, mixity, density, angle)
    assert prob.shape == (batch, 1)
    assert prob.min() >= 0 and prob.max() <= 1
