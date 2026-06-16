import pytest
import torch
from src.models.snpi_net.snpi_net import SNPINet

def test_snpi_net_structure():
    config = {
        'adaptive_kernel': {'input_dim': 6, 'hidden_dims': [32, 32]},
        'uncertainty': {'dropout_rate': 0.1}
    }
    model = SNPINet(config)
    
    # Fake input batch [batch_size=16, features=6]
    x = torch.randn(16, 6)
    
    output = model(x, return_horizon=True)
    
    assert 'prediction' in output
    assert 'aleatoric_log_var' in output
    assert 'horizon' in output
    
    assert output['prediction'].shape == (16, 3) # default output dim
    assert output['horizon'].shape == (16, 1)

def test_uncertainty_prediction():
    config = {}
    model = SNPINet(config)
    x = torch.randn(10, 6)
    
    results = model.predict_uncertainty(x, n_samples=10)
    
    assert 'mean' in results
    assert 'epistemic' in results
    assert 'aleatoric' in results
    assert results['mean'].shape == (10, 3)

def test_multi_fidelity_gp():
    from src.models.snpi_net.multi_fidelity import MultiFidelityGP
    mf_gp = MultiFidelityGP(input_dim=6)
    x = torch.randn(5, 6)
    
    mean, std = mf_gp(x, fidelity_level='low')
    assert mean.shape == (5,)
