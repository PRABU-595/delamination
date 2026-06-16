import pytest
import torch
from src.models.al_vtfd.virtual_testing import VirtualTestingFramework
from src.models.snpi_net.snpi_net import SNPINet

def test_al_framework_flow():
    # Setup dummy model
    config = {} # Defaults
    model = SNPINet(config)
    
    framework = VirtualTestingFramework(model)
    
    # Run small budget optimization
    pool = torch.randn(10, 6)
    results = framework.run_optimization('TestMaterial', budget_tests=5, candidate_pool=pool)
    
    assert results['n_tests_conducted'] <= 5
    assert 'final_accuracy' in results
    assert 'cost_savings' in results

def test_acquisition_score():
    from src.models.al_vtfd.acquisition import AcquisitionFunction
    model = SNPINet({})
    acq = AcquisitionFunction(model)
    
    candidates = torch.randn(5, 6)
    scores = acq.compute_score(candidates)
    
    assert scores.shape == (5,)
    assert not torch.any(torch.isnan(scores))
