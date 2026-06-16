import torch
import pytest
import numpy as np
from src.physics import laminate_theory, cohesive_zone, fracture_mechanics, peridynamics

def test_laminate_theory_abd():
    """Verify ABD matrix computation for a symmetric laminate."""
    # [0, 90]_s laminate -> Symmetric
    stacking = [0, 90, 90, 0]
    result = laminate_theory.ABD_matrix(stacking)
    
    A, B, D = result['A'], result['B'], result['D']
    
    # Check shape
    assert A.shape == (3, 3)
    assert B.shape == (3, 3)
    assert D.shape == (3, 3)
    
    # Check symmetry (B should be zero for symmetric laminate)
    assert torch.allclose(B, torch.zeros_like(B), atol=1e-4)
    assert result['is_symmetric'] is True
    
    # Check A11 > 0, D11 > 0
    assert A[0, 0] > 0
    assert D[0, 0] > 0

def test_cohesive_zone_model():
    """Verify bilinear traction-separation behavior."""
    delta_0 = 0.001
    delta_f = 0.01
    sigma_max = 50e6
    
    # Test elastic region
    d_elastic = torch.tensor([0.0005])
    res_elastic = cohesive_zone.bilinear_traction_mode_i(d_elastic, delta_0, delta_f, sigma_max)
    sigma_elastic = res_elastic['traction']
    # Check K * delta
    K = sigma_max / delta_0
    assert torch.allclose(sigma_elastic, K * d_elastic, rtol=1e-4)
    assert res_elastic['damage'].item() == 0.0
    
    # Test softening region
    d_soft = torch.tensor([0.005])
    res_soft = cohesive_zone.bilinear_traction_mode_i(d_soft, delta_0, delta_f, sigma_max)
    # Check damage is between 0 and 1
    d_val = res_soft['damage'].item()
    assert 0.0 < d_val < 1.0
    # Traction should be less than sigma_max
    assert res_soft['traction'].item() < sigma_max

def test_mixed_mode_bk():
    """Verify Benzeggagh-Kenane criterion."""
    G_Ic = 300.0
    G_IIc = 800.0
    eta = 1.5
    
    # Pure Mode I
    G_c_I = fracture_mechanics.benzeggagh_kenane_criterion(
        G_I=torch.tensor(100.0), G_II=torch.tensor(0.0), 
        G_Ic=G_Ic, G_IIc=G_IIc, eta=eta
    )
    assert torch.allclose(G_c_I, torch.tensor(G_Ic))
    
    # Pure Mode II
    G_c_II = fracture_mechanics.benzeggagh_kenane_criterion(
        G_I=torch.tensor(0.0), G_II=torch.tensor(100.0),
        G_Ic=G_Ic, G_IIc=G_IIc, eta=eta
    )
    assert torch.allclose(G_c_II, torch.tensor(G_IIc))

def test_peridynamic_kernel():
    """Verify peridynamic influence function and horizon."""
    delta = torch.tensor(1.0)
    
    # Test influence function (cubic spline)
    # r=0 -> wt=1
    w0 = peridynamics.influence_function(torch.tensor(0.0), delta, 'cubic_spline')
    assert torch.allclose(w0, torch.tensor(1.0))
    
    # r=delta -> wt=0
    wd = peridynamics.influence_function(torch.tensor(1.0), delta, 'cubic_spline')
    assert torch.allclose(wd, torch.tensor(0.0), atol=1e-6)
    
    # r>delta -> wt=0
    w_far = peridynamics.influence_function(torch.tensor(1.5), delta, 'cubic_spline')
    assert w_far.item() == 0.0

def test_paris_law_fatigue():
    """Verify modified Paris law growth rate."""
    C = 1e-10
    m = 3.0
    G_th = 50.0
    G_c = 1000.0
    
    # Case 1: Below threshold
    G_low = torch.tensor(40.0)
    da_dn_low = fracture_mechanics.modified_paris_law(G_low, G_th=G_th, G_c=G_c, C=C, m=m)
    assert da_dn_low.item() == 0.0
    
    # Case 2: Standard growth
    G_mid = torch.tensor(200.0)
    da_dn_mid = fracture_mechanics.modified_paris_law(G_mid, G_th=G_th, G_c=G_c, C=C, m=m)
    assert da_dn_mid.item() > 0.0
    
    # Case 3: Near instability (should accelerate)
    G_high = torch.tensor(900.0)
    da_dn_high = fracture_mechanics.modified_paris_law(G_high, G_th=G_th, G_c=G_c, C=C, m=m)
    
    # Instability term should increase rate
    # Simple Paris would be C * (G(1-R))^m
    # Modified has denominator < 1, so rate > simple paris
    simple_rate = C * (G_high * 0.9)**m
    assert da_dn_high.item() > simple_rate.item()
