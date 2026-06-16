"""
Cohesive Zone Model (CZM) Implementation for Delamination Simulation.

Implements bilinear traction-separation laws for Mode I, Mode II,
and mixed-mode delamination following the Benzeggagh-Kenane criterion.

References:
    - Camanho, P.P., Davila, C.G., "Mixed-Mode Decohesion Elements"
    - Turon, A., et al., "Accurate simulation of delamination growth"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


def bilinear_traction_mode_i(delta: torch.Tensor,
                              delta_0: float = 0.003e-3,
                              delta_f: float = 0.06e-3,
                              sigma_max: float = 60e6) -> Dict[str, torch.Tensor]:
    """
    Bilinear traction-separation law for pure Mode I (opening).
    
    Traction:
        σ = K * δ           for δ ≤ δ_0  (elastic)
        σ = σ_max * (δ_f - δ) / (δ_f - δ_0)  for δ_0 < δ < δ_f  (softening)
        σ = 0                for δ ≥ δ_f  (fully damaged)
    
    Args:
        delta: Opening displacement [batch] or [batch, n_points]
        delta_0: Damage initiation displacement [m]
        delta_f: Complete failure displacement [m]
        sigma_max: Maximum cohesive traction [Pa]
        
    Returns:
        Dict with:
            'traction': Cohesive traction [same shape as delta]
            'damage': Damage variable d ∈ [0, 1]
            'G_dissipated': Energy dissipated per unit area
    """
    K = sigma_max / delta_0  # Initial interface stiffness
    
    # Damage variable
    d = torch.clamp((delta - delta_0) / (delta_f - delta_0 + 1e-12), 0.0, 1.0)
    
    # Traction
    elastic = K * delta
    softening = sigma_max * (delta_f - delta) / (delta_f - delta_0 + 1e-12)
    
    traction = torch.where(
        delta <= delta_0,
        elastic,
        torch.where(delta < delta_f, softening, torch.zeros_like(delta))
    )
    traction = torch.clamp(traction, min=0.0)
    
    # Energy dissipated = area under the curve up to current delta
    # G_Ic = 0.5 * sigma_max * delta_f (total fracture energy)
    G_Ic = 0.5 * sigma_max * delta_f
    G_dissipated = d * G_Ic
    
    return {
        'traction': traction,
        'damage': d,
        'G_dissipated': G_dissipated,
        'G_Ic': G_Ic,
        'stiffness': K
    }


def bilinear_traction_mode_ii(delta_s: torch.Tensor,
                               delta_s0: float = 0.006e-3,
                               delta_sf: float = 0.12e-3,
                               tau_max: float = 90e6) -> Dict[str, torch.Tensor]:
    """
    Bilinear traction-separation law for Mode II (sliding/shear).
    
    Args:
        delta_s: Sliding displacement [batch]
        delta_s0: Shear damage initiation displacement [m]
        delta_sf: Shear complete failure displacement [m]
        tau_max: Maximum shear traction [Pa]
        
    Returns:
        Dict with traction, damage, and dissipated energy
    """
    K_s = tau_max / delta_s0
    
    delta_s_abs = torch.abs(delta_s)
    d = torch.clamp((delta_s_abs - delta_s0) / (delta_sf - delta_s0 + 1e-12), 0.0, 1.0)
    
    elastic = K_s * delta_s
    softening_mag = tau_max * (delta_sf - delta_s_abs) / (delta_sf - delta_s0 + 1e-12)
    softening = torch.sign(delta_s) * softening_mag
    
    traction = torch.where(
        delta_s_abs <= delta_s0,
        elastic,
        torch.where(delta_s_abs < delta_sf, softening, torch.zeros_like(delta_s))
    )
    
    G_IIc = 0.5 * tau_max * delta_sf
    G_dissipated = d * G_IIc
    
    return {
        'traction': traction,
        'damage': d,
        'G_dissipated': G_dissipated,
        'G_IIc': G_IIc,
        'stiffness': K_s
    }


def mixed_mode_damage(delta_n: torch.Tensor,
                       delta_s: torch.Tensor,
                       G_Ic: float = 0.28e3,
                       G_IIc: float = 0.79e3,
                       sigma_max: float = 60e6,
                       tau_max: float = 90e6,
                       eta: float = 1.45) -> Dict[str, torch.Tensor]:
    """
    Mixed-mode cohesive zone model using the Benzeggagh-Kenane criterion.
    
    Damage initiation (Quadratic interaction):
        (σ_n/σ_max)² + (τ_s/τ_max)² = 1
    
    Damage evolution (B-K criterion):
        G_c = G_Ic + (G_IIc - G_Ic) * (G_II / G_T)^η
    
    Args:
        delta_n: Normal (opening) displacement [batch]
        delta_s: Shear (sliding) displacement [batch]
        G_Ic: Mode I fracture toughness [J/m²]
        G_IIc: Mode II fracture toughness [J/m²]
        sigma_max: Mode I interface strength [Pa]
        tau_max: Mode II interface strength [Pa]
        eta: B-K mixed-mode interaction parameter
        
    Returns:
        Dict with mixed-mode damage, tractions, and energy
    """
    K_n = sigma_max**2 / (2 * G_Ic + 1e-12)
    K_s = tau_max**2 / (2 * G_IIc + 1e-12)
    
    # Effective displacement
    delta_m = torch.sqrt(torch.clamp(delta_n, min=0.0)**2 + delta_s**2 + 1e-12)
    
    # Mode mixity
    beta = torch.abs(delta_s) / (torch.abs(delta_n) + torch.abs(delta_s) + 1e-12)
    
    # Mixed-mode initiation displacement
    K_eff = K_n * (1 - beta**2) + K_s * beta**2
    sigma_eff = torch.sqrt(sigma_max**2 * (1 - beta**2) + tau_max**2 * beta**2)
    delta_0m = sigma_eff / (K_eff + 1e-12)
    
    # Mixed-mode fracture energy (B-K)
    G_c = G_Ic + (G_IIc - G_Ic) * beta**eta
    
    # Failure displacement
    delta_fm = 2 * G_c / (sigma_eff + 1e-12)
    
    # Damage variable
    d = torch.clamp(
        delta_fm * (delta_m - delta_0m) / (delta_m * (delta_fm - delta_0m) + 1e-12),
        0.0, 1.0
    )
    
    # Apply only when delta_m > delta_0m
    d = torch.where(delta_m > delta_0m, d, torch.zeros_like(d))
    
    # Tractions
    traction_n = (1 - d) * K_n * delta_n
    traction_n = torch.where(delta_n >= 0, traction_n, K_n * delta_n)  # No damage in compression
    traction_s = (1 - d) * K_s * delta_s
    
    # Energy
    G_dissipated = d * G_c
    
    return {
        'traction_n': traction_n,
        'traction_s': traction_s,
        'damage': d,
        'mode_mixity': beta,
        'G_c': G_c,
        'G_dissipated': G_dissipated,
        'delta_m': delta_m
    }


def traction_separation(delta: torch.Tensor,
                         mode: str = 'mixed',
                         **kwargs) -> Dict[str, torch.Tensor]:
    """
    Unified traction-separation interface.
    
    Args:
        delta: Displacement tensor.
            For 'mode_i': [batch] or [batch, n_points] — normal opening
            For 'mode_ii': [batch] — shear sliding
            For 'mixed': [batch, 2] — [normal, shear] columns
        mode: 'mode_i', 'mode_ii', or 'mixed'
        **kwargs: Parameters passed to the specific CZM function
        
    Returns:
        Result dict from the corresponding CZM function
    """
    if mode == 'mode_i':
        return bilinear_traction_mode_i(delta, **kwargs)
    elif mode == 'mode_ii':
        return bilinear_traction_mode_ii(delta, **kwargs)
    elif mode == 'mixed':
        if delta.dim() == 1:
            delta_n = delta
            delta_s = torch.zeros_like(delta)
        else:
            delta_n = delta[:, 0]
            delta_s = delta[:, 1]
        return mixed_mode_damage(delta_n, delta_s, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'mode_i', 'mode_ii', or 'mixed'")


class CohesiveElement(nn.Module):
    """
    Differentiable cohesive zone element for integration in neural network training.
    
    Wraps the mixed-mode CZM as a trainable constraint layer where
    material properties can be learned from data.
    """
    
    def __init__(self, 
                 G_Ic: float = 0.28e3,
                 G_IIc: float = 0.79e3,
                 sigma_max: float = 60e6,
                 tau_max: float = 90e6,
                 eta: float = 1.45,
                 learnable: bool = False):
        super().__init__()
        
        if learnable:
            self.log_G_Ic = nn.Parameter(torch.tensor(np.log(G_Ic)))
            self.log_G_IIc = nn.Parameter(torch.tensor(np.log(G_IIc)))
            self.log_sigma_max = nn.Parameter(torch.tensor(np.log(sigma_max)))
            self.log_tau_max = nn.Parameter(torch.tensor(np.log(tau_max)))
            self.eta = nn.Parameter(torch.tensor(eta))
        else:
            self.register_buffer('log_G_Ic', torch.tensor(np.log(G_Ic)))
            self.register_buffer('log_G_IIc', torch.tensor(np.log(G_IIc)))
            self.register_buffer('log_sigma_max', torch.tensor(np.log(sigma_max)))
            self.register_buffer('log_tau_max', torch.tensor(np.log(tau_max)))
            self.register_buffer('eta', torch.tensor(eta))
    
    def forward(self, delta_n: torch.Tensor, delta_s: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            delta_n: Normal displacement [batch]
            delta_s: Shear displacement [batch]
            
        Returns:
            Mixed-mode CZM result dict
        """
        return mixed_mode_damage(
            delta_n, delta_s,
            G_Ic=torch.exp(self.log_G_Ic).item(),
            G_IIc=torch.exp(self.log_G_IIc).item(),
            sigma_max=torch.exp(self.log_sigma_max).item(),
            tau_max=torch.exp(self.log_tau_max).item(),
            eta=torch.clamp(self.eta, 1.0, 3.0).item()
        )
    
    def fracture_energy(self) -> Dict[str, float]:
        """Return current fracture energy parameters."""
        return {
            'G_Ic': torch.exp(self.log_G_Ic).item(),
            'G_IIc': torch.exp(self.log_G_IIc).item(),
            'sigma_max': torch.exp(self.log_sigma_max).item(),
            'tau_max': torch.exp(self.log_tau_max).item(),
            'eta': self.eta.item()
        }
