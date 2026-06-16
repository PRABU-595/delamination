"""
Fracture Mechanics Implementation for Composite Delamination.

Implements energy release rate computation, R-curve modeling,
fatigue delamination growth laws, and VCCT methods.

References:
    - Irwin, G.R., "Analysis of stresses and strains near the end of a crack"
    - Hojo, M., et al., "Effect of stress ratio on near-threshold propagation"
    - Rybicki, E.F., Kanninen, M.F., "VCCT for fracture mechanics analysis"
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


def benzeggagh_kenane_criterion(G_I: torch.Tensor,
                                 G_II: torch.Tensor,
                                 G_Ic: float = 0.28e3,
                                 G_IIc: float = 0.79e3,
                                 eta: float = 1.45) -> torch.Tensor:
    """
    Benzeggagh-Kenane mixed-mode fracture criterion.
    
    G_c = G_Ic + (G_IIc - G_Ic) * (G_II / G_T)^η
    
    Failure when G_T ≥ G_c.
    
    Args:
        G_I: Mode I energy release rate [J/m²]
        G_II: Mode II energy release rate [J/m²]
        G_Ic: Mode I fracture toughness
        G_IIc: Mode II fracture toughness
        eta: B-K interaction parameter (material-dependent, typically 1.0-2.5)
        
    Returns:
        G_c: Mixed-mode fracture toughness
    """
    G_total = G_I + G_II + 1e-12
    ratio = G_II / G_total
    G_c = G_Ic + (G_IIc - G_Ic) * torch.pow(ratio, eta)
    return G_c


def power_law_criterion(G_I: torch.Tensor,
                         G_II: torch.Tensor,
                         G_Ic: float = 0.28e3,
                         G_IIc: float = 0.79e3,
                         alpha: float = 1.0,
                         beta: float = 1.0) -> torch.Tensor:
    """
    Power law mixed-mode fracture criterion.
    
    (G_I / G_Ic)^α + (G_II / G_IIc)^β = 1 at failure
    
    Returns failure index f (≥ 1 means failure).
    """
    f = (G_I / G_Ic)**alpha + (G_II / G_IIc)**beta
    return f


def paris_law(delta_G: torch.Tensor,
              C: float = 1e-12,
              m: float = 5.0) -> torch.Tensor:
    """
    Paris law for fatigue delamination growth.
    
    da/dN = C * (ΔG)^m
    
    where ΔG = G_max - G_min (energy release rate range).
    
    Args:
        delta_G: Energy release rate range [J/m²]
        C: Paris law coefficient (material constant)
        m: Paris law exponent (typically 3-8 for composites)
        
    Returns:
        da_dN: Crack growth rate per cycle [m/cycle]
    """
    delta_G = torch.clamp(delta_G, min=1e-12)
    da_dN = C * torch.pow(delta_G, m)
    return da_dN


def modified_paris_law(G_max: torch.Tensor,
                        R_ratio: float = 0.1,
                        G_th: float = 50.0,
                        G_c: float = 280.0,
                        C: float = 1e-12,
                        m: float = 5.0) -> torch.Tensor:
    """
    Modified Paris law with threshold and instability effects.
    
    da/dN = C * (ΔG_eff)^m / [(1 - G_th/G_max) * (1 - G_max/G_c)]
    
    This captures:
    - Threshold behavior: no growth below G_th
    - Instability: rapid growth as G_max → G_c
    - Stress ratio effects via R
    
    Args:
        G_max: Maximum energy release rate [J/m²]
        R_ratio: Stress ratio (G_min/G_max)
        G_th: Threshold energy release rate [J/m²]
        G_c: Critical fracture toughness [J/m²]
        C: Paris law coefficient
        m: Paris law exponent
        
    Returns:
        da_dN: Crack growth rate [m/cycle]
    """
    delta_G = G_max * (1 - R_ratio)
    delta_G = torch.clamp(delta_G, min=1e-12)
    
    # Threshold correction
    threshold_term = torch.clamp(1.0 - G_th / (G_max + 1e-12), min=1e-12)
    
    # Instability correction
    instability_term = torch.clamp(1.0 - G_max / (G_c + 1e-12), min=1e-12)
    
    da_dN = C * torch.pow(delta_G, m) / (threshold_term * instability_term)
    
    # Below threshold: no growth
    da_dN = torch.where(G_max > G_th, da_dN, torch.zeros_like(da_dN))
    
    return da_dN


def r_curve(crack_length: torch.Tensor,
            G_Ic_init: float = 200.0,
            G_Ic_ss: float = 350.0,
            a_bridge: float = 5e-3,
            model: str = 'exponential') -> torch.Tensor:
    """
    R-curve modeling: fracture resistance as a function of crack length.
    
    The R-curve captures fiber bridging effects where fracture resistance
    increases with crack extension until reaching a steady-state plateau.
    
    Models:
        exponential: G_R(Δa) = G_ss - (G_ss - G_init) * exp(-Δa / a_bridge)
        power_law: G_R(Δa) = G_init + (G_ss - G_init) * (1 - exp(-(Δa/a_bridge)^n))
    
    Args:
        crack_length: Crack extension Δa from initial tip [m]
        G_Ic_init: Initiation fracture toughness [J/m²]
        G_Ic_ss: Steady-state fracture toughness [J/m²]
        a_bridge: Characteristic bridging length [m]
        model: 'exponential' or 'power_law'
        
    Returns:
        G_R: Fracture resistance [J/m²]
    """
    delta_a = torch.clamp(crack_length, min=0.0)
    
    if model == 'exponential':
        G_R = G_Ic_ss - (G_Ic_ss - G_Ic_init) * torch.exp(-delta_a / (a_bridge + 1e-12))
    elif model == 'power_law':
        n = 0.7
        G_R = G_Ic_init + (G_Ic_ss - G_Ic_init) * (
            1.0 - torch.exp(-(delta_a / (a_bridge + 1e-12))**n)
        )
    else:
        raise ValueError(f"Unknown R-curve model: {model}")
    
    return G_R


def fiber_bridging_degradation(N_cycles: torch.Tensor,
                                 sigma_bridge_0: float = 50e6,
                                 N_degrade: float = 1e5,
                                 alpha: float = 0.3) -> torch.Tensor:
    """
    Model fiber bridging degradation under fatigue loading.
    
    σ_bridge(N) = σ_bridge_0 * exp(-α * (N / N_degrade))
    
    As bridges degrade, R-curve effect reduces → G_R approaches G_init.
    
    Args:
        N_cycles: Number of fatigue cycles
        sigma_bridge_0: Initial bridging stress [Pa]
        N_degrade: Characteristic degradation cycles
        alpha: Degradation rate parameter
        
    Returns:
        sigma_bridge: Remaining bridging stress [Pa]
    """
    return sigma_bridge_0 * torch.exp(-alpha * N_cycles / (N_degrade + 1e-12))


def vcct_energy_release_rate(forces: torch.Tensor,
                              displacements: torch.Tensor,
                              element_width: float = 1e-3,
                              element_length: float = 1e-3) -> Dict[str, torch.Tensor]:
    """
    Virtual Crack Closure Technique (VCCT) for energy release rate computation.
    
    G_I = F_y * Δv / (2 * b * Δa)
    G_II = F_x * Δu / (2 * b * Δa)
    
    Args:
        forces: Nodal forces at crack tip [batch, 2] (F_x, F_y)
        displacements: Relative displacements behind tip [batch, 2] (Δu, Δv)
        element_width: Element width b [m]
        element_length: Element length Δa [m]
        
    Returns:
        Dict with G_I, G_II, G_total, and mode_mixity
    """
    F_x = forces[:, 0]
    F_y = forces[:, 1]
    delta_u = displacements[:, 0]
    delta_v = displacements[:, 1]
    
    area = 2 * element_width * element_length
    
    G_I = torch.abs(F_y * delta_v) / (area + 1e-12)
    G_II = torch.abs(F_x * delta_u) / (area + 1e-12)
    
    G_total = G_I + G_II
    mode_mixity = G_II / (G_total + 1e-12)
    
    return {
        'G_I': G_I,
        'G_II': G_II,
        'G_total': G_total,
        'mode_mixity': mode_mixity
    }


def stress_intensity_factor(G: torch.Tensor,
                             E: float = 135e9,
                             nu: float = 0.3,
                             plane: str = 'strain') -> torch.Tensor:
    """
    Compute stress intensity factor K from energy release rate G.
    
    Plane stress:  K = sqrt(G * E)
    Plane strain:  K = sqrt(G * E / (1 - ν²))
    
    Args:
        G: Energy release rate [J/m²]
        E: Young's modulus [Pa]
        nu: Poisson's ratio
        plane: 'stress' or 'strain'
        
    Returns:
        K: Stress intensity factor [Pa·√m]
    """
    if plane == 'stress':
        K = torch.sqrt(torch.clamp(G * E, min=0.0))
    elif plane == 'strain':
        K = torch.sqrt(torch.clamp(G * E / (1 - nu**2), min=0.0))
    else:
        raise ValueError(f"Unknown plane condition: {plane}")
    
    return K


def compliance_method_G(load: torch.Tensor,
                         compliance: torch.Tensor,
                         dC_da: torch.Tensor,
                         width: float = 0.025) -> torch.Tensor:
    """
    Energy release rate from compliance method (Irwin-Kies).
    
    G = P² / (2b) * dC/da
    
    Used for DCB, ENF, and MMB test data reduction.
    
    Args:
        load: Applied load P [N]
        compliance: Specimen compliance C [m/N]
        dC_da: Rate of change of compliance with crack length [1/(N·m)]
        width: Specimen width b [m]
        
    Returns:
        G: Energy release rate [J/m²]
    """
    G = load**2 / (2 * width + 1e-12) * torch.abs(dC_da)
    return G


def fatigue_life_prediction(G_max_seq: torch.Tensor,
                             C: float = 1e-12,
                             m: float = 5.0,
                             G_th: float = 50.0,
                             G_c: float = 280.0,
                             R_ratio: float = 0.1,
                             a_init: float = 25e-3,
                             a_crit: float = 75e-3,
                             dN: int = 100) -> Dict[str, torch.Tensor]:
    """
    Predict fatigue delamination growth history using cycle-by-cycle integration.
    
    Integrates the modified Paris law over loading cycles to predict
    crack length as a function of cycle count.
    
    Args:
        G_max_seq: Sequence of max energy release rates per block [n_blocks]
        C, m, G_th, G_c, R_ratio: Paris law parameters
        a_init: Initial crack length [m]
        a_crit: Critical crack length for failure [m]
        dN: Cycle increment per integration step
        
    Returns:
        Dict with:
            'crack_length': Crack length history [n_steps]
            'cycles': Cycle count history [n_steps]
            'growth_rate': Growth rate history [n_steps]
    """
    a = a_init
    N = 0
    
    crack_history = [a]
    cycle_history = [0]
    rate_history = [0.0]
    
    n_blocks = len(G_max_seq)
    block_idx = 0
    
    max_steps = 10000
    for step in range(max_steps):
        G_max = G_max_seq[block_idx % n_blocks]
        
        da_dN = modified_paris_law(
            G_max.unsqueeze(0), R_ratio, G_th, G_c, C, m
        ).item()
        
        a += da_dN * dN
        N += dN
        
        crack_history.append(a)
        cycle_history.append(N)
        rate_history.append(da_dN)
        
        if a >= a_crit:
            break
        
        block_idx += 1
    
    return {
        'crack_length': torch.tensor(crack_history),
        'cycles': torch.tensor(cycle_history),
        'growth_rate': torch.tensor(rate_history),
        'failed': a >= a_crit,
        'final_cycles': N
    }
