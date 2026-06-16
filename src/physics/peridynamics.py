"""
Peridynamic Theory Implementation for Nonlocal Damage Modeling.

Provides bond-based peridynamic formulations for composite delamination,
including nonlocal force density, adaptive horizon, and damage evolution.

References:
    - Silling, S.A., "Reformulation of elasticity theory for discontinuities"
    - Madenci, E., Oterkus, E., "Peridynamic Theory and Its Applications"
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


def bond_stretch(u_diff: torch.Tensor, x_diff: torch.Tensor) -> torch.Tensor:
    """
    Calculate bond stretch s.
    
    s = (|y' - y| - |x' - x|) / |x' - x|
    where y = u + x (deformed position)
    
    Args:
        u_diff: Displacement difference (u' - u) [batch, n_bonds, dim]
        x_diff: Reference position difference (x' - x) [batch, n_bonds, dim]
        
    Returns:
        stretch: Bond stretch values [batch, n_bonds]
    """
    orig_len = torch.norm(x_diff, dim=-1)
    deformed_len = torch.norm(x_diff + u_diff, dim=-1)
    stretch = (deformed_len - orig_len) / (orig_len + 1e-12)
    return stretch


def critical_stretch_model(G_c: torch.Tensor, 
                            horizon: torch.Tensor, 
                            bulk_modulus: torch.Tensor) -> torch.Tensor:
    """
    Compute critical stretch for bond failure.
    
    For 3D:  s_c = sqrt(5 * G_c / (9 * K * δ))
    For 2D:  s_c = sqrt(4 * π * G_c / (9 * E * δ))
    
    Using generalized form:
        s_c = sqrt(G_c / (c_dim * K * δ))
    
    Args:
        G_c: Critical energy release rate [J/m²]
        horizon: Peridynamic horizon δ [m]
        bulk_modulus: Material bulk modulus K [Pa]
        
    Returns:
        s_c: Critical stretch threshold
    """
    return torch.sqrt(G_c / (horizon * bulk_modulus + 1e-12))


def influence_function(xi_norm: torch.Tensor, 
                        delta: torch.Tensor,
                        kernel_type: str = 'cubic_spline') -> torch.Tensor:
    """
    Compute the peridynamic influence function ω(|ξ|).
    
    Physics constraints:
        1. ω(r) ≥ 0 for r < δ
        2. ω(r) = 0 for r ≥ δ (compact support)
        3. ω(r) monotonically decreasing
        4. Smooth at r = δ
    
    Args:
        xi_norm: Bond lengths |ξ| = |x' - x| [batch, n_bonds]
        delta: Horizon values [batch, 1] or scalar
        kernel_type: 'constant', 'linear', 'cubic_spline', 'gaussian'
        
    Returns:
        omega: Influence function values [batch, n_bonds]
    """
    # Normalized distance
    r = xi_norm / (delta + 1e-12)
    
    # Compact support mask
    mask = (r < 1.0).float()
    
    if kernel_type == 'constant':
        omega = mask
        
    elif kernel_type == 'linear':
        omega = mask * (1.0 - r)
        
    elif kernel_type == 'cubic_spline':
        # C² continuous cubic B-spline kernel
        omega = mask * torch.clamp(1.0 - 3*r**2 + 2*r**3, min=0.0)
        
    elif kernel_type == 'gaussian':
        # Truncated Gaussian
        sigma = delta / 3.0
        omega = mask * torch.exp(-0.5 * (xi_norm / (sigma + 1e-12))**2)
        
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    return omega


def micromodulus(xi_norm: torch.Tensor,
                 delta: torch.Tensor,
                 E: float = 135e9,
                 dim: int = 3) -> torch.Tensor:
    """
    Compute the bond micromodulus c(ξ) for the bond-based model.
    
    For 3D: c = 18K / (π δ⁴)
    For 2D plane stress: c = 12E / (π δ³ h) 
    For 1D: c = 2E / (A δ²)
    
    With conical correction: c(ξ) = c₀ * ω(|ξ|)
    
    Args:
        xi_norm: Bond lengths [batch, n_bonds]
        delta: Horizon [batch, 1]
        E: Young's modulus
        dim: Spatial dimension (2 or 3)
        
    Returns:
        c: Micromodulus values [batch, n_bonds]
    """
    if dim == 3:
        nu = 0.25  # Bond-based PD constrains Poisson's ratio to 1/4
        K = E / (3 * (1 - 2 * nu))
        c0 = 18 * K / (np.pi * delta**4 + 1e-12)
    elif dim == 2:
        c0 = 12 * E / (np.pi * delta**3 + 1e-12)
    else:
        c0 = 2 * E / (delta**2 + 1e-12)
    
    omega = influence_function(xi_norm, delta, kernel_type='cubic_spline')
    return c0 * omega


def nonlocal_force_density(x_diff: torch.Tensor,
                            u_diff: torch.Tensor,
                            delta: torch.Tensor,
                            bond_damage: Optional[torch.Tensor] = None,
                            E: float = 135e9,
                            dim: int = 3,
                            volumes: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the nonlocal peridynamic force density at a material point.
    
    L(u)(x) = ∫_H f(x', x) dV_x'
    
    For bond-based:
        f(x', x) = c(ξ) * s * (y' - y)/|y' - y| * μ(x', x)
    
    where μ is the damage function (1 = intact, 0 = broken).
    
    Args:
        x_diff: Reference bond vectors (x' - x) [batch, n_bonds, dim]
        u_diff: Displacement differences (u' - u) [batch, n_bonds, dim]
        delta: Horizon values [batch, 1]
        bond_damage: Damage indicators μ [batch, n_bonds], 1=intact, 0=broken
        E: Young's modulus
        dim: Spatial dimension
        volumes: Neighbor volumes for integration [batch, n_bonds]
        
    Returns:
        force_density: Net force density vector [batch, dim]
    """
    xi_norm = torch.norm(x_diff, dim=-1)  # [batch, n_bonds]
    
    # Deformed bond vectors
    y_diff = x_diff + u_diff
    y_norm = torch.norm(y_diff, dim=-1)  # [batch, n_bonds]
    
    # Bond stretch
    s = (y_norm - xi_norm) / (xi_norm + 1e-12)
    
    # Micromodulus
    c = micromodulus(xi_norm, delta, E=E, dim=dim)
    
    # Unit vector in deformed configuration
    e = y_diff / (y_norm.unsqueeze(-1) + 1e-12)
    
    # Bond force magnitude
    f_mag = c * s  # [batch, n_bonds]
    
    # Apply damage
    if bond_damage is not None:
        f_mag = f_mag * bond_damage
    
    # Force density vector
    f_vec = f_mag.unsqueeze(-1) * e  # [batch, n_bonds, dim]
    
    # Integrate (sum with volume weighting)
    if volumes is not None:
        f_vec = f_vec * volumes.unsqueeze(-1)
    
    force_density = f_vec.sum(dim=1)  # [batch, dim]
    
    return force_density


def damage_index(bond_damage: torch.Tensor) -> torch.Tensor:
    """
    Compute the local damage index φ(x) from bond damage states.
    
    φ(x) = 1 - ∫_H μ(x,x') dV / ∫_H dV
    
    φ = 0: no damage, φ = 1: fully damaged
    
    Args:
        bond_damage: Bond integrity μ [batch, n_bonds], 1=intact, 0=broken
        
    Returns:
        phi: Local damage index [batch, 1]
    """
    n_bonds = bond_damage.shape[-1]
    phi = 1.0 - bond_damage.sum(dim=-1, keepdim=True) / (n_bonds + 1e-12)
    return phi


def damage_evolution(stretch: torch.Tensor,
                      stretch_history_max: torch.Tensor,
                      s_c: torch.Tensor,
                      softening: bool = True,
                      s_f: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update bond damage based on stretch history (irreversible).
    
    Standard PMB model:
        μ = 0 if max(s(t')) ≥ s_c for any t' ≤ t
        μ = 1 otherwise
        
    With softening:
        μ = (s_f - s_max) / (s_f - s_c) for s_c ≤ s_max < s_f
        μ = 0 for s_max ≥ s_f
    
    Args:
        stretch: Current bond stretch [batch, n_bonds]
        stretch_history_max: Maximum stretch experienced [batch, n_bonds]
        s_c: Critical stretch for initiation [batch, 1] or scalar
        softening: Whether to use gradual softening
        s_f: Final failure stretch (required if softening=True)
        
    Returns:
        bond_damage: Updated bond integrity μ [batch, n_bonds]
        stretch_max_updated: Updated max stretch history [batch, n_bonds]
    """
    # Update max stretch
    stretch_max = torch.max(stretch.abs(), stretch_history_max)
    
    if softening and s_f is not None:
        # Gradual damage with linear softening
        mu = torch.clamp((s_f - stretch_max) / (s_f - s_c + 1e-12), 0.0, 1.0)
        mu = torch.where(stretch_max < s_c, torch.ones_like(mu), mu)
    else:
        # Brittle: step function (smooth approximation for differentiability)
        mu = 1.0 - torch.sigmoid(50.0 * (stretch_max - s_c))
    
    return mu, stretch_max


def volume_correction(xi_norm: torch.Tensor, 
                       delta: torch.Tensor,
                       dx: float = 0.001) -> torch.Tensor:
    """
    Partial volume correction for bonds near the horizon boundary.
    
    Neighbors at the boundary contribute partial volume, preventing
    artificial stiffness variations near the horizon edge.
    
    Args:
        xi_norm: Bond lengths [batch, n_bonds]
        delta: Horizon [batch, 1]
        dx: Grid spacing [m]
        
    Returns:
        correction: Volume correction factors [batch, n_bonds]
    """
    # Linearly reduce volume for bonds near the boundary
    ratio = (delta - xi_norm) / (dx + 1e-12)
    correction = torch.clamp(ratio, 0.0, 1.0)
    
    # Interior bonds get full volume
    correction = torch.where(xi_norm < delta - dx, torch.ones_like(correction), correction)
    
    return correction


def surface_correction(x_positions: torch.Tensor,
                        domain_bounds: Tuple[torch.Tensor, torch.Tensor],
                        delta: float) -> torch.Tensor:
    """
    Compute surface correction factors for points near domain boundaries.
    
    Points near the surface have incomplete neighborhoods, requiring
    correction to maintain consistent material behavior.
    
    Args:
        x_positions: Material point positions [n_points, dim]
        domain_bounds: Tuple of (min_bounds, max_bounds) each [dim]
        delta: Horizon value
        
    Returns:
        correction: Surface correction factor [n_points]
    """
    min_b, max_b = domain_bounds
    
    # Distance to nearest boundary
    dist_to_min = x_positions - min_b.unsqueeze(0)
    dist_to_max = max_b.unsqueeze(0) - x_positions
    min_dist = torch.min(torch.min(dist_to_min, dim=-1).values,
                          torch.min(dist_to_max, dim=-1).values)
    
    # Correction: 1.0 for interior, < 1.0 near boundaries
    correction = torch.clamp(min_dist / delta, 0.3, 1.0)
    
    return correction
