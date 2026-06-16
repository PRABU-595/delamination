"""
Classical Lamination Theory (CLT) Implementation.

Provides ABD matrix computation, ply stiffness transformations,
and failure criteria for composite laminate analysis.

References:
    - Jones, R.M., "Mechanics of Composite Materials", 2nd Ed.
    - Reddy, J.N., "Mechanics of Laminated Composite Plates and Shells"
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict


def ply_stiffness_matrix(E1: float, E2: float, G12: float, nu12: float) -> torch.Tensor:
    """
    Compute the reduced stiffness matrix Q for a single unidirectional ply.
    
    Q = [[Q11, Q12, 0  ],
         [Q12, Q22, 0  ],
         [0,   0,   Q66]]
    
    Args:
        E1: Longitudinal modulus (fiber direction) [Pa]
        E2: Transverse modulus [Pa]
        G12: In-plane shear modulus [Pa]
        nu12: Major Poisson's ratio
        
    Returns:
        Q: [3, 3] reduced stiffness matrix
    """
    nu21 = nu12 * E2 / E1
    denom = 1.0 - nu12 * nu21
    
    Q11 = E1 / denom
    Q22 = E2 / denom
    Q12 = nu12 * E2 / denom
    Q66 = G12
    
    Q = torch.zeros(3, 3)
    Q[0, 0] = Q11
    Q[1, 1] = Q22
    Q[0, 1] = Q12
    Q[1, 0] = Q12
    Q[2, 2] = Q66
    
    return Q


def transformation_matrix(theta_deg: float) -> torch.Tensor:
    """
    Compute the stress transformation matrix T for rotation by angle theta.
    
    Args:
        theta_deg: Ply orientation angle in degrees
        
    Returns:
        T: [3, 3] transformation matrix
    """
    theta = torch.tensor(theta_deg * np.pi / 180.0)
    c = torch.cos(theta)
    s = torch.sin(theta)
    
    T = torch.tensor([
        [c**2,     s**2,      2*c*s    ],
        [s**2,     c**2,     -2*c*s    ],
        [-c*s,     c*s,       c**2 - s**2]
    ])
    
    return T


def rotated_stiffness(Q: torch.Tensor, theta_deg: float) -> torch.Tensor:
    """
    Compute the transformed (rotated) stiffness matrix Q̄ = T^(-1) Q T^(-T).
    
    Uses the Reuter matrix approach for proper engineering strain transformation.
    
    Args:
        Q: [3, 3] reduced stiffness matrix (material axes)
        theta_deg: Ply angle in degrees
        
    Returns:
        Q_bar: [3, 3] transformed stiffness matrix (laminate axes)
    """
    theta = torch.tensor(theta_deg * np.pi / 180.0, dtype=Q.dtype)
    c = torch.cos(theta)
    s = torch.sin(theta)
    
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    
    Q_bar = torch.zeros(3, 3, dtype=Q.dtype)
    
    Q_bar[0, 0] = Q11*c**4 + 2*(Q12 + 2*Q66)*s**2*c**2 + Q22*s**4
    Q_bar[1, 1] = Q11*s**4 + 2*(Q12 + 2*Q66)*s**2*c**2 + Q22*c**4
    Q_bar[0, 1] = (Q11 + Q22 - 4*Q66)*s**2*c**2 + Q12*(c**4 + s**4)
    Q_bar[1, 0] = Q_bar[0, 1]
    Q_bar[0, 2] = (Q11 - Q12 - 2*Q66)*s*c**3 + (Q12 - Q22 + 2*Q66)*s**3*c
    Q_bar[2, 0] = Q_bar[0, 2]
    Q_bar[1, 2] = (Q11 - Q12 - 2*Q66)*s**3*c + (Q12 - Q22 + 2*Q66)*s*c**3
    Q_bar[2, 1] = Q_bar[1, 2]
    Q_bar[2, 2] = (Q11 + Q22 - 2*Q12 - 2*Q66)*s**2*c**2 + Q66*(c**4 + s**4)
    
    return Q_bar


def ABD_matrix(stacking_sequence: List[float],
               ply_thickness: float = 0.125e-3,
               E1: float = 135e9,
               E2: float = 9.5e9,
               G12: float = 5.0e9,
               nu12: float = 0.3) -> Dict[str, torch.Tensor]:
    """
    Compute the ABD stiffness matrix for a composite laminate.
    
    [N]   [A  B] [ε°]
    [M] = [B  D] [κ ]
    
    A = Extensional stiffness (membrane)
    B = Coupling stiffness (bending-extension)  
    D = Bending stiffness
    
    Args:
        stacking_sequence: List of ply angles in degrees, e.g. [0, 45, -45, 90]_s
        ply_thickness: Individual ply thickness [m]
        E1: Longitudinal modulus [Pa]
        E2: Transverse modulus [Pa]
        G12: Shear modulus [Pa]
        nu12: Poisson's ratio
        
    Returns:
        Dict with keys:
            'A': [3,3] extensional stiffness
            'B': [3,3] coupling stiffness
            'D': [3,3] bending stiffness
            'ABD': [6,6] full ABD matrix
            'abd': [6,6] compliance matrix (inverse of ABD)
            'z_coords': [n+1] interface z-coordinates
            'is_symmetric': bool
    """
    n_plies = len(stacking_sequence)
    h_total = n_plies * ply_thickness
    
    # Z-coordinates of ply interfaces (measured from midplane)
    z = torch.linspace(-h_total / 2, h_total / 2, n_plies + 1)
    
    # Base ply stiffness
    Q = ply_stiffness_matrix(E1, E2, G12, nu12)
    
    # Initialize ABD
    A = torch.zeros(3, 3)
    B = torch.zeros(3, 3)
    D = torch.zeros(3, 3)
    
    for k, theta in enumerate(stacking_sequence):
        Q_bar = rotated_stiffness(Q, theta)
        
        z_bot = z[k]
        z_top = z[k + 1]
        
        # A_ij = Σ Q̄_ij (z_top - z_bot)
        A += Q_bar * (z_top - z_bot)
        
        # B_ij = (1/2) Σ Q̄_ij (z_top² - z_bot²)
        B += 0.5 * Q_bar * (z_top**2 - z_bot**2)
        
        # D_ij = (1/3) Σ Q̄_ij (z_top³ - z_bot³)
        D += (1.0 / 3.0) * Q_bar * (z_top**3 - z_bot**3)
    
    # Assemble full ABD
    ABD = torch.zeros(6, 6)
    ABD[:3, :3] = A
    ABD[:3, 3:] = B
    ABD[3:, :3] = B
    ABD[3:, 3:] = D
    
    # Compliance matrix
    try:
        abd = torch.inverse(ABD)
    except RuntimeError:
        abd = torch.zeros(6, 6)  # Singular — shouldn't happen for valid laminates
    
    # Check symmetry (B ≈ 0 for symmetric laminates)
    is_symmetric = bool(torch.max(torch.abs(B)) < 1e-6 * torch.max(torch.abs(A)))
    
    return {
        'A': A,
        'B': B,
        'D': D,
        'ABD': ABD,
        'abd': abd,
        'z_coords': z,
        'is_symmetric': is_symmetric
    }


def interface_depths(stacking_sequence: List[float],
                     ply_thickness: float = 0.125e-3) -> torch.Tensor:
    """
    Return the z-coordinates of all ply interfaces.
    
    Args:
        stacking_sequence: List of ply angles
        ply_thickness: Individual ply thickness [m]
        
    Returns:
        depths: [n_plies - 1] tensor of interface z-coordinates
    """
    n_plies = len(stacking_sequence)
    h_total = n_plies * ply_thickness
    z = torch.linspace(-h_total / 2, h_total / 2, n_plies + 1)
    return z[1:-1]  # Interior interfaces only


def ply_angle_mismatch(stacking_sequence: List[float]) -> torch.Tensor:
    """
    Compute the angle mismatch Δθ between adjacent plies at each interface.
    
    Higher mismatch → higher migration probability (Section 7.2.3).
    
    Args:
        stacking_sequence: List of ply angles
        
    Returns:
        mismatch: [n_plies - 1] tensor of |θ_k+1 - θ_k|
    """
    angles = torch.tensor(stacking_sequence, dtype=torch.float32)
    return torch.abs(angles[1:] - angles[:-1])


def tsai_wu_criterion(sigma: torch.Tensor,
                      strengths: Dict[str, float]) -> torch.Tensor:
    """
    Tsai-Wu failure criterion for a single ply.
    
    f = F1*σ1 + F2*σ2 + F11*σ1² + F22*σ2² + F66*τ12² + 2*F12*σ1*σ2
    
    Failure when f ≥ 1.
    
    Args:
        sigma: [batch, 3] stress tensor [σ1, σ2, τ12] in material coordinates
        strengths: Dict with keys 'Xt', 'Xc', 'Yt', 'Yc', 'S12'
            Xt: Longitudinal tensile strength
            Xc: Longitudinal compressive strength
            Yt: Transverse tensile strength
            Yc: Transverse compressive strength
            S12: In-plane shear strength
            
    Returns:
        f: [batch] failure index (≥ 1 means failure)
    """
    Xt = strengths['Xt']
    Xc = strengths['Xc']
    Yt = strengths['Yt']
    Yc = strengths['Yc']
    S12 = strengths['S12']
    
    F1 = 1.0/Xt - 1.0/Xc
    F2 = 1.0/Yt - 1.0/Yc
    F11 = 1.0 / (Xt * Xc)
    F22 = 1.0 / (Yt * Yc)
    F66 = 1.0 / (S12**2)
    F12 = -0.5 * torch.sqrt(torch.tensor(F11 * F22))
    
    s1 = sigma[:, 0]
    s2 = sigma[:, 1]
    t12 = sigma[:, 2]
    
    f = (F1*s1 + F2*s2 + F11*s1**2 + F22*s2**2 + F66*t12**2 + 2*F12*s1*s2)
    
    return f


def hashin_criterion(sigma: torch.Tensor,
                     strengths: Dict[str, float]) -> Dict[str, torch.Tensor]:
    """
    Hashin failure criteria distinguishing fiber and matrix failure modes.
    
    Args:
        sigma: [batch, 3] stress [σ1, σ2, τ12] in material coordinates
        strengths: Dict with 'Xt', 'Xc', 'Yt', 'Yc', 'S12', 'S23'
        
    Returns:
        Dict with failure indices for each mode:
            'fiber_tension': Tensile fiber failure
            'fiber_compression': Compressive fiber failure
            'matrix_tension': Tensile matrix failure
            'matrix_compression': Compressive matrix failure
    """
    Xt = strengths['Xt']
    Xc = strengths['Xc']
    Yt = strengths['Yt']
    Yc = strengths['Yc']
    S12 = strengths['S12']
    S23 = strengths.get('S23', Yt / 2.0)
    
    s1 = sigma[:, 0]
    s2 = sigma[:, 1]
    t12 = sigma[:, 2]
    
    # Fiber tension (σ1 ≥ 0)
    ft = (s1 / Xt)**2 + (t12 / S12)**2
    
    # Fiber compression (σ1 < 0)
    fc = (-s1 / Xc)**2
    
    # Matrix tension (σ2 ≥ 0)
    mt = (s2 / Yt)**2 + (t12 / S12)**2
    
    # Matrix compression (σ2 < 0)
    mc = (s2 / (2*S23))**2 + ((Yc / (2*S23))**2 - 1) * (s2 / Yc) + (t12 / S12)**2
    
    # Apply correct mode based on stress sign
    fiber = torch.where(s1 >= 0, ft, fc)
    matrix = torch.where(s2 >= 0, mt, mc)
    
    return {
        'fiber_tension': ft,
        'fiber_compression': fc,
        'matrix_tension': mt,
        'matrix_compression': mc,
        'fiber': fiber,
        'matrix': matrix
    }


def laminate_stiffness_tensor(stacking_sequence: List[float],
                              ply_thickness: float = 0.125e-3,
                              E1: float = 135e9,
                              E2: float = 9.5e9,
                              G12: float = 5.0e9,
                              nu12: float = 0.3) -> torch.Tensor:
    """
    Compute the effective laminate engineering constants from ABD matrix.
    
    Returns:
        props: [6] tensor [Ex, Ey, Gxy, nu_xy, nu_yx, h_total]
    """
    result = ABD_matrix(stacking_sequence, ply_thickness, E1, E2, G12, nu12)
    abd = result['abd']
    h = len(stacking_sequence) * ply_thickness
    
    # Effective properties from compliance
    a = abd[:3, :3]  # Extensional compliance
    
    Ex = 1.0 / (h * a[0, 0])  if abs(a[0, 0]) > 1e-20 else 0.0
    Ey = 1.0 / (h * a[1, 1])  if abs(a[1, 1]) > 1e-20 else 0.0
    Gxy = 1.0 / (h * a[2, 2]) if abs(a[2, 2]) > 1e-20 else 0.0
    nu_xy = -a[0, 1] / a[0, 0] if abs(a[0, 0]) > 1e-20 else 0.0
    nu_yx = -a[0, 1] / a[1, 1] if abs(a[1, 1]) > 1e-20 else 0.0
    
    return torch.tensor([Ex, Ey, Gxy, nu_xy, nu_yx, h], dtype=torch.float32)
