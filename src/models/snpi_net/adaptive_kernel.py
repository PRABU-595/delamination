import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptiveNonlocalKernel(nn.Module):
    """
    Learns spatially-varying horizon (delta) for peridynamic interactions.
    Implements physics constraints from Section 7.1.2:
    - Monotonically decreasing ω with distance
    - Compact support: ω = 0 for ||x'-x|| > δ
    - Normalization: ∫ω dV' = 1
    """
    def __init__(self, input_dim=6, hidden_dims=[64, 128, 64], delta_min=0.5, delta_max=5.0):
        super().__init__()
        self.delta_min = delta_min
        self.delta_max = delta_max
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim))
            prev_dim = dim
            
        self.encoder = nn.Sequential(*layers)
        self.horizon_predictor = nn.Linear(hidden_dims[-1], 1)
        
        # Weight function shape parameters (learnable)
        self.weight_shape = nn.Parameter(torch.tensor(2.0))  # Controls decay rate
        
    def forward(self, local_state, damage_gradient=None, fiber_orient=None):
        """
        Predict adaptive horizon value δ(x) based on local features.
        
        Args:
            local_state: Basic local features [batch, dim1]
            damage_gradient: Gradient of damage field [batch, dim2]
            fiber_orient: Fiber orientation tensor [batch, dim3]
        
        Returns:
            delta: Adaptive horizon values [batch, 1]
        """
        features_list = [local_state]
        if damage_gradient is not None:
            features_list.append(damage_gradient)
        if fiber_orient is not None:
            features_list.append(fiber_orient)
             
        features = torch.cat(features_list, dim=-1)
        
        # Handle single sample (no batch dimension for BatchNorm)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        h = self.encoder(features)
        
        # Predict positive horizon value with upper bound
        raw_delta = self.horizon_predictor(h)
        delta = self.delta_min + (self.delta_max - self.delta_min) * torch.sigmoid(raw_delta)
        
        return delta
    
    def compute_nonlocal_weight(self, distance, delta):
        """
        Compute nonlocal influence weight ω(||x'-x||, δ).
        
        Implements physics constraints:
        1. Monotonically decreasing with distance
        2. Compact support: ω = 0 for distance > δ
        3. Smooth decay (differentiable)
        
        Args:
            distance: ||x' - x|| distances [batch, n_neighbors]
            delta: Horizon values [batch, 1]
        
        Returns:
            weights: Normalized influence weights [batch, n_neighbors]
        """
        # Normalized distance (0 at center, 1 at horizon boundary)
        normalized_dist = distance / (delta + 1e-8)
        
        # Compact support mask
        mask = (normalized_dist < 1.0).float()
        
        # Smooth monotonically decreasing weight (polynomial kernel)
        # ω(r) = (1 - r^n)^m for r < 1, else 0
        n = F.softplus(self.weight_shape) + 1.0  # Ensure n > 1
        m = 2.0  # Fixed for smoothness
        
        weights = mask * torch.pow(torch.clamp(1.0 - torch.pow(normalized_dist, n), min=0.0), m)
        
        # Normalize weights (∫ω dV' = 1 approximated as sum = 1)
        weight_sum = weights.sum(dim=-1, keepdim=True) + 1e-8
        normalized_weights = weights / weight_sum
        
        return normalized_weights
    
    def compute_nonlocal_interaction(self, x_center, x_neighbors, features_neighbors, delta):
        """
        Compute nonlocal peridynamic-style interaction at a point.
        
        f(x) = ∫_H ω(||x'-x||) * g(x') dV'
        
        Args:
            x_center: Center point coordinates [batch, spatial_dim]
            x_neighbors: Neighbor coordinates [batch, n_neighbors, spatial_dim]
            features_neighbors: Features at neighbors [batch, n_neighbors, feat_dim]
            delta: Horizon values [batch, 1]
        
        Returns:
            interaction: Weighted nonlocal features [batch, feat_dim]
        """
        # Compute distances
        diff = x_neighbors - x_center.unsqueeze(1)  # [batch, n_neighbors, spatial_dim]
        distance = torch.norm(diff, dim=-1)  # [batch, n_neighbors]
        
        # Compute weights
        weights = self.compute_nonlocal_weight(distance, delta)  # [batch, n_neighbors]
        
        # Weighted sum of neighbor features
        interaction = torch.einsum('bn,bnf->bf', weights, features_neighbors)
        
        return interaction


class PeridynamicDamageModel(nn.Module):
    """
    Complete peridynamic damage model using adaptive nonlocal kernel.
    Models bond damage and damage accumulation per Appendix A.1.
    """
    def __init__(self, input_dim=6, hidden_dim=128, critical_stretch=0.01):
        super().__init__()
        self.adaptive_kernel = AdaptiveNonlocalKernel(input_dim=input_dim)
        self.critical_stretch = critical_stretch
        
        # Bond force network
        self.bond_force_net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for stretch
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Scalar force magnitude
        )
        
        # Damage accumulation network
        self.damage_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Damage in [0, 1]
        )
        
    def compute_bond_damage(self, stretch_history):
        """
        Compute bond damage based on stretch history.
        d(x,x',t) = 1 if s(x,x',t') >= s_c for any t' <= t, else 0
        
        Args:
            stretch_history: Max stretch experienced [batch, n_bonds]
        
        Returns:
            damage: Binary damage indicators [batch, n_bonds]
        """
        # Smooth approximation of step function for differentiability
        damage = torch.sigmoid(10.0 * (stretch_history - self.critical_stretch))
        return damage
    
    def forward(self, local_state, neighbor_states, bond_stretches):
        """
        Compute damage and nonlocal forces.
        
        Args:
            local_state: Local damage state [batch, input_dim]
            neighbor_states: Neighbor states [batch, n_neighbors, input_dim]
            bond_stretches: Current bond stretches [batch, n_neighbors]
        
        Returns:
            damage: Local damage value [batch, 1]
            delta: Adaptive horizon [batch, 1]
        """
        # Get adaptive horizon
        delta = self.adaptive_kernel(local_state)
        
        # Compute bond damages
        bond_damage = self.compute_bond_damage(bond_stretches)
        
        # Local damage = fraction of broken bonds (simplified)
        local_damage = bond_damage.mean(dim=-1, keepdim=True)
        
        return local_damage, delta
