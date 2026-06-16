import torch
import torch.nn as nn
import torch.nn.functional as F

class InterfaceMigrationPredictor(nn.Module):
    """
    Predicts probability of delamination migrating from interface i to j.
    From Section 7.2.3:
    
    Physics-based features:
    - Shear stress (sign change indicates migration)
    - Mode mixity (GII/GI ~ 0.3-0.7 favors migration)
    - Crack density (matrix cracking enables pathways)
    - Ply angle mismatch (larger Δθ increases migration)
    """
    def __init__(self, feature_dim=256, physics_dim=4, hidden_dim=128):
        super().__init__()
        
        self.interface_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Physics feature processor
        self.physics_encoder = nn.Sequential(
            nn.Linear(physics_dim, 32),
            nn.ReLU()
        )
        
        # Transition predictor: takes encoded features from both interfaces + physics
        self.transition_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Probability in [0, 1]
        )
        
    def forward(self, interface_i_features, interface_j_features,
                shear_stress, mode_mixity, crack_density, ply_angle_diff):
        """
        Args:
            interface_i_features: Source interface [batch, feature_dim]
            interface_j_features: Target interface [batch, feature_dim]
            shear_stress: Interlaminar shear stress [batch, 1]
            mode_mixity: GII/(GI+GII) ratio [batch, 1]
            crack_density: Matrix crack density between interfaces [batch, 1]
            ply_angle_diff: |θ_i - θ_j| orientation mismatch [batch, 1]
        
        Returns:
            p_migration: Migration probability [batch, 1]
        """
        # Encode interface states
        h_i = self.interface_encoder(interface_i_features)
        h_j = self.interface_encoder(interface_j_features)
        
        # Stack physics features
        physics_feats = torch.cat([
            shear_stress,
            mode_mixity,
            crack_density,
            ply_angle_diff
        ], dim=-1)
        
        physics_enc = self.physics_encoder(physics_feats)
        
        # Concatenate and predict
        combined = torch.cat([h_i, h_j, physics_enc], dim=-1)
        p_migration = self.transition_predictor(combined)
        
        return p_migration
    
    def compute_migration_matrix(self, interface_features, physics_params):
        """
        Compute full migration probability matrix for all interface pairs.
        
        Args:
            interface_features: [n_interfaces, feature_dim]
            physics_params: Dict with keys 'shear_stress', 'mode_mixity', 
                           'crack_density', 'ply_angles' each [n_interfaces]
        
        Returns:
            migration_matrix: [n_interfaces, n_interfaces] probability matrix
        """
        n = interface_features.shape[0]
        device = interface_features.device
        
        migration_matrix = torch.zeros(n, n, device=device)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Compute physics features for this pair
                    ply_angle_diff = torch.abs(
                        physics_params['ply_angles'][i] - physics_params['ply_angles'][j]
                    ).unsqueeze(0).unsqueeze(0)
                    
                    p_ij = self.forward(
                        interface_features[i:i+1],
                        interface_features[j:j+1],
                        physics_params['shear_stress'][i:i+1].unsqueeze(-1),
                        physics_params['mode_mixity'][i:i+1].unsqueeze(-1),
                        physics_params['crack_density'][i:i+1].unsqueeze(-1),
                        ply_angle_diff
                    )
                    migration_matrix[i, j] = p_ij.squeeze()
        
        return migration_matrix


class MigrationPathwayGenerator:
    """
    Generates probabilistic graph of migration routes.
    Uses migration probabilities to construct a directed graph.
    """
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        
    def generate_pathways(self, migration_matrix, interface_names=None):
        """
        Generate migration pathway graph from probability matrix.
        
        Args:
            migration_matrix: [n, n] migration probabilities
            interface_names: Optional list of interface names
        
        Returns:
            edges: List of (source, target, probability) tuples
        """
        n = migration_matrix.shape[0]
        if interface_names is None:
            interface_names = [f"Interface_{i}" for i in range(n)]
        
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and migration_matrix[i, j].item() > self.threshold:
                    edges.append((
                        interface_names[i],
                        interface_names[j],
                        migration_matrix[i, j].item()
                    ))
        
        return edges
