import torch
import torch.nn as nn

class InterfaceMigrationPredictor(nn.Module):
    """
    Predicts probability of delamination migrating from interface i to interface j.
    Incorporates physics-based features: shear stress, mode mixity, etc.
    """
    def __init__(self, feature_dim=256):
        super().__init__()
        self.interface_encoder = nn.Linear(feature_dim, 128)
        
        # Physics inputs dim = 4 (shear + mixity + density + angle_diff)
        self.transition_predictor = nn.Sequential(
            nn.Linear(128*2 + 4, 64), 
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, interface_i_features, interface_j_features, 
                shear_stress, mode_mixity, crack_density, ply_angles_diff):
        """
        Args:
            interface_features: [batch, feature_dim]
            physics_params: [batch, 1 each]
        """
        
        # Encode interface states
        h_i = self.interface_encoder(interface_i_features)
        h_j = self.interface_encoder(interface_j_features)
        
        # Physics-based features
        physics_feats = torch.cat([
            shear_stress,      # Interlaminar shear stress
            mode_mixity,       # GII/GI ratio
            crack_density,     # Matrix cracks bridging interfaces
            ply_angles_diff    # |θ_i - θ_j| orientation mismatch
        ], dim=-1)
        
        # Concatenate and predict
        combined = torch.cat([h_i, h_j, physics_feats], dim=-1)
        p_migration = self.transition_predictor(combined)
        
        return p_migration  # Probability: [0, 1]
