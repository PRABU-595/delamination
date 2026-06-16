import torch
import torch.nn as nn

class LaminatePositionalEncoding(nn.Module):
    """
    Encodes physical laminate information (ply angles, depth, stiffness) instead of just index.
    """
    def __init__(self, d_model=256, angle_dim=1):
        super().__init__()
        self.ply_angle_encoder = nn.Linear(angle_dim, d_model//4)
        self.interface_position_encoder = nn.Linear(1, d_model//4)
        self.stiffness_encoder = nn.Linear(3, d_model//4)  # ABD matrix info (A11, A22, D11)
        self.symmetry_encoder = nn.Linear(1, d_model//4)
        
    def forward(self, ply_angles, depths, abd_matrix, is_symmetric):
        """
        Args:
            ply_angles: [batch, n_interfaces, 1]
            depths: [batch, n_interfaces, 1]
            abd_matrix: [batch, n_interfaces, 3] (or global broadcasted)
            is_symmetric: [batch, n_interfaces, 1]
        """
        angle_enc = self.ply_angle_encoder(ply_angles)
        depth_enc = self.interface_position_encoder(depths)
        stiff_enc = self.stiffness_encoder(abd_matrix)
        symm_enc = self.symmetry_encoder(is_symmetric)
        
        pos_encoding = torch.cat([angle_enc, depth_enc, stiff_enc, symm_enc], dim=-1)
        return pos_encoding  # Shape: [batch, n_interfaces, d_model]
