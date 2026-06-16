import torch
import torch.nn as nn
import math

class LaminatePositionalEncoding(nn.Module):
    """
    Physics-embedded positional encoding for transformer (Section 7.2.3).
    
    Encodes laminate structural information:
    - Ply angles (fiber orientation)
    - Interface depths (through-thickness position)
    - ABD matrix (Classical Laminate Theory stiffness)
    - Symmetry (symmetric laminates have mirrored encodings)
    """
    def __init__(self, d_model=256):
        super().__init__()
        
        # Separate encoders for each physical property
        self.ply_angle_encoder = nn.Linear(1, d_model // 4)
        self.interface_depth_encoder = nn.Linear(1, d_model // 4)
        self.stiffness_encoder = nn.Linear(3, d_model // 4)  # A11, A22, D11
        self.symmetry_encoder = nn.Linear(1, d_model // 4)
        
        # Learnable combination weights
        self.combine_proj = nn.Linear(d_model, d_model)
        
    def forward(self, laminate_config):
        """
        Args:
            laminate_config: Dict with keys:
                - 'ply_angles': [batch, n_plies, 1]
                - 'depths': [batch, n_interfaces, 1]
                - 'abd_matrix': [batch, n_interfaces, 3] (A11, A22, D11)
                - 'is_symmetric': [batch, n_interfaces, 1]
        
        Returns:
            pos_encoding: [batch, n_interfaces, d_model]
        """
        # Encode each component
        angle_enc = self.ply_angle_encoder(laminate_config['ply_angles'])
        depth_enc = self.interface_depth_encoder(laminate_config['depths'])
        stiff_enc = self.stiffness_encoder(laminate_config['abd_matrix'])
        symm_enc = self.symmetry_encoder(laminate_config['is_symmetric'])
        
        # Concatenate along feature dimension
        pos_encoding = torch.cat([angle_enc, depth_enc, stiff_enc, symm_enc], dim=-1)
        
        # Project to final dimension
        pos_encoding = self.combine_proj(pos_encoding)
        
        return pos_encoding


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequence position.
    Used for loading history sequences in temporal attention.
    """
    def __init__(self, d_model=256, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class HybridPositionalEncoding(nn.Module):
    """
    Combines laminate physics encoding with temporal sequence encoding.
    For sequences of laminate states over loading history.
    """
    def __init__(self, d_model=256, max_len=1000):
        super().__init__()
        
        self.laminate_pe = LaminatePositionalEncoding(d_model)
        self.temporal_pe = SinusoidalPositionalEncoding(d_model, max_len)
        
        # Gate to balance laminate vs temporal info
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x, laminate_config):
        """
        Args:
            x: Sequence features [batch, seq_len, d_model]
            laminate_config: Laminate configuration dict
        
        Returns:
            Encoded sequence [batch, seq_len, d_model]
        """
        # Get laminate encoding (same for all time steps)
        lam_enc = self.laminate_pe(laminate_config)  # [batch, n_interfaces, d_model]
        
        # Average over interfaces for global laminate encoding
        if lam_enc.dim() == 3:
            lam_enc_global = lam_enc.mean(dim=1, keepdim=True)  # [batch, 1, d_model]
            lam_enc_global = lam_enc_global.expand(-1, x.size(1), -1)
        else:
            lam_enc_global = lam_enc.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # Apply temporal encoding
        x_temporal = self.temporal_pe(x)
        
        # Gate combination
        combined = torch.cat([x_temporal, lam_enc_global], dim=-1)
        gate_weight = self.gate(combined)
        
        output = gate_weight * x_temporal + (1 - gate_weight) * (x + lam_enc_global)
        
        return output
