import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    """
    Models interactions between different ply interfaces to identify migration likelihood.
    """
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, interface_features, mask=None):
        """
        Args:
            interface_features: [batch, n_interfaces, d_model]
            
        Returns:
            output: [batch, n_interfaces, d_model]
            attn_weights: [batch, n_interfaces, n_interfaces]
        """
        # Self-attention over interfaces
        attn_output, attn_weights = self.multihead_attn(
            query=interface_features,
            key=interface_features,
            value=interface_features,
            attn_mask=mask
        )
        
        # Residual connection + Norm (standard Transformer block part)
        # Note: In the paper's arch, this might be just the attention part, 
        # but adding residual/norm is standard practice for stability.
        # We'll just return attention output as per doc description, 
        # integration happens in main model.
        
        return attn_output, attn_weights
