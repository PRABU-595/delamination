import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    """
    Captures loading history effects critical for fatigue delamination.
    """
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, current_state, loading_history):
        """
        Args:
            current_state: [batch, 1, d_model] or [batch, n_interfaces, d_model]
            loading_history: [batch, seq_len, d_model]
            
        Returns:
            output: Context vector from history
            weights: Attention weights
        """
        # Query: Current state
        # Key/Value: History
        
        attn_output, attn_weights = self.temporal_attn(
            query=current_state,
            key=loading_history,
            value=loading_history
        )
        
        return attn_output, attn_weights
