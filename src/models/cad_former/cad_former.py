import torch
import torch.nn as nn
import torch.nn.functional as F

from .micro_gnn import MicroGNN, SimpleMicroEncoder
from .meso_cnn import MesoCNN, MacroMLP
from .spatial_attention import SpatialAttention
from .temporal_attention import TemporalAttention
from .migration import InterfaceMigrationPredictor
from .positional_encoding import LaminatePositionalEncoding


class CADFormer(nn.Module):
    """
    Cross-Scale Attentive Delamination Transformer (CAD-Former)
    Main model for Feature 2.
    """
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}
            
        d_model = config.get('d_model', 256)
        n_heads = config.get('n_heads', 8)
        n_layers = config.get('n_layers', 6)
        
        # Multi-scale feature extractors
        # Use SimpleMicroEncoder instead of GNN for simpler initialization
        micro_cfg = config.get('micro_gnn', {})
        self.micro_encoder = SimpleMicroEncoder(
            input_dim=micro_cfg.get('input_dim', 80),
            hidden_dim=micro_cfg.get('hidden_dim', 128),
            output_dim=128
        )
        meso_cfg = config.get('meso_cnn', {})
        self.meso_cnn = MesoCNN(
            in_channels=meso_cfg.get('in_channels', 3),
            hidden_channels=meso_cfg.get('hidden_channels', [32, 64])
        )
        macro_cfg = config.get('macro_mlp', {})
        self.macro_mlp = MacroMLP(
            input_dim=macro_cfg.get('input_dim', 64),
            hidden_dims=macro_cfg.get('hidden_dims', [128, 64]),
            output_dim=64
        )
        
        # Calculate dimension after concatenation
        # Micro (128) + Meso (4096) + Macro (64) = 4288
        fusion_input_dim = 128 + 4096 + 64 
        
        # Cross-scale fusion
        self.scale_fusion = nn.Linear(fusion_input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = LaminatePositionalEncoding(d_model, angle_dim=config.get('angle_dim', 1))
        
        # Transformer layers
        self.dropout = nn.Dropout(config.get('dropout', 0.1))
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'spatial': SpatialAttention(d_model, n_heads, dropout=config.get('dropout', 0.1)),
                'temporal': TemporalAttention(d_model, n_heads, dropout=config.get('dropout', 0.1)),
                'ff': nn.Sequential(
                    nn.Linear(d_model, 4*d_model),
                    nn.GELU(),
                    nn.Linear(4*d_model, d_model),
                    nn.Dropout(config.get('dropout', 0.1))
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model)
            }))
        
        # Migration Predictor (Head)
        self.migration_head = InterfaceMigrationPredictor(d_model)
        
        # History Projection (Raw 1D signal -> d_model)
        self.history_projection = nn.Linear(1, d_model)
        
        # Standard Output heads
        self.area_head = nn.Linear(d_model, 1)
        self.growth_head = nn.Linear(d_model, 1)
        # For overall interface migration prob distribution
        self.migration_dist_head = nn.Linear(d_model, 1) 
        
    def forward(self, micro_data, meso_data, macro_data, laminate_config, loading_history=None, physics_params=None, horizon=None):
        """
        Args:
        - micro_data, meso_data, macro_data: Multi-scale inputs
        - laminate_config: {ply_angles, depths, abd_matrix, is_symmetric}
        - loading_history: [batch, seq_len, ... ] features for temporal attn
        - physics_params: {shear, mode_mixity, ...} for migration head
        - horizon: [batch, 1] Peridynamic horizon for attention masking
        """
        if physics_params is None:
            physics_params = {}
        
        # 1. Feature Extraction
        micro_feat = self.micro_encoder(micro_data) # [batch, 128]
        meso_feat = self.meso_cnn(meso_data)   # [batch, 4096]
        macro_feat = self.macro_mlp(macro_data) # [batch, 64]

        
        # 2. Fusion
        fused = torch.cat([micro_feat, meso_feat, macro_feat], dim=-1)
        features = self.scale_fusion(fused) # [batch, d_model]
        
        # Reshape for sequence processing: [batch, n_interfaces, d_model]
        # For simplicity, if inputs are single-interface, we unsqueeze. 
        # But CAD-Former is designed for multiple interfaces.
        # Assuming current 'features' represents the state of ALL interfaces (concatenated)
        # OR we process each interface separately before fusion.
        # Let's assume 'features' is per-laminate, and we expand it to per-interface
        n_interfaces = laminate_config['ply_angles'].shape[1]
        features = features.unsqueeze(1).expand(-1, n_interfaces, -1) # Broadcast global features
        
        # 3. Add Positional Encoding (Physics-based)
        pos_enc = self.pos_encoder(
            laminate_config['ply_angles'],
            laminate_config['depths'],
            laminate_config['abd_matrix'],
            laminate_config['is_symmetric']
        )
        features = features + pos_enc
        features = self.dropout(features)
        
        # Compute Horizon Mask
        txn_mask = None
        if horizon is not None:
            # horizon: [batch, 1]
            # depths: [batch, n_interfaces, 1]
            z = laminate_config['depths']
            if z.dim() == 2: z = z.unsqueeze(-1) # Handle missing dim
            
            # [batch, n, n] distance matrix
            dist = torch.abs(z - z.transpose(1, 2))
            
            # Mask: 0 if dist <= horizon, -inf else
            h = horizon.view(-1, 1, 1)
            txn_mask = torch.where(dist <= h, torch.zeros_like(dist), torch.tensor(float('-inf')).to(dist.device))
            
            # MultiHeadAttention mask shape requirements might vary by version
            # Usually [batch*num_heads, L, S] or [batch, L, S] is fine
            # We assume [batch, L, S] works
        
        # 4. Transformer Blocks
        spatial_weights_list = []
        temporal_weights_list = []
        
        # Project history once if present
        history_emb = None
        if loading_history is not None:
             if loading_history.dim() == 2:
                 # [batch, seq_len] -> [batch, seq_len, 1]
                 loading_history = loading_history.unsqueeze(-1)
             
             # Project to d_model: [batch, seq_len, d_model]
             history_emb = self.history_projection(loading_history)
        
        for layer in self.layers:
            # Spatial Attention (Interface-Interface)
            # Pass mask here
            sp_out, sp_weights = layer['spatial'](features, mask=txn_mask)
            spatial_weights_list.append(sp_weights)
            sp_out = self.dropout(sp_out)
            
            # Temporal Attention (if history exists)
            temporal_out = 0
            if history_emb is not None:
                tp_out, tp_weights = layer['temporal'](features, history_emb)
                temporal_out = self.dropout(tp_out)
                temporal_weights_list.append(tp_weights)
                
            # Residual + Norm
            features = layer['norm1'](features + sp_out + temporal_out)
            
            # Feed Forward
            ff_out = layer['ff'](features)
            features = layer['norm2'](features + ff_out)
            
        # 5. Predictions
        # Global pooling (e.g., max damage across interfaces) or specific interface query
        # Here we take mean for global scalar predictions
        global_feat = features.mean(dim=1) 
        
        delamination_area = self.area_head(global_feat)
        growth_rate = self.growth_head(global_feat)
        
        # Migration Probabilities (End-to-end for all pairs)
        # This requires constructing pairs. For simplicity, we output marginal prob per interface
        migration_logits = self.migration_dist_head(features)  # [batch, n_interfaces, 1]
        migration_probs = torch.softmax(migration_logits.squeeze(-1), dim=1).unsqueeze(-1)
        
        return {
            'delamination_area': delamination_area,
            'growth_rate': growth_rate,
            'migration_probs': migration_probs,
            'spatial_attention': spatial_weights_list,
            'temporal_attention': temporal_weights_list
        }
