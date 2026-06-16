import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class MicroGNN(nn.Module):
    """
    Graph Neural Network for fiber-matrix microstructure (Section 7.2.3).
    Extracts microscale features from RVE topology.
    
    Input: Node features representing fiber-matrix interface properties
    Output: [batch, 128] micro descriptor
    """
    def __init__(self, node_feat_dim=8, edge_feat_dim=4, hidden_dims=[64, 128]):
        super().__init__()
        
        # GCN layers
        self.conv1 = GCNConv(node_feat_dim, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features [total_nodes, node_feat_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for nodes [total_nodes] (for batched graphs)
        
        Returns:
            micro_descriptor: [batch_size, 128]
        """
        # First GCN layer
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Second GCN layer
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        
        # Global pooling to get graph-level representation
        if batch is not None:
            micro_descriptor = global_mean_pool(h, batch)
        else:
            # Single graph - just mean pool all nodes
            micro_descriptor = h.mean(dim=0, keepdim=True)
        
        return micro_descriptor


class SimpleMicroEncoder(nn.Module):
    """
    Simplified micro-scale encoder when GNN is not needed.
    Uses MLP on flattened/aggregated micro features.
    """
    def __init__(self, input_dim=80, hidden_dim=128, output_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        """
        Args:
            x: Micro features [batch, nodes, feat] or [batch, flat_dim]
        
        Returns:
            micro_descriptor: [batch, output_dim]
        """
        if x.dim() == 3:
            # Flatten node features
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
        
        return self.encoder(x)
