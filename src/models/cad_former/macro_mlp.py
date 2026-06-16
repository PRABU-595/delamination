import torch
import torch.nn as nn

class MacroMLP(nn.Module):
    """
    MLP for processing macroscopic structural features (e.g., global delamination front geometry, load).
    """
    def __init__(self, input_dim=64, output_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.model(x)
