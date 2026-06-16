import torch
import torch.nn as nn

class MesoCNN(nn.Module):
    """
    CNN for ply-level mesoscale features (Section 7.2.3).
    Processes matrix crack patterns and local delamination zones.
    
    Architecture from Appendix B.2:
    - Conv2d(3, 32, k=3) -> ReLU -> MaxPool(2)
    - Conv2d(32, 64, k=3) -> ReLU -> AdaptiveAvgPool(8,8)
    
    Output: [batch, 4096] meso descriptor
    """
    def __init__(self, in_channels=3, hidden_channels=[32, 64]):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels[1]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        self.flatten = nn.Flatten()
        
        # Output dimension: 64 * 8 * 8 = 4096
        self.output_dim = hidden_channels[1] * 8 * 8
        
    def forward(self, crack_density_map):
        """
        Args:
            crack_density_map: [batch, C, H, W] image of damage patterns
                C can be: 1 (grayscale), 3 (crack density tensor components)
        
        Returns:
            meso_descriptor: [batch, 4096]
        """
        x = self.conv_layers(crack_density_map)
        meso_descriptor = self.flatten(x)
        return meso_descriptor


class MacroMLP(nn.Module):
    """
    MLP for macroscale structural features (Section 7.2.3).
    Processes global delamination front geometry and loading conditions.
    
    Input features:
    - Delamination area
    - Front curvature
    - Mode mixity (GII/GI)
    - Applied loads
    - Global compliance
    
    Output: [batch, 64] macro descriptor
    """
    def __init__(self, input_dim=64, hidden_dims=[128, 64], output_dim=64):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, macro_features):
        """
        Args:
            macro_features: [batch, input_dim] structural-level features
        
        Returns:
            macro_descriptor: [batch, output_dim]
        """
        return self.encoder(macro_features)
