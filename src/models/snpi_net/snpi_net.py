import torch
import torch.nn as nn
from .adaptive_kernel import AdaptiveNonlocalKernel, PeridynamicDamageModel
from .uncertainty import UncertaintyNetwork
from .multi_fidelity import MultiFidelityGP

class SNPINet(nn.Module):
    """
    Stochastic Nonlocal Peridynamic-Informed Neural Network (SNPI-Net)
    
    Integrates:
    1. Adaptive Nonlocal Kernel learning
    2. Uncertainty decomposition (Aleatoric + Epistemic)
    3. Multi-fidelity data fusion
    """
    def __init__(self, config):
        super().__init__()
        
        # 1. Adaptive Nonlocal Kernel (Physics Core)
        # We access the kernel via the full damage model container
        kernel_cfg = config.get('adaptive_kernel', {})
        self.pd_model = PeridynamicDamageModel(
            input_dim=kernel_cfg.get('input_dim', 6),
            hidden_dim=kernel_cfg.get('hidden_dim', 128),
            critical_stretch=kernel_cfg.get('critical_stretch', 0.01)
        )
        
        # 2. Uncertainty Network (Deep Neural Network part)
        unc_cfg = config.get('uncertainty', {})
        # Input dim is feature_dim (256) + horizon (1) + local_damage (1)
        self.uncertainty_net = UncertaintyNetwork(
            input_dim=256 + 2, 
            dropout_rate=unc_cfg.get('dropout_rate', 0.1)
        )
        
        # 3. Multi-Fidelity Component
        self.mf_gp = MultiFidelityGP(input_dim=6)
        
        # Main Feature Extractor (Encoder)
        self.feature_encoder = nn.Sequential(
            nn.Linear(6, 128),
            nn.LeakyReLU(0.2), 
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x, damage_grad=None, fiber_orient=None, return_horizon=False, 
                neighbor_states=None, bond_stretches=None):
        """
        Forward pass for training/inference.
        
        Args:
            x: Local state features [batch, 6]
            neighbor_states: (Optional) [batch, n_neighbors, 6]
            bond_stretches: (Optional) [batch, n_neighbors]
        """
        # 1. Compute Physics State (Peridynamics)
        if neighbor_states is not None and bond_stretches is not None:
            # Full nonlocal simulation step
            local_damage, delta = self.pd_model(x, neighbor_states, bond_stretches)
        else:
            # Approximation: Predict horizon, assume zero damage if no history
            delta = self.pd_model.adaptive_kernel(x, damage_grad, fiber_orient)
            local_damage = torch.zeros_like(delta)
        
        # 2. Extract Data Features
        features = self.feature_encoder(x)
        
        # 3. Physics-Informed Fusion
        # Concatenate learned features with physics state (Horizon + Damage)
        features_augmented = torch.cat([features, delta, local_damage], dim=-1)
        
        # 4. Predict Delamination & Uncertainty
        mean, log_var = self.uncertainty_net(features_augmented)
        
        outputs = {
            'prediction': mean,
            'aleatoric_log_var': log_var,
            'local_damage': local_damage,
            'horizon': delta
        }
        
        if return_horizon:
            outputs['horizon'] = delta
            
        return outputs
    
    def predict_uncertainty(self, x, n_samples=50):
        """
        Full prediction with decomposed uncertainty.
        """
        self.train() 
        means = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x) 
                means.append(out['prediction'])
        
        stacked_means = torch.stack(means)
        mean_pred = torch.mean(stacked_means, dim=0)
        epistemic = torch.var(stacked_means, dim=0)
        
        # Aleatoric (using single pass for efficiency)
        out_final = self.forward(x)
        aleatoric = torch.exp(out_final['aleatoric_log_var'])
        
        return {
            'mean': mean_pred,
            'epistemic': epistemic,
            'aleatoric': aleatoric
        }
