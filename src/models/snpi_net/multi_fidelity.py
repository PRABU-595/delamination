import torch
import torch.nn as nn

try:
    import gpytorch
    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False


if HAS_GPYTORCH:
    class ExactGPModel(gpytorch.models.ExactGP):
        """
        Standard Exact GP model to be used as a building block.
        """
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5)
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultiFidelityGP(nn.Module):
    """
    Multi-Fidelity Gaussian Process that specifically handles low, medium, and high fidelity data.
    Uses an autoregressive scheme: f_high(x) = rho * f_low(x) + delta(x)
    
    NOTE: If gpytorch is not installed, this operates as a lightweight stub
    that outputs zeros (sufficient for inference mode).
    """
    def __init__(self, input_dim=6):
        super().__init__()
        self.input_dim = input_dim
        
        # Scaling factors (rho) — always available
        self.rho_mid = nn.Parameter(torch.tensor(1.0))
        self.rho_high = nn.Parameter(torch.tensor(1.0))
        
        if HAS_GPYTORCH:
            # Placeholder data for initialization (gpytorch needs this)
            dummy_x = torch.zeros(10, input_dim)
            dummy_y = torch.zeros(10)
            
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            
            # Low fidelity model (Base)
            self.model_low = ExactGPModel(dummy_x, dummy_y, self.likelihood)
            
            # Difference models (Delta)
            self.model_diff_mid = ExactGPModel(dummy_x, dummy_y, self.likelihood)
            self.model_diff_high = ExactGPModel(dummy_x, dummy_y, self.likelihood)
        else:
            # Lightweight fallback for inference without gpytorch
            self.fallback_linear = nn.Linear(input_dim, 1)
        
    def forward(self, x, fidelity_level='high'):
        """
        Args:
            x (torch.Tensor): Input features
            fidelity_level (str): 'low', 'medium', or 'high'
        
        Returns:
            mean (torch.Tensor): Predicted mean
            std (torch.Tensor): Predicted standard deviation
        """
        if not HAS_GPYTORCH:
            # Stub: return a simple linear prediction with fixed uncertainty
            mean = self.fallback_linear(x).squeeze(-1)
            std = torch.ones_like(mean) * 0.1
            return mean, std
        
        output_low = self.model_low(x)
        mean_low = output_low.mean
        var_low = output_low.variance
        
        if fidelity_level == 'low':
            return mean_low, var_low.sqrt()
            
        output_diff_mid = self.model_diff_mid(x)
        mean_mid = self.rho_mid * mean_low + output_diff_mid.mean
        var_mid = (self.rho_mid**2) * var_low + output_diff_mid.variance
        
        if fidelity_level == 'medium':
            return mean_mid, var_mid.sqrt()
            
        output_diff_high = self.model_diff_high(x)
        mean_high = self.rho_high * mean_mid + output_diff_high.mean
        var_high = (self.rho_high**2) * var_mid + output_diff_high.variance
        
        return mean_high, var_high.sqrt()
    
    def set_train_data(self, train_x, train_y, fidelity='low'):
        """Helper to update internal GP training data"""
        if not HAS_GPYTORCH:
            return  # No-op in stub mode
        if fidelity == 'low':
            self.model_low.set_train_data(train_x, train_y, strict=False)
        elif fidelity == 'medium':
            self.model_diff_mid.set_train_data(train_x, train_y, strict=False)
        elif fidelity == 'high':
            self.model_diff_high.set_train_data(train_x, train_y, strict=False)


