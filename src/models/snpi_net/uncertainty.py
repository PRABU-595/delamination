import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyNetwork(nn.Module):
    """
    Network head that estimates both mean prediction and aleatoric uncertainty (data noise).
    Designed to be used with Monte Carlo Dropout for epistemic uncertainty.
    """
    def __init__(self, input_dim=64, hidden_dims=[64, 32], output_dim=3, dropout_rate=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate)) # Critical for MC Dropout
            prev_dim = dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Dual heads: one for mean, one for variance (aleatoric uncertainty)
        self.mean_head = nn.Linear(prev_dim, output_dim)
        self.log_var_head = nn.Linear(prev_dim, output_dim) 
        
    def forward(self, x):
        """
        Returns:
            mean (torch.Tensor): Predicted mean values
            log_var (torch.Tensor): Log of the variance (for numerical stability)
        """
        features = self.feature_extractor(x)
        
        mean = self.mean_head(features)
        log_var = self.log_var_head(features)
        
        # Numerical stability clamp: prevent loss explosion
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        
        return mean, log_var

    def predict_with_uncertainty(self, x, n_samples=50):
        """
        Perform Monte Carlo Dropout to estimate epistemic uncertainty.
        
        Args:
            x (torch.Tensor): Input features
            n_samples (int): Number of MC stochastic forward passes
            
        Returns:
            mean_pred (torch.Tensor): Averaged mean prediction
            epistemic_var (torch.Tensor): Variance due to model uncertainty
            aleatoric_var (torch.Tensor): Expected data noise variance
        """
        self.train() # Enable dropout active during inference
        
        means_list = []
        log_vars_list = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                mean, log_var = self.forward(x)
                means_list.append(mean)
                log_vars_list.append(log_var)
        
        # Stack results: [n_samples, batch, output_dim]
        means_stack = torch.stack(means_list)
        log_vars_stack = torch.stack(log_vars_list)
        
        # Epistemic Uncertainty = Variance of the means
        mean_pred = torch.mean(means_stack, dim=0)
        epistemic_var = torch.var(means_stack, dim=0)
        
        # Aleatoric Uncertainty = Average of the variances
        # Convert log_var to var first: exp(log_var)
        aleatoric_var = torch.mean(torch.exp(log_vars_stack), dim=0)
        
        return mean_pred, epistemic_var, aleatoric_var
