import torch
import gpytorch
from ..snpi_net.multi_fidelity import MultiFidelityGP

class SurrogateGP:
    """
    Wrapper around MultiFidelityGP to manage training and state for the Active Learning loop.
    Acts as the 'Surrogate Model' described in Feature 3.
    """
    def __init__(self, input_dim=6):
        self.model = MultiFidelityGP(input_dim)
        self.likelihood = self.model.likelihood
        self.optimizer = None
        self.mll = None # Marginal Log Likelihood
        
        self.train_x = None
        self.train_y = None
        
    def fit(self, x, y, fidelity='low', steps=50, lr=0.1):
        """
        Fit the GP to new data for a specific fidelity level.
        Args:
            x: Input features
            y: Targets
            fidelity: 'low', 'medium', or 'high'
        """
        self.train_x = x
        self.train_y = y
        
        # Update model data for specific fidelity
        self.model.set_train_data(x, y, fidelity=fidelity)
        
        self.model.train()
        self.likelihood.train()
        
        # Select sub-model to train
        if fidelity == 'low':
            sub_model = self.model.model_low
        elif fidelity == 'medium':
            sub_model = self.model.model_diff_mid
        elif fidelity == 'high':
            sub_model = self.model.model_diff_high
        else:
            raise ValueError(f"Unknown fidelity: {fidelity}")
            
        # Re-init optimizer for specific parameters
        # In a full MF-GP, we might want to train joint params (like rho), 
        # but here we train components sequentially or independently
        params = list(sub_model.parameters())
        if fidelity == 'medium':
            params.append(self.model.rho_mid)
        elif fidelity == 'high':
            params.append(self.model.rho_high)
            
        self.optimizer = torch.optim.Adam(params, lr=lr)
        
        # MLL for the specific sub-model
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, sub_model)
        
        for i in range(steps):
            self.optimizer.zero_grad()
            output = sub_model(x) 
            loss = -self.mll(output, y)
            loss.backward()
            self.optimizer.step()
            
        self.model.eval()
        self.likelihood.eval()
        
    def predict(self, x):
        """
        Returns mean, std
        """
        self.model.eval()
        with torch.no_grad():
            mean, std = self.model(x, fidelity_level='high')
        return mean, std
