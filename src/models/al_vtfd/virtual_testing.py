import torch
import numpy as np
import sys
import os
# Adjust path to allow relative import if running as script vs module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.data.nasa_loader import get_nasa_loader
from .acquisition import AcquisitionFunction
from .surrogate_gp import SurrogateGP
from .convergence import ConvergenceCriteria

class VirtualTestingFramework:
    """
    Manages the Active Learning loop:
    1. Suggestions (Acquisition)
    2. Execution (Simulation/Experiment)
    3. Update (Model Training)
    4. Convergence Check
    """
    def __init__(self, integrated_model):
        self.model = integrated_model # Analyzed by Feature 1 & 2
        self.surrogate = SurrogateGP()
        self.acquisition = AcquisitionFunction(self.model)
        self.convergence = ConvergenceCriteria()
        
        self.dataset_x = []
        self.dataset_y = []
        


    def run_optimization(self, material_system, budget_tests=80, candidate_pool=None):
        """
        Main loop.
        candidate_pool: torch.Tensor of potential test configurations
        """
        print(f"Starting AL Optimization for {material_system}")
        
        # Load Real Data Pool if candidate_pool is None
        if candidate_pool is None:
            print("Loading real experimental candidates from NASA Dataset...")
            try:
                # Use loader to get real feature vectors
                loader = get_nasa_loader("config/data_config.yaml", batch_size=1000, split='train')
                # Extract all features to form the pool
                all_features = []
                # Just take one batch for pool to avoid memory issues in demo
                for batch in loader:
                    feats = batch['features']
                    # Filter out zeros (invalid loads)
                    valid_mask = torch.sum(feats, dim=1) != 0
                    if valid_mask.any():
                        all_features.append(feats[valid_mask])
                
                if all_features:
                    candidate_pool = torch.cat(all_features, dim=0)
                    # Normalize pool if needed or ensure dimension matches model input
                    # Model expects 6 inputs (phys params), but features are 2048 (signals).
                    # This is a mismatch in the current prototype:
                    # SNPI/CAD-Former expect specific physics inputs [shear, mixity...] or we need an adapter.
                    # For this prototype, we will reduce 2048 -> 6 via PCA or just slice for demo purposes.
                    # Or better: Assume the 'pool' is the *configuration* (params), not the signal.
                    # Since we don't have the params extracted from Excel yet, we will generate synthetic params
                    # coupled with real signal data retrieval later.
                    
                    print(f"Loaded {len(candidate_pool)} real signal samples. Using as proxy for test configs.")
                    # Project to 6 dims for inputs
                    candidate_pool = candidate_pool[:, :6] 
                else:
                    raise ValueError("No valid data found in loader")
            except Exception as e:
                print(f"Warning: Failed to load real data ({e}). Falling back to synthetic pool.")
                candidate_pool = torch.rand(1000, 6) # Placeholder dims
            
        n_tests = 0
        final_accuracy = 0.0
        
        # Initial training on small subset
        print("Performing initial calibration...")
        init_size = min(10, len(candidate_pool))
        init_indices = torch.randperm(len(candidate_pool))[:init_size]
        init_x = candidate_pool[init_indices]
        init_y = self.simulate_test(init_x) # Oracle
        self.surrogate.fit(init_x, init_y, fidelity='low')
        
        # Remove initial from pool
        # (Simplified: just ignore overlap for now or mask)
        
        while n_tests < budget_tests:
            # 1. Compute Acquisition Scores
            existing = torch.stack(self.dataset_x) if self.dataset_x else init_x
            scores = self.acquisition.compute_score(candidate_pool, existing)
            
            # 2. Select Best
            best_idx = torch.argmax(scores)
            next_test = candidate_pool[best_idx]
            
            # 3. "Conduct Test"
            # In a real scenario, this would trigger an experiment.
            # Here we query the Oracle (Simulation/Lookup)
            result = self.simulate_test(next_test.unsqueeze(0)).squeeze()
            
            # 4. Update Dataset & Surrogate
            self.dataset_x.append(next_test)
            self.dataset_y.append(result)
            
            X = torch.stack(self.dataset_x)
            Y = torch.stack(self.dataset_y)
            
            # Train surrogate on high fidelity (assumed 'real' test is high fidelity)
            self.surrogate.fit(X, Y, fidelity='high', steps=20)
            
            n_tests += 1
            print(f"Test {n_tests}/{budget_tests} complete. Result: {result:.4f}")
            
            # 5. Check Convergence
            # Pseudo-validation metric
            val_r2 = 1.0 - torch.var(Y - self.surrogate.predict(X)[0]) / (torch.var(Y) + 1e-6)
            val_unc = self.surrogate.predict(X)[1].mean().item()
            
            self.convergence.update(val_r2.item(), val_unc)
            status = self.convergence.check_details()
            
            if status['r2_met'] and status['unc_met'] and status['stable']:
                print(f"Converged after {n_tests} tests. R2: {status['current_r2']:.4f}")
                final_accuracy = status['current_r2']
                break
                
        return {
            'n_tests_conducted': n_tests,
            'final_accuracy': final_accuracy,
            'cost_savings': (budget_tests - n_tests) / budget_tests
        }
        
    def simulate_test(self, config):
        """
        Oracle: Simulates a physical experiment.
        In reality, this maps config -> result via FEM or Lookup.
        Here we use a dummy function representing 'Energy Release Rate' or similar.
        """
        # Simple Rosenbrock-like or Sphere function for demo
        return torch.sum(config**2, dim=-1)
