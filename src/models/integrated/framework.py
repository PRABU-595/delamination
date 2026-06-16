import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from pathlib import Path

# Ensure the project root is on sys.path for robust imports
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from src.models.snpi_net.snpi_net import SNPINet
    from src.models.cad_former.cad_former import CADFormer
except ImportError:
    try:
        # Relative import fallback (when running as part of package)
        from ..snpi_net.snpi_net import SNPINet
        from ..cad_former.cad_former import CADFormer
    except (ImportError, ValueError):
        raise ImportError(
            "Could not import SNPINet or CADFormer. "
            "Ensure you are running from the project root directory."
        )

class IntegratedDelaminationFramework(nn.Module):
    """
    Synergistic Framework (Section 8.1 & 8.2):
    Integrates SNPI-Net (Nonlocal) and CAD-Former (Migration) with active learning.
    """
    def __init__(self, config=None):
        super().__init__()
        if config is None: config = {}
        
        self.snpi_net = SNPINet(config.get('snpi_net', config.get('snpi', {})))
        self.cad_former = CADFormer(config.get('cad_former', {}))
        
        # In a real implementation, AL-VTFD is a wrapper around this model, 
        # but the doc lists it as a member. We will reference the framework class externally
        # or initialize it here if circular dependencies allow.
        # self.al_vtfd = VirtualTestingFramework()... avoid circular import here.
        
    def forward(self, *args, **kwargs):
        """
        Alias for predict_delamination to support standard PyTorch conventions.
        """
        return self.predict_delamination(*args, **kwargs)

    def predict_delamination(self, laminate_config, loading_history, physics_inputs=None, **kwargs):
        """
        Complete delamination prediction with uncertainty quantification (Section 8.2)
        
        Args:
            laminate_config: Dict of laminate properties
            loading_history: Temporal loading sequence
            physics_inputs: Additional inputs like damage_state if available
        """
        
        # 1. Refine with nonlocal mechanics (SNPI-Net)
        # Note: Doc says SNPI computes nonlocal effects.
        # We run SNPI first to get adaptive horizon
        
        # Construct SNPI inputs from config/history features
        # Placeholder feature extraction for SNPI
        # In reality, this would flatten the laminate_config + current state
        if physics_inputs is None:
             # Create dummy physics inputs [batch, 6]
             batch_size = loading_history.shape[0] if loading_history.ndim > 1 else 1
             snpi_input = torch.zeros(batch_size, 6).to(loading_history.device)
        else:
             snpi_input = physics_inputs
             
        snpi_out = self.snpi_net(snpi_input, return_horizon=True)
        horizon = snpi_out['horizon']
        
        # 2. Extract multiscale features & Predict (CAD-Former)
        # We pass the horizon to CAD-Former to modulate attention if supported
        
        batch_size = snpi_input.shape[0]
        device = snpi_input.device
        
        # Micro/Macro features extracted from loading_history if not provided
        # loading_history shape is typically [batch, 2048]
        
        if 'micro_data' in kwargs:
            micro_data = kwargs['micro_data']
        else:
            # Use part of loading_history as proxy for micro-scale AE features
            # Input dim for SimpleMicroEncoder is 80.
            if loading_history.dim() > 1 and loading_history.shape[-1] >= 1024:
                 # Taking slice from signal part (end of buffer)
                 micro_data = loading_history[:, -80:] 
            else:
                 micro_data = torch.zeros(batch_size, 80).to(device)
        
        # Use provided meso_data (X-ray) if available, else zero placeholder
        if 'meso_data' in kwargs:
             meso_data = kwargs['meso_data']
        else:
             meso_data = torch.zeros(batch_size, 3, 32, 32).to(device)

        if 'macro_data' in kwargs:
            macro_data = kwargs['macro_data']
        else:
            # Macro features (Load/Displacement history)
            # Input dim 64. Using indices matching loader structure (262:362 is history)
            if loading_history.dim() > 1 and loading_history.shape[-1] >= 326:
                macro_data = loading_history[:, 262:262+64]
            else:
                macro_data = torch.zeros(batch_size, 64).to(device)
        
        # Construct proper laminate_config dict for CAD-Former
        if isinstance(laminate_config, torch.Tensor):
            if laminate_config.dim() == 2:
                # Flat features [batch, dims] -> [batch, 4, 64]
                batch_size = laminate_config.shape[0]
                dims = laminate_config.shape[1]
                n_interfaces = 4
                feat_per_int = dims // n_interfaces
                reshaped_angles = laminate_config.view(batch_size, n_interfaces, feat_per_int)
            else:
                # Already structured [batch, n, d]
                reshaped_angles = laminate_config
                n_interfaces = reshaped_angles.shape[1]
            
            structured_config = {
                'ply_angles': reshaped_angles,
                'depths': torch.linspace(0, 1, n_interfaces).unsqueeze(0).unsqueeze(-1).expand(reshaped_angles.shape[0], -1, 1).to(device),
                'abd_matrix': torch.zeros(reshaped_angles.shape[0], n_interfaces, 3).to(device),
                'is_symmetric': torch.ones(reshaped_angles.shape[0], n_interfaces, 1).to(device)
            }
        else:
            structured_config = laminate_config
        
        cad_out = self.cad_former(micro_data, meso_data, macro_data, 
                                  structured_config, loading_history)
        
        # 3. Combine predictions
        # Fuse predictions with uncertainty weighting
        # If SNPI uncertainty is high, trust CAD-Former more, and vice versa
        
        snpi_pred = snpi_out['prediction'] # [batch, 3] -> Area, Growth, Damage
        
        # Physics-constrained outputs (ReLU to ensure non-negative)
        # CRITICAL UPDATE for Standalone App: 
        # If one modality strongly predicts damage (e.g. Vision), we should trust it 
        # over the other if the other is near zero (which happens with dummy inputs).
        # We use a soft-max approach or simply max() for safety critical detection.
        
        # Area: Take the maximum of physics and vision to be safe (conservative safety)
        pred_area = torch.max(snpi_pred[:, 0:1], cad_out['delamination_area'])
        pred_area = F.relu(pred_area)
        
        # Growth: Averaging is okay here, but let's bias towards higher growth for safety
        pred_growth = torch.max(snpi_pred[:, 1:2], cad_out['growth_rate'])
        pred_growth = F.softplus(pred_growth)
        
        final_prediction = {
            'delamination_area': pred_area,
            'growth_rate': pred_growth,
            'migration_interface': cad_out['migration_probs'].squeeze(-1),
            'uncertainty': torch.exp(snpi_out['aleatoric_log_var']),
            'aleatoric_log_var': snpi_out['aleatoric_log_var'],
            'horizon': horizon,
            'snpi_raw': snpi_pred,
            'cad_raw': cad_out
        }
        
        return final_prediction

    def predict_uncertainty(self, x, n_samples=50):
        """
        Delegate uncertainty prediction to SNPI-Net.
        Args:
            x: Input tensor [batch, input_dim] (e.g., physics candidates)
            n_samples: Number of MC dropout passes
        """
        return self.snpi_net.predict_uncertainty(x, n_samples=n_samples)

    def train_integrated_system(self, initial_data, budget_tests=80, al_framework=None):
        """
        Coordinated training using active learning (Section 8.2)
        """
        if al_framework is None:
            # Lazy import to avoid circular dependency
            from src.models.al_vtfd.virtual_testing import VirtualTestingFramework
            al_framework = VirtualTestingFramework(self)
        
        print("Starting Integrated Training Loop...")
        
        try:
            al_framework.run_optimization("NASA_CFRP_Composite", budget_tests=budget_tests)
        except Exception as e:
            print(f"AL Loop interrupted: {e}")
            import traceback
            traceback.print_exc()
        
        return self.snpi_net, self.cad_former
