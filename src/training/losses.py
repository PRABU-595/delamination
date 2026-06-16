import torch
import torch.nn as nn
import torch.nn.functional as F

class HeteroscedasticNLL(nn.Module):
    """
    Negative Log-Likelihood loss for heteroscedastic regression.
    Handles uncertainty-aware predictions where model outputs mean and log_var.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, mean, log_var, target):
        """
        NLL = 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2)
        
        Args:
            mean: Predicted mean [batch, output_dim]
            log_var: Log variance (aleatoric) [batch, output_dim]
            target: Ground truth [batch, output_dim]
        
        Returns:
            loss: Scalar NLL loss
        """
        precision = torch.exp(-log_var)  # 1/sigma^2
        squared_error = (target - mean) ** 2
        
        loss = 0.5 * (log_var + precision * squared_error)
        return loss.mean()


class MigrationCrossEntropy(nn.Module):
    """
    Cross-entropy loss for migration prediction.
    Handles both binary (migrate/stay) and multi-class (which interface) predictions.
    """
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
        
    def forward(self, migration_probs, migration_targets):
        """
        Args:
            migration_probs: [batch, n_interfaces] or [batch, n_interfaces, n_interfaces]
            migration_targets: [batch, n_interfaces] binary or [batch] interface index
        
        Returns:
            loss: Cross-entropy loss
        """
        if migration_probs.dim() == 2:
            # Binary migration per interface
            loss = F.binary_cross_entropy(
                migration_probs, 
                migration_targets.float(),
                weight=self.weights
            )
        else:
            # Full migration matrix - flatten and compare
            probs_flat = migration_probs.view(-1)
            targets_flat = migration_targets.view(-1)
            loss = F.binary_cross_entropy(probs_flat, targets_flat.float())
            
        return loss


class RCurveConstraint(nn.Module):
    """
    Physics constraint for R-curve behavior.
    R (fracture resistance) should monotonically increase with crack length
    until reaching plateau.
    """
    def __init__(self, plateau_value=300.0):
        super().__init__()
        self.plateau_value = plateau_value  # Typical G_Rss value in J/m²
        
    def forward(self, predicted_R, crack_lengths):
        """
        Args:
            predicted_R: Predicted fracture resistance [batch, n_points]
            crack_lengths: Corresponding crack lengths [batch, n_points]
        
        Returns:
            loss: Penalty for non-monotonic or non-physical R values
        """
        # Sort by crack length to check monotonicity
        sorted_idx = torch.argsort(crack_lengths, dim=-1)
        R_sorted = torch.gather(predicted_R, -1, sorted_idx)
        
        # Compute differences (should be non-negative for monotonic increase)
        R_diff = R_sorted[:, 1:] - R_sorted[:, :-1]
        
        # Penalize decreasing R (except near plateau)
        # Allow small decreases near plateau
        below_plateau = R_sorted[:, :-1] < 0.95 * self.plateau_value
        monotonicity_penalty = F.relu(-R_diff) * below_plateau.float()
        
        # Penalize values exceeding plateau significantly
        plateau_violation = F.relu(predicted_R - 1.1 * self.plateau_value)
        
        loss = monotonicity_penalty.mean() + 0.1 * plateau_violation.mean()
        return loss


class PhysicsInformedLoss(nn.Module):
    """
    Combined loss function for delamination prediction (Section 9.1):
    
    L = w_mse * L_data 
      + w_nll * L_heteroscedastic  
      + w_mig * L_migration
      + w_rcurve * L_rcurve
      + w_physics * L_constraints
    """
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = {
                'mse': 1.0, 
                'nll': 0.5,
                'migration': 0.3,
                'rcurve': 0.1,
                'physics': 0.1,
                'energy': 0.05,
                'thermo': 0.05
            }
        self.weights = weights
        
        # Sub-losses
        self.heteroscedastic_nll = HeteroscedasticNLL()
        self.migration_ce = MigrationCrossEntropy()
        self.rcurve_constraint = RCurveConstraint()
        
    def forward(self, predictions, targets, physics_params=None, physics_state=None):
        """
        Args:
            predictions: dict with keys:
                - 'delamination_area': [batch, 1]
                - 'growth_rate': [batch, 1]
                - 'aleatoric_log_var': [batch, output_dim] (optional)
                - 'migration_probs': [batch, n_interfaces] (optional)
                - 'predicted_R': [batch, n_points] (optional for R-curve)
                - 'horizon': [batch, 1] (optional)
            targets: dict with keys:
                - 'area': Ground truth area
                - 'growth_rate': Ground truth growth rate
                - 'migration': Migration ground truth (optional)
            physics_params: dict with:
                - 'crack_lengths': For R-curve validation
            physics_state: dict with:
                - 'strain_energy': Strain energy density
                - 'fracture_energy': Fracture energy Gc
                - 'work_done': External work
                - 'dissipated_energy': Dissipated energy
        
        Returns:
            total_loss: Combined loss scalar
            loss_components: Dict with individual loss values for logging
        """
        loss_components = {}
        batch_size = predictions['delamination_area'].shape[0]
        
        # 1. Data Loss (MSE)
        mse_area = F.mse_loss(predictions['delamination_area'], targets['area'])
        mse_growth = F.mse_loss(predictions['growth_rate'], targets['growth_rate'])
        l_data = mse_area + mse_growth
        loss_components['mse'] = l_data.item()
        
        # 2. Heteroscedastic NLL (if uncertainty is available)
        l_nll = torch.tensor(0.0, device=predictions['delamination_area'].device)
        if 'aleatoric_log_var' in predictions:
            pred_mean = torch.cat([
                predictions['delamination_area'], 
                predictions['growth_rate']
            ], dim=-1)
            target_combined = torch.cat([
                targets['area'], 
                targets['growth_rate']
            ], dim=-1)
            
            log_var = predictions['aleatoric_log_var']
            if log_var.shape[-1] != pred_mean.shape[-1]:
                log_var = log_var[..., :pred_mean.shape[-1]]
                
            l_nll = self.heteroscedastic_nll(pred_mean, log_var, target_combined)
            loss_components['nll'] = l_nll.item()
        
        # 3. Migration Cross-Entropy
        l_mig = torch.tensor(0.0, device=predictions['delamination_area'].device)
        if 'migration_probs' in predictions and 'migration' in targets:
            # Handle potential shape mismatch (if target is index, prob is vector)
            t = targets['migration']
            p = predictions['migration_probs']
            if t.dim() == 1 and p.dim() == 2:
                # p: [batch, n_interfaces], t: [batch] indices
                 l_mig = F.cross_entropy(p, t.long())
            else:
                 l_mig = self.migration_ce(p, t)
            loss_components['migration'] = l_mig.item()
        
        # 4. R-curve Constraint
        l_rcurve = torch.tensor(0.0, device=predictions['delamination_area'].device)
        if 'predicted_R' in predictions and physics_params is not None:
            if 'crack_lengths' in physics_params:
                l_rcurve = self.rcurve_constraint(
                    predictions['predicted_R'],
                    physics_params['crack_lengths']
                )
                loss_components['rcurve'] = l_rcurve.item()
        
        # 5. Physics Constraints (Constraints & Thermodynamics)
        l_physics = torch.tensor(0.0, device=predictions['delamination_area'].device)
        l_thermo = torch.tensor(0.0, device=predictions['delamination_area'].device)
        
        # A. Non-negative Area & Growth (Thermodynamic Consistency)
        neg_area = F.relu(-predictions['delamination_area'])
        neg_growth = F.relu(-predictions['growth_rate'])
        l_thermo += neg_area.mean() + 0.5 * neg_growth.mean()
        
        # B. Horizon Consistency (Peridynamics)
        if 'horizon' in predictions:
            horizon = predictions['horizon']
            # Horizon should be > 0 and reasonable (e.g., < 5mm)
            horizon_penalty = F.relu(-horizon) + F.relu(horizon - 5e-3)
            l_physics += 0.1 * horizon_penalty.mean()
            
        loss_components['physics'] = l_physics.item()
        loss_components['thermo'] = l_thermo.item()
        
        # 6. Energy Conservation (First Law)
        l_energy = torch.tensor(0.0, device=predictions['delamination_area'].device)
        if physics_state is not None:
            # Check: Work Done = Strain Energy + Fracture/Dissipated Energy
            # W = U + Gamma
            # We assume physics_state contains terms normalized per unit volume or similar
            if all(k in physics_state for k in ['work_done', 'strain_energy', 'dissipated_energy']):
                 lhs = physics_state['work_done']
                 rhs = physics_state['strain_energy'] + physics_state['dissipated_energy']
                 energy_residual = torch.abs(lhs - rhs)
                 l_energy = torch.mean(energy_residual)
                 loss_components['energy'] = l_energy.item()

        # Total weighted loss
        total_loss = (
            self.weights.get('mse', 1.0) * l_data +
            self.weights.get('nll', 0.5) * l_nll +
            self.weights.get('migration', 0.3) * l_mig +
            self.weights.get('rcurve', 0.1) * l_rcurve +
            self.weights.get('physics', 0.1) * l_physics +
            self.weights.get('thermo', 0.05) * l_thermo +
            self.weights.get('energy', 0.05) * l_energy
        )
        
        loss_components['total'] = total_loss.item()
        
        return total_loss, loss_components


class MultiTaskLoss(nn.Module):
    """
    Learnable multi-task loss weighting (Kendall et al. style).
    Automatically balances task losses using homoscedastic uncertainty.
    """
    def __init__(self, n_tasks=4):
        super().__init__()
        # Log variance parameters (learnable)
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        
    def forward(self, losses):
        """
        Args:
            losses: List of task losses [L1, L2, L3, L4]
        
        Returns:
            weighted_loss: Automatically weighted total loss
        """
        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
        
        return sum(weighted_losses)

