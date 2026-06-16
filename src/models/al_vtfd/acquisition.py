import torch
import torch.nn.functional as F
import numpy as np

class CostModel:
    """
    Predicts experimental cost based on test configuration.
    
    Cost factors:
    - Test type (DCB=1.0, ENF=1.2, MMB=1.5)
    - Material cost scaling
    - Instrumentation level
    """
    def __init__(self):
        self.test_costs = {
            'dcb': 1.0,  # Mode I baseline
            'enf': 1.2,  # Mode II
            'mmb': 1.5,  # Mixed-mode
            'fatigue': 3.0,  # Fatigue testing
            'biaxial': 2.5  # Complex loading
        }
        
    def predict_cost(self, test_configs):
        """
        Args:
            test_configs: List of dicts with 'test_type' key or tensor [n, feat_dim]
        
        Returns:
            costs: [n] tensor of relative costs
        """
        if isinstance(test_configs, list):
            costs = torch.tensor([
                self.test_costs.get(cfg.get('test_type', 'dcb'), 1.0)
                for cfg in test_configs
            ])
        else:
            # Tensor input - assume uniform cost for simplicity
            costs = torch.ones(len(test_configs))
        return costs


class AcquisitionFunction:
    """
    Multi-objective acquisition function balancing (Section 7.3):
    1. Uncertainty Reduction (Information Gain)
    2. Exploration (Distance to existing data)
    3. Exploitation (Critical regions)
    4. Cost Efficiency
    
    Implements adaptive weight updates based on optimization progress.
    """
    def __init__(self, model, cost_model=None, weights=None):
        """
        Args:
            model: The trained ML model (e.g., SNPINet or Integrated Framework)
            cost_model: Predicts cost of an experiment (optional)
            weights: Optional dict with keys 'info', 'explore', 'exploit', 'cost'
        """
        self.model = model
        self.cost_model = cost_model or CostModel()
        
        # Weights (can be tuned or adaptive)
        if weights is None:
            weights = {'info': 0.4, 'explore': 0.2, 'exploit': 0.3, 'cost': 0.1}
        
        self.w_info = weights.get('info', 0.4)
        self.w_explore = weights.get('explore', 0.2)
        self.w_exploit = weights.get('exploit', 0.3)
        self.w_cost = weights.get('cost', 0.1)
        
        # Track historical for weight adaptation
        self.info_history = []
        self.exploration_coverage = 0.0
        
    def update_weights(self, stage='early'):
        """
        Adaptively update weights based on optimization stage.
        Early: High exploration
        Mid: Balance  
        Late: High exploitation + cost awareness
        """
        if stage == 'early':
            self.w_info = 0.35
            self.w_explore = 0.35
            self.w_exploit = 0.2
            self.w_cost = 0.1
        elif stage == 'mid':
            self.w_info = 0.4
            self.w_explore = 0.2
            self.w_exploit = 0.3
            self.w_cost = 0.1
        else:  # late
            self.w_info = 0.25
            self.w_explore = 0.1
            self.w_exploit = 0.45
            self.w_cost = 0.2
        
    def compute_score(self, candidates, existing_data=None, test_configs=None):
        """
        Compute acquisition score for a batch of candidates.
        
        Args:
            candidates: [n_candidates, input_dim]
            existing_data: [n_existing, input_dim] for exploration
            test_configs: List of test configuration dicts for cost estimation
        
        Returns:
            total_score: [n_candidates] acquisition values
            components: Dict with individual score components
        """
        self.model.eval()
        device = candidates.device
        
        # 1. Information Gain (Epistemic Uncertainty)
        with torch.no_grad():
            unc_out = self.model.predict_uncertainty(candidates, n_samples=20)
            epistemic = unc_out['epistemic']
            if epistemic.dim() > 1:
                epistemic = epistemic.mean(dim=-1)
                
        # Normalize to [0, 1]
        epsil_range = epistemic.max() - epistemic.min() + 1e-8
        score_info = (epistemic - epistemic.min()) / epsil_range
        
        # 2. Exploration (Euclidean distance to nearest existing point)
        if existing_data is not None and len(existing_data) > 0:
            dists = torch.cdist(candidates, existing_data)
            min_dist, _ = dists.min(dim=1)
            dist_range = min_dist.max() - min_dist.min() + 1e-8
            score_explore = (min_dist - min_dist.min()) / dist_range
        else:
            score_explore = torch.ones(len(candidates), device=device)
             
        # 3. Exploitation (Predicted criticality - high delamination regions)
        with torch.no_grad():
            preds = self.model(candidates)
            if 'prediction' in preds:
                val = preds['prediction']
            elif 'delamination_area' in preds:
                val = preds['delamination_area']
            else:
                val = list(preds.values())[0]

            if isinstance(val, torch.Tensor) and val.dim() > 1:
                val = val.mean(dim=-1)
            
            val_range = val.max() - val.min() + 1e-8
            score_exploit = (val - val.min()) / val_range
             
        # 4. Cost Efficiency
        if test_configs is not None:
            costs = self.cost_model.predict_cost(test_configs).to(device)
        else:
            costs = torch.ones(len(candidates), device=device)
        
        # Lower cost = higher score
        cost_range = costs.max() - costs.min() + 1e-8
        score_cost = 1.0 - (costs - costs.min()) / cost_range
        score_cost = (score_cost - score_cost.min()) / (score_cost.max() - score_cost.min() + 1e-8)
        
        # Total Score
        total_score = (self.w_info * score_info + 
                       self.w_explore * score_explore +
                       self.w_exploit * score_exploit +
                       self.w_cost * score_cost)
        
        components = {
            'info_gain': score_info,
            'exploration': score_explore,
            'exploitation': score_exploit,
            'cost_efficiency': score_cost
        }
                       
        return total_score, components
    
    def select_batch(self, candidates, batch_size, existing_data=None, 
                     test_configs=None, diverse=True):
        """
        Select a batch of candidates for next experiments.
        
        Args:
            candidates: [n_candidates, input_dim]
            batch_size: Number of experiments to select
            existing_data: Existing experimental data
            test_configs: Test configurations for cost estimation
            diverse: If True, use diverse batch selection
        
        Returns:
            selected_indices: Indices of selected candidates
            selected_candidates: Selected candidate tensors
        """
        scores, components = self.compute_score(candidates, existing_data, test_configs)
        
        if diverse:
            # Diverse batch selection using sequential greedy
            selected_idx = []
            remaining_mask = torch.ones(len(candidates), dtype=torch.bool)
            
            # Combine existing and selected for diversity computation
            if existing_data is not None:
                current_selected = existing_data.clone()
            else:
                current_selected = torch.empty(0, candidates.shape[1])
            
            for _ in range(min(batch_size, len(candidates))):
                # Mask already selected
                masked_scores = scores.clone()
                masked_scores[~remaining_mask] = -float('inf')
                
                # Add diversity penalty based on distance to already selected
                if len(selected_idx) > 0:
                    selected_tensor = candidates[selected_idx]
                    dist_to_selected = torch.cdist(candidates, selected_tensor)
                    min_dist_selected, _ = dist_to_selected.min(dim=1)
                    # Add diversity bonus (proportional to distance)
                    diversity_bonus = min_dist_selected / (min_dist_selected.max() + 1e-8)
                    masked_scores = masked_scores + 0.1 * diversity_bonus
                
                # Select top scoring remaining candidate
                best_idx = masked_scores.argmax().item()
                selected_idx.append(best_idx)
                remaining_mask[best_idx] = False
            
            selected_indices = torch.tensor(selected_idx)
        else:
            # Simple top-k selection
            _, selected_indices = torch.topk(scores, min(batch_size, len(scores)))
        
        selected_candidates = candidates[selected_indices]
        
        return selected_indices, selected_candidates

