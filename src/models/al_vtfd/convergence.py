import torch
import numpy as np

class ConvergenceCriteria:
    def __init__(self, target_r2=0.95, max_uncertainty=0.1, window_size=5):
        self.target_r2 = target_r2
        self.max_uncertainty = max_uncertainty
        self.window_size = window_size
        self.history = []
        
    def update(self, r2_score, uncertainty_val):
        self.history.append({'r2': r2_score, 'unc': uncertainty_val})
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
    def check_convergence(self):
        if not self.history:
            return False
            
        # Check latest metrics
        latest = self.history[-1]
        
        r2_met = latest['r2'] >= self.target_r2
        unc_met = latest['unc'] <= self.max_uncertainty
        
        # Check stability (variance over window)
        if len(self.history) == self.window_size:
            r2_vals = [h['r2'] for h in self.history]
            stable = np.std(r2_vals) < 0.01
        else:
            stable = False
            
        return r2_met and unc_met and stable

    def check_details(self):
        """
        Returns dictionary of convergence status for debugging/logging.
        """
        if not self.history:
            return {'status': 'No history'}
            
        latest = self.history[-1]
        r2_met = latest['r2'] >= self.target_r2
        unc_met = latest['unc'] <= self.max_uncertainty
        
        stable = False
        if len(self.history) == self.window_size:
            r2_vals = [h['r2'] for h in self.history]
            stable = np.std(r2_vals) < 0.01
            
        return {
            'r2_met': r2_met,
            'unc_met': unc_met,
            'stable': stable,
            'current_r2': latest['r2'],
            'current_unc': latest['unc']
        }
