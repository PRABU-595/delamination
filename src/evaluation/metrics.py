"""
Evaluation Metrics for Delamination Prediction Framework.

Implements metrics from Section 9 of the research document:
- RMSE, R², MAE for regression
- Migration accuracy
- Uncertainty calibration (95% CI coverage)
- Computational cost comparison
"""
import torch
import numpy as np
from typing import Dict, Tuple, Optional


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Square Error."""
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def r_squared(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Coefficient of determination (R²)."""
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    return (1 - ss_res / (ss_tot + 1e-8)).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return torch.mean(torch.abs(pred - target)).item()


def mape(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Percentage Error."""
    mask = target.abs() > 1e-8
    if mask.sum() == 0:
        return 0.0
    return (torch.mean(torch.abs((target[mask] - pred[mask]) / target[mask])) * 100).item()


def migration_accuracy(pred_probs: torch.Tensor, true_migration: torch.Tensor, 
                       threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute migration prediction accuracy metrics.
    
    Args:
        pred_probs: Predicted migration probabilities [batch, n_interfaces]
        true_migration: Ground truth binary migration [batch, n_interfaces]
        threshold: Classification threshold
    
    Returns:
        Dict with accuracy, precision, recall, f1
    """
    pred_binary = (pred_probs > threshold).float()
    
    tp = ((pred_binary == 1) & (true_migration == 1)).sum().float()
    fp = ((pred_binary == 1) & (true_migration == 0)).sum().float()
    fn = ((pred_binary == 0) & (true_migration == 1)).sum().float()
    tn = ((pred_binary == 0) & (true_migration == 0)).sum().float()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }


def uncertainty_calibration(pred_mean: torch.Tensor, pred_std: torch.Tensor,
                            target: torch.Tensor, confidence: float = 0.95) -> Dict[str, float]:
    """
    Evaluate uncertainty calibration using prediction intervals.
    
    Good calibration: ~95% of targets should fall within 95% CI.
    
    Args:
        pred_mean: Predicted mean values
        pred_std: Predicted standard deviation
        target: Ground truth values
        confidence: Confidence level (default 0.95)
    
    Returns:
        Dict with coverage, mean_width, and calibration_error
    """
    # For 95% CI, use ~1.96 std from mean
    z_score = 1.96 if confidence == 0.95 else 2.576  # 99% CI
    
    lower = pred_mean - z_score * pred_std
    upper = pred_mean + z_score * pred_std
    
    # Coverage: fraction of targets within interval
    in_interval = ((target >= lower) & (target <= upper)).float()
    coverage = in_interval.mean().item()
    
    # Mean interval width (sharpness)
    mean_width = (upper - lower).mean().item()
    
    # Calibration error: |coverage - confidence|
    calibration_error = abs(coverage - confidence)
    
    return {
        'coverage': coverage,
        'mean_width': mean_width,
        'calibration_error': calibration_error,
        'is_well_calibrated': coverage >= confidence - 0.05
    }


def compute_all_metrics(predictions: Dict[str, torch.Tensor],
                        targets: Dict[str, torch.Tensor],
                        uncertainty: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: Dict with 'area', 'growth_rate', 'migration'
        targets: Dict with ground truth values
        uncertainty: Optional uncertainty estimates (std)
    
    Returns:
        Dict with all computed metrics
    """
    metrics = {}
    
    # Delamination area metrics
    if 'area' in predictions and 'area' in targets:
        pred_area = predictions['area'].flatten()
        true_area = targets['area'].flatten()
        metrics['area_rmse'] = rmse(pred_area, true_area)
        metrics['area_r2'] = r_squared(pred_area, true_area)
        metrics['area_mae'] = mae(pred_area, true_area)
        metrics['area_mape'] = mape(pred_area, true_area)
    
    # Growth rate metrics
    if 'growth_rate' in predictions and 'growth_rate' in targets:
        pred_growth = predictions['growth_rate'].flatten()
        true_growth = targets['growth_rate'].flatten()
        metrics['growth_rmse'] = rmse(pred_growth, true_growth)
        metrics['growth_r2'] = r_squared(pred_growth, true_growth)
    
    # Migration metrics
    if 'migration' in predictions and 'migration' in targets:
        mig_metrics = migration_accuracy(
            predictions['migration'], 
            targets['migration']
        )
        metrics.update({f'migration_{k}': v for k, v in mig_metrics.items()})
    
    # Uncertainty calibration
    if uncertainty is not None and 'area' in predictions and 'area' in targets:
        pred_std = torch.sqrt(torch.exp(uncertainty))
        cal_metrics = uncertainty_calibration(
            predictions['area'].flatten(),
            pred_std.flatten(),
            targets['area'].flatten()
        )
        metrics.update({f'uncertainty_{k}': v for k, v in cal_metrics.items()})
    
    return metrics


class BenchmarkSuite:
    """
    Benchmark suite for comparing against literature results.
    
    Target metrics from Section 9:
    - Mode I: RMSE < 0.096 (vs Jahanshahi et al.)
    - Migration: Accuracy > 85%
    - R-curve: RMSE < 10%
    - Fatigue: Paris law ±20%
    """
    
    def __init__(self):
        self.literature_targets = {
            'mode_i_rmse': 0.096,
            'migration_accuracy': 0.85,
            'rcurve_rmse_percent': 10.0,
            'fatigue_paris_error': 20.0
        }
        self.results = {}
    
    def evaluate_mode_i(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Evaluate Mode I delamination prediction."""
        metrics = {
            'rmse': rmse(predictions, targets),
            'r2': r_squared(predictions, targets),
            'mae': mae(predictions, targets)
        }
        metrics['passes_benchmark'] = metrics['rmse'] < self.literature_targets['mode_i_rmse']
        self.results['mode_i'] = metrics
        return metrics
    
    def evaluate_migration(self, pred_probs: torch.Tensor, 
                           true_migration: torch.Tensor) -> Dict:
        """Evaluate migration prediction accuracy."""
        metrics = migration_accuracy(pred_probs, true_migration)
        metrics['passes_benchmark'] = metrics['accuracy'] > self.literature_targets['migration_accuracy']
        self.results['migration'] = metrics
        return metrics
    
    def evaluate_rcurve(self, predicted_R: torch.Tensor, 
                        true_R: torch.Tensor) -> Dict:
        """Evaluate R-curve prediction."""
        rmse_val = rmse(predicted_R, true_R)
        rmse_percent = (rmse_val / true_R.mean().item()) * 100
        
        metrics = {
            'rmse': rmse_val,
            'rmse_percent': rmse_percent,
            'r2': r_squared(predicted_R, true_R)
        }
        metrics['passes_benchmark'] = rmse_percent < self.literature_targets['rcurve_rmse_percent']
        self.results['rcurve'] = metrics
        return metrics
    
    def evaluate_fatigue(self, predicted_dadN: torch.Tensor,
                         true_dadN: torch.Tensor) -> Dict:
        """Evaluate fatigue (Paris law) prediction."""
        # Compute log-space error for Paris law
        log_pred = torch.log10(predicted_dadN.clamp(min=1e-12))
        log_true = torch.log10(true_dadN.clamp(min=1e-12))
        
        metrics = {
            'log_rmse': rmse(log_pred, log_true),
            'mape': mape(predicted_dadN, true_dadN),
            'r2': r_squared(predicted_dadN, true_dadN)
        }
        metrics['passes_benchmark'] = metrics['mape'] < self.literature_targets['fatigue_paris_error']
        self.results['fatigue'] = metrics
        return metrics
    
    def summary(self) -> Dict:
        """Return summary of all benchmark results."""
        return {
            'results': self.results,
            'all_passed': all(r.get('passes_benchmark', False) for r in self.results.values())
        }
