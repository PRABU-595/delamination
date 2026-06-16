import torch
import torch.nn.functional as F
import numpy as np

def compute_rmse(predictions, targets):
    """Compute Root Mean Squared Error."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    return np.sqrt(np.mean((predictions - targets)**2))

def compute_r2(predictions, targets):
    """Compute Coefficient of Determination (R^2)."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
        
    ss_res = np.sum((targets - predictions)**2)
    ss_tot = np.sum((targets - np.mean(targets))**2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def compute_migration_accuracy(pred_probs, target_indices):
    """
    Compute classification accuracy for delamination migration interface.
    Args:
        pred_probs: [batch, n_classes] probabilities or logits
        target_indices: [batch] ground truth class indices
    """
    if isinstance(pred_probs, torch.Tensor):
        pred_labels = torch.argmax(pred_probs, dim=1).detach().cpu().numpy()
    else:
        pred_labels = np.argmax(pred_probs, axis=1)
        
    if isinstance(target_indices, torch.Tensor):
        target_indices = target_indices.detach().cpu().numpy()
        
    return np.mean(pred_labels == target_indices)

def compute_uncertainty_calibration(means, stds, targets, interval=0.95):
    """
    Check what fraction of targets fall within the predicted confidence interval.
    Args:
        means: [batch] predicted means
        stds: [batch] predicted standard deviations
        targets: [batch] actual values
        interval: Confidence level (e.g. 0.95 for 2 stds roughly)
    """
    if isinstance(means, torch.Tensor): means = means.detach().cpu().numpy()
    if isinstance(stds, torch.Tensor): stds = stds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor): targets = targets.detach().cpu().numpy()
    
    # Z-score for interval (assuming Gaussian)
    # 0.95 -> +/- 1.96 std
    from scipy.stats import norm
    z = norm.ppf(0.5 + interval/2)
    
    lower = means - z * stds
    upper = means + z * stds
    
    in_bounds = (targets >= lower) & (targets <= upper)
    coverage = np.mean(in_bounds)
    return coverage

def compute_negative_log_likelihood(means, logs_vars, targets):
    """
    Compute Heteroscedastic Negative Log Likelihood.
    """
    # NLL = 0.5 * (log(var) + (y - mu)^2 / var)
    if isinstance(means, np.ndarray):
        means = torch.from_numpy(means)
        logs_vars = torch.from_numpy(logs_vars)
        targets = torch.from_numpy(targets)
        
    precision = torch.exp(-logs_vars)
    loss = 0.5 * precision * (targets - means)**2 + 0.5 * logs_vars
    return loss.mean().item()
