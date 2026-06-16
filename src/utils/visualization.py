"""
Visualization Utilities for Delamination Framework.

Provides comprehensive visualization tools:
1. Migration pathway graphs
2. Attention weight heatmaps
3. Adaptive horizon spatial maps
4. Uncertainty distribution plots
5. R-curve predictions
6. Dashboard generator
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_migration_graph(migration_matrix: torch.Tensor, 
                        interface_names: List[str] = None,
                        threshold: float = 0.1,
                        save_path: str = None) -> Figure:
    """
    Visualize delamination migration pathways as a directed graph.
    
    Args:
        migration_matrix: [n_interfaces, n_interfaces] probability matrix
        interface_names: Optional interface labels (e.g., ['0/45', '45/-45', ...])
        threshold: Minimum probability to show edge
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure
    """
    n = migration_matrix.shape[0]
    if interface_names is None:
        interface_names = [f'I{i}' for i in range(n)]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Circular layout for interfaces
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radius = 3
    positions = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    
    # Draw nodes
    for i, (x, y) in enumerate(positions):
        circle = plt.Circle((x, y), 0.3, color='steelblue', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, interface_names[i], ha='center', va='center', 
                fontsize=9, fontweight='bold', color='white')
    
    # Draw edges (migration pathways)
    if isinstance(migration_matrix, torch.Tensor):
        migration_matrix = migration_matrix.detach().numpy()
    
    for i in range(n):
        for j in range(n):
            if i != j and migration_matrix[i, j] > threshold:
                prob = migration_matrix[i, j]
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                
                # Arrow from i to j
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                dx, dy = dx/length * (length - 0.5), dy/length * (length - 0.5)
                
                arrow = mpatches.FancyArrowPatch(
                    (x1 + 0.3*dx/length, y1 + 0.3*dy/length),
                    (x1 + dx, y1 + dy),
                    connectionstyle="arc3,rad=0.1",
                    arrowstyle="->,head_width=0.15,head_length=0.1",
                    linewidth=1 + 3*prob,
                    color=plt.cm.Reds(0.3 + 0.7*prob),
                    alpha=0.7 + 0.3*prob
                )
                ax.add_patch(arrow)
                
                # Label probability
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                ax.text(mid_x, mid_y, f'{prob:.2f}', fontsize=8, alpha=0.8)
    
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Delamination Migration Pathways', fontsize=14, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_attention_heatmap(attention_weights: torch.Tensor,
                          layer_idx: int = 0,
                          interface_names: List[str] = None,
                          save_path: str = None) -> Figure:
    """
    Visualize spatial attention weights as heatmap.
    
    Args:
        attention_weights: [n_layers, n_interfaces, n_interfaces] or single layer
        layer_idx: Which layer to visualize
        interface_names: Optional interface labels
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    if isinstance(attention_weights, list):
        attn = attention_weights[layer_idx]
    elif attention_weights.dim() == 3:
        attn = attention_weights[layer_idx]
    else:
        attn = attention_weights
    
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().numpy()
    
    # Average over batch if present
    if attn.ndim == 3:
        attn = attn.mean(axis=0)
    
    n = attn.shape[0]
    if interface_names is None:
        interface_names = [f'I{i}' for i in range(n)]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    im = ax.imshow(attn, cmap='viridis', aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight', fontsize=10)
    
    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(interface_names, rotation=45, ha='right')
    ax.set_yticklabels(interface_names)
    
    ax.set_xlabel('Key Interface', fontsize=11)
    ax.set_ylabel('Query Interface', fontsize=11)
    ax.set_title(f'Spatial Attention Weights (Layer {layer_idx})', 
                 fontsize=12, fontweight='bold')
    
    # Add values
    for i in range(n):
        for j in range(n):
            text_color = 'white' if attn[i, j] > 0.5 * attn.max() else 'black'
            ax.text(j, i, f'{attn[i,j]:.2f}', ha='center', va='center',
                   fontsize=8, color=text_color)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_adaptive_horizon(horizon_values: torch.Tensor,
                         positions: torch.Tensor = None,
                         save_path: str = None) -> Figure:
    """
    Visualize spatially-varying adaptive horizon as heatmap.
    
    Args:
        horizon_values: Horizon δ values [n_points] or [grid_x, grid_y]
        positions: Optional spatial positions [n_points, 2]
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    if isinstance(horizon_values, torch.Tensor):
        horizon_values = horizon_values.detach().numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    if horizon_values.ndim == 1:
        # 1D - create line plot
        ax.plot(horizon_values, 'b-', linewidth=2, label='Horizon δ(x)')
        ax.fill_between(range(len(horizon_values)), 0, horizon_values, alpha=0.3)
        ax.set_xlabel('Spatial Position', fontsize=11)
        ax.set_ylabel('Horizon δ (mm)', fontsize=11)
        ax.legend()
    else:
        # 2D grid - create heatmap
        im = ax.imshow(horizon_values, cmap='plasma', aspect='auto', origin='lower')
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Horizon δ (mm)', fontsize=10)
        ax.set_xlabel('X Position', fontsize=11)
        ax.set_ylabel('Y Position', fontsize=11)
    
    ax.set_title('Adaptive Nonlocal Horizon', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_uncertainty_distribution(epistemic: torch.Tensor,
                                  aleatoric: torch.Tensor,
                                  predictions: torch.Tensor = None,
                                  save_path: str = None) -> Figure:
    """
    Visualize uncertainty decomposition.
    
    Args:
        epistemic: Epistemic (model) uncertainty [n_samples]
        aleatoric: Aleatoric (data) uncertainty [n_samples]
        predictions: Optional predictions for context
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    if isinstance(epistemic, torch.Tensor):
        epistemic = epistemic.detach().numpy().flatten()
    if isinstance(aleatoric, torch.Tensor):
        aleatoric = aleatoric.detach().numpy().flatten()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 1. Histogram comparison
    ax1 = axes[0]
    ax1.hist(epistemic, bins=30, alpha=0.6, label='Epistemic', color='steelblue')
    ax1.hist(aleatoric, bins=30, alpha=0.6, label='Aleatoric', color='coral')
    ax1.set_xlabel('Uncertainty', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Uncertainty Distribution', fontsize=11, fontweight='bold')
    ax1.legend()
    
    # 2. Scatter: Epistemic vs Aleatoric
    ax2 = axes[1]
    total = epistemic + aleatoric
    sc = ax2.scatter(aleatoric, epistemic, c=total, cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(sc, ax=ax2, label='Total Uncertainty')
    ax2.set_xlabel('Aleatoric Uncertainty', fontsize=10)
    ax2.set_ylabel('Epistemic Uncertainty', fontsize=10)
    ax2.set_title('Uncertainty Decomposition', fontsize=11, fontweight='bold')
    
    # 3. Pie chart of average contributions
    ax3 = axes[2]
    avg_epi = epistemic.mean()
    avg_ale = aleatoric.mean()
    ax3.pie([avg_epi, avg_ale], labels=['Epistemic', 'Aleatoric'],
            colors=['steelblue', 'coral'], autopct='%1.1f%%',
            explode=(0.02, 0.02), shadow=True)
    ax3.set_title('Average Contribution', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_rcurve(crack_lengths: torch.Tensor,
               predicted_R: torch.Tensor,
               true_R: torch.Tensor = None,
               save_path: str = None) -> Figure:
    """
    Plot R-curve (resistance curve) comparison.
    
    Args:
        crack_lengths: Crack lengths [n_points]
        predicted_R: Predicted fracture resistance
        true_R: Ground truth R values (optional)
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    if isinstance(crack_lengths, torch.Tensor):
        crack_lengths = crack_lengths.detach().numpy()
    if isinstance(predicted_R, torch.Tensor):
        predicted_R = predicted_R.detach().numpy()
    if true_R is not None and isinstance(true_R, torch.Tensor):
        true_R = true_R.detach().numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(crack_lengths, predicted_R, 'b-', linewidth=2, 
            label='Predicted R', marker='o', markersize=4)
    
    if true_R is not None:
        ax.plot(crack_lengths, true_R, 'r--', linewidth=2, 
                label='Ground Truth', marker='s', markersize=4)
    
    ax.set_xlabel('Crack Length (mm)', fontsize=11)
    ax.set_ylabel('Fracture Resistance G_R (kJ/m²)', fontsize=11)
    ax.set_title('R-Curve Prediction', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_dashboard(outputs: Dict[str, torch.Tensor],
                       save_dir: str = None) -> Dict[str, Figure]:
    """
    Generate comprehensive visualization dashboard.
    
    Args:
        outputs: Model outputs dict with keys:
            - 'migration_probs' or 'migration_interface'
            - 'spatial_attention'
            - 'horizon'
            - 'uncertainty' or 'aleatoric'/'epistemic'
            - 'delamination_area', 'growth_rate'
        save_dir: Directory to save all figures
    
    Returns:
        Dict of figure names to Figure objects
    """
    figures = {}
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Migration graph
    if 'migration_probs' in outputs or 'migration_interface' in outputs:
        mig_key = 'migration_probs' if 'migration_probs' in outputs else 'migration_interface'
        mig_data = outputs[mig_key].squeeze()
        if mig_data.dim() >= 2:
            n = mig_data.shape[-1] if mig_data.dim() == 2 else mig_data.shape[-2]
            mig_matrix = mig_data.mean(dim=0) if mig_data.dim() > 2 else mig_data
            if mig_matrix.dim() == 1:
                # Convert to matrix form
                mig_matrix = mig_matrix.unsqueeze(0).expand(n, -1) * 0.5
            save_path = str(save_dir / 'migration_graph.png') if save_dir else None
            figures['migration'] = plot_migration_graph(mig_matrix, save_path=save_path)
    
    # 2. Attention heatmap
    if 'spatial_attention' in outputs:
        attn = outputs['spatial_attention']
        if isinstance(attn, list) and len(attn) > 0:
            attn = attn[0]  # First layer
        if attn.numel() > 1:
            save_path = str(save_dir / 'attention_heatmap.png') if save_dir else None
            figures['attention'] = plot_attention_heatmap(attn, save_path=save_path)
    
    # 3. Horizon plot
    if 'horizon' in outputs:
        horizon = outputs['horizon'].squeeze()
        save_path = str(save_dir / 'adaptive_horizon.png') if save_dir else None
        figures['horizon'] = plot_adaptive_horizon(horizon, save_path=save_path)
    
    # 4. Uncertainty
    if 'uncertainty' in outputs:
        unc = outputs['uncertainty'].squeeze()
        # Assume first half epistemic, second half aleatoric (or same)
        epistemic = unc[..., :unc.shape[-1]//2] if unc.dim() > 1 else unc
        aleatoric = unc[..., unc.shape[-1]//2:] if unc.dim() > 1 else unc
        save_path = str(save_dir / 'uncertainty.png') if save_dir else None
        figures['uncertainty'] = plot_uncertainty_distribution(
            epistemic.abs(), aleatoric.abs(), save_path=save_path
        )
    
    print(f"Generated {len(figures)} visualizations")
    if save_dir:
        print(f"Saved to: {save_dir}")
    
    return figures


def create_html_report(figures_dir: str, output_path: str = None):
    """
    Create HTML report from saved figures.
    
    Args:
        figures_dir: Directory containing PNG figures
        output_path: Output HTML file path
    """
    figures_dir = Path(figures_dir)
    if output_path is None:
        output_path = figures_dir / 'report.html'
    
    png_files = list(figures_dir.glob('*.png'))
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Delamination Framework Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        h1 { color: #2c3e50; }
        .figure { 
            background: white; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .figure img { max-width: 100%; height: auto; }
        .figure h3 { color: #34495e; margin-top: 0; }
    </style>
</head>
<body>
    <h1>Delamination Prediction Framework - Visualization Report</h1>
"""
    
    for png in png_files:
        name = png.stem.replace('_', ' ').title()
        html_content += f"""
    <div class="figure">
        <h3>{name}</h3>
        <img src="{png.name}" alt="{name}">
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to: {output_path}")
