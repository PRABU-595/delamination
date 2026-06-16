"""
Comprehensive Metrics & Visualization Generator.

Generates publication-ready metrics and plots from the trained model:
1. Confusion Matrix (binary: delaminated vs safe)
2. ROC Curve + AUC
3. Precision-Recall Curve
4. Predicted vs Actual scatter plot
5. Residual distribution
6. Per-class classification report
7. Regression metrics table

Usage:
    .venv\\Scripts\\python.exe scripts/generate_metrics.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    classification_report, accuracy_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from src.data.multimodal_loader import get_mega_loader
from src.models.integrated.framework import IntegratedDelaminationFramework

# ── Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'font.family': 'sans-serif',
    'font.size': 11,
})

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "visualizations" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'primary': '#58a6ff',
    'accent': '#f78166',
    'green': '#3fb950',
    'purple': '#bc8cff',
    'yellow': '#d29922',
    'red': '#f85149',
    'cyan': '#39d2c0',
}


def load_model_and_data():
    """Load the trained model and full dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = {
        'snpi_net': {'adaptive_kernel': {'input_dim': 6}},
        'cad_former': {'d_model': 128, 'n_layers': 4},
        'al_vtfd': {}
    }
    model = IntegratedDelaminationFramework(config).to(device)
    
    # Load best checkpoint
    ckpt_path = PROJECT_ROOT / "experiments" / "checkpoints" / "full_real" / "best_model.pt"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"  ✅ Loaded checkpoint from {ckpt_path}")
    else:
        # Fallback
        fallback = PROJECT_ROOT / "src" / "training" / "checkpoints" / "mega_run" / "mega_best_model.pt"
        if fallback.exists():
            model.load_state_dict(torch.load(fallback, map_location=device), strict=False)
            print(f"  ✅ Loaded fallback checkpoint")
        else:
            print("  ⚠️  No checkpoint found, using random weights")
    
    model.eval()
    
    # Load data
    data_root = Path("data/raw")
    loader = get_mega_loader(data_root, batch_size=256)
    
    return model, loader, device


@torch.no_grad()
def run_inference(model, loader, device, max_samples=5000):
    """Run inference on data and collect predictions + targets."""
    all_preds = []
    all_targets = []
    all_growth = []
    all_uncertainty = []
    all_migration = []
    n_collected = 0
    
    for batch in loader:
        if n_collected >= max_samples:
            break
            
        features = batch['features'].to(device)
        image = batch['image'].to(device) if 'image' in batch else None
        targets = batch['target']
        
        # Prepare inputs
        if features.shape[1] >= 262:
            laminate_config = features[:, 6:262].view(-1, 4, 64)
            loading_history = features[:, 262:362]
            physics_inputs = features[:, :6]
        else:
            physics_inputs = features[:, :6]
            laminate_config = torch.zeros(features.size(0), 4, 64, device=device)
            loading_history = torch.zeros(features.size(0), 100, device=device)
        
        outputs = model.predict_delamination(
            laminate_config, loading_history,
            physics_inputs=physics_inputs,
            meso_data=image
        )
        
        all_preds.append(outputs['delamination_area'].cpu().numpy().flatten())
        all_targets.append(targets.numpy().flatten())
        all_growth.append(outputs['growth_rate'].cpu().numpy().flatten())
        
        if 'uncertainty' in outputs:
            all_uncertainty.append(outputs['uncertainty'].cpu().numpy().flatten())
        if 'migration_interface' in outputs:
            mig = outputs['migration_interface'].cpu().numpy()
            if mig.ndim > 1:
                all_migration.append(mig.reshape(mig.shape[0], -1))
        
        n_collected += features.shape[0]
    
    results = {
        'preds': np.concatenate(all_preds),
        'targets': np.concatenate(all_targets),
        'growth': np.concatenate(all_growth),
    }
    if all_uncertainty:
        results['uncertainty'] = np.concatenate(all_uncertainty)
    if all_migration:
        results['migration'] = np.concatenate(all_migration, axis=0)
    
    print(f"  Collected {len(results['preds'])} predictions")
    return results


# ── PLOT 1: Confusion Matrix ─────────────────────────────────
def plot_confusion_matrix(targets, preds, threshold=0.05):
    """Binary classification: delaminated (>threshold) vs safe."""
    y_true = (targets > threshold).astype(int)
    y_pred = (preds > threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Raw counts
    disp1 = ConfusionMatrixDisplay(cm, display_labels=['Safe', 'Delaminated'])
    disp1.plot(ax=axes[0], cmap='Blues', values_format='d', colorbar=False)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold', color='white')
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Actual', fontsize=12)
    
    # Normalized
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=['Safe', 'Delaminated'])
    disp2.plot(ax=axes[1], cmap='RdYlGn_r', values_format='.3f', colorbar=False)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold', color='white')
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Actual', fontsize=12)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved {path.name}")
    
    return cm, y_true, y_pred


# ── PLOT 2: ROC Curve ────────────────────────────────────────
def plot_roc_curve(targets, preds, threshold=0.05):
    """ROC curve with AUC score."""
    y_true = (targets > threshold).astype(int)
    y_scores = preds  # Continuous prediction as probability score
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=COLORS['primary'], lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color=COLORS['red'], lw=1.5, linestyle='--', alpha=0.7, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS['primary'])
    
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve — Delamination Detection', fontsize=15, fontweight='bold', color='white')
    ax.legend(loc='lower right', fontsize=11, facecolor='#161b22', edgecolor='#30363d')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "roc_curve.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved {path.name} (AUC = {roc_auc:.4f})")
    
    return roc_auc


# ── PLOT 3: Precision-Recall Curve ───────────────────────────
def plot_precision_recall(targets, preds, threshold=0.05):
    """Precision-Recall curve with AP score."""
    y_true = (targets > threshold).astype(int)
    
    precision, recall, thresholds = precision_recall_curve(y_true, preds)
    ap = average_precision_score(y_true, preds)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color=COLORS['green'], lw=2.5, label=f'PR Curve (AP = {ap:.4f})')
    ax.fill_between(recall, precision, alpha=0.15, color=COLORS['green'])
    
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title('Precision-Recall Curve', fontsize=15, fontweight='bold', color='white')
    ax.legend(loc='lower left', fontsize=11, facecolor='#161b22', edgecolor='#30363d')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "precision_recall_curve.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved {path.name} (AP = {ap:.4f})")
    
    return ap


# ── PLOT 4: Predicted vs Actual Scatter ──────────────────────
def plot_pred_vs_actual(targets, preds):
    """Scatter plot of predicted vs actual delamination area."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    ax.scatter(targets, preds, alpha=0.3, s=8, c=COLORS['cyan'], edgecolors='none')
    
    # Perfect prediction line
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax.plot(lims, lims, '--', color=COLORS['accent'], lw=2, label='Perfect Prediction', alpha=0.8)
    
    # Regression line
    z = np.polyfit(targets, preds, 1)
    p = np.poly1d(z)
    x_line = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_line, p(x_line), '-', color=COLORS['primary'], lw=2, alpha=0.8,
            label=f'Fit: y={z[0]:.3f}x + {z[1]:.4f}')
    
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    
    ax.set_xlabel('Actual Delamination Area', fontsize=13)
    ax.set_ylabel('Predicted Delamination Area', fontsize=13)
    ax.set_title(f'Predicted vs Actual (R² = {r2:.4f}, RMSE = {rmse:.4f})',
                 fontsize=14, fontweight='bold', color='white')
    ax.legend(fontsize=10, facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "pred_vs_actual.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved {path.name}")


# ── PLOT 5: Residual Distribution ────────────────────────────
def plot_residuals(targets, preds):
    """Distribution of prediction residuals."""
    residuals = preds - targets
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Histogram
    axes[0].hist(residuals, bins=60, color=COLORS['purple'], alpha=0.8, edgecolor='#30363d')
    axes[0].axvline(0, color=COLORS['accent'], lw=2, linestyle='--')
    axes[0].axvline(np.mean(residuals), color=COLORS['yellow'], lw=2, linestyle='-',
                    label=f'Mean = {np.mean(residuals):.5f}')
    axes[0].set_xlabel('Residual (Pred - Actual)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Residual Distribution', fontsize=14, fontweight='bold', color='white')
    axes[0].legend(fontsize=10, facecolor='#161b22', edgecolor='#30363d')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q Plot style: Residuals vs Index (sorted)
    sorted_res = np.sort(residuals)
    axes[1].scatter(range(len(sorted_res)), sorted_res, s=2, c=COLORS['cyan'], alpha=0.5)
    axes[1].axhline(0, color=COLORS['accent'], lw=1.5, linestyle='--')
    axes[1].axhline(np.std(residuals), color=COLORS['yellow'], lw=1, linestyle=':', label=f'±1σ = {np.std(residuals):.4f}')
    axes[1].axhline(-np.std(residuals), color=COLORS['yellow'], lw=1, linestyle=':')
    axes[1].set_xlabel('Sample Index (sorted)', fontsize=12)
    axes[1].set_ylabel('Residual', fontsize=12)
    axes[1].set_title('Sorted Residuals', fontsize=14, fontweight='bold', color='white')
    axes[1].legend(fontsize=10, facecolor='#161b22', edgecolor='#30363d')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "residual_distribution.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved {path.name}")


# ── PLOT 6: Multi-threshold Confusion Matrices ───────────────
def plot_multi_threshold_cm(targets, preds):
    """Confusion matrices at different classification thresholds."""
    thresholds = [0.02, 0.05, 0.10, 0.20]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    
    for idx, t in enumerate(thresholds):
        y_true = (targets > t).astype(int)
        y_pred = (preds > t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
        
        disp = ConfusionMatrixDisplay(cm_norm, display_labels=['Safe', 'Delam.'])
        disp.plot(ax=axes[idx], cmap='Blues', values_format='.2f', colorbar=False)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        axes[idx].set_title(f't={t:.0%}\nAcc={acc:.1%} F1={f1:.3f}',
                           fontsize=11, fontweight='bold', color='white')
    
    fig.suptitle('Confusion Matrices at Multiple Thresholds', fontsize=15,
                 fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    path = OUTPUT_DIR / "multi_threshold_cm.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved {path.name}")


# ── PLOT 7: Master Dashboard ─────────────────────────────────
def plot_dashboard(targets, preds, growth, roc_auc, ap):
    """Combined dashboard with all key metrics."""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    threshold = 0.05
    y_true_cls = (targets > threshold).astype(int)
    y_pred_cls = (preds > threshold).astype(int)
    
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    acc = accuracy_score(y_true_cls, y_pred_cls)
    f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
    
    # ─ Panel 1: Metrics Summary ─
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    metrics_text = (
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  REGRESSION METRICS\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  RMSE:         {rmse:.4f}\n"
        f"  MAE:          {mae:.4f}\n"
        f"  R²:           {r2:+.4f}\n"
        f"  Samples:      {len(targets):,}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  CLASSIFICATION (t=5%)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  Accuracy:     {acc:.4f}\n"
        f"  F1 Score:     {f1:.4f}\n"
        f"  ROC AUC:      {roc_auc:.4f}\n"
        f"  Avg Precision: {ap:.4f}\n"
    )
    ax1.text(0.05, 0.95, metrics_text, fontsize=11, fontfamily='monospace',
             verticalalignment='top', color=COLORS['cyan'],
             bbox=dict(boxstyle='round', facecolor='#0d1117', edgecolor=COLORS['primary'], alpha=0.9))
    ax1.set_title('Key Metrics', fontsize=14, fontweight='bold', color='white')
    
    # ─ Panel 2: Pred vs Actual ─
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(targets, preds, alpha=0.25, s=5, c=COLORS['cyan'], edgecolors='none')
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax2.plot(lims, lims, '--', color=COLORS['accent'], lw=2)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title(f'Predicted vs Actual (R²={r2:.4f})', fontsize=12, fontweight='bold', color='white')
    ax2.grid(True, alpha=0.3)
    
    # ─ Panel 3: Confusion Matrix ─
    ax3 = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Safe', 'Delam.'])
    disp.plot(ax=ax3, cmap='Blues', values_format='d', colorbar=False)
    ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold', color='white')
    
    # ─ Panel 4: ROC Curve ─
    ax4 = fig.add_subplot(gs[1, 0])
    fpr, tpr, _ = roc_curve(y_true_cls, preds)
    ax4.plot(fpr, tpr, color=COLORS['primary'], lw=2.5)
    ax4.plot([0, 1], [0, 1], '--', color=COLORS['red'], lw=1.5, alpha=0.5)
    ax4.fill_between(fpr, tpr, alpha=0.1, color=COLORS['primary'])
    ax4.set_xlabel('FPR')
    ax4.set_ylabel('TPR')
    ax4.set_title(f'ROC Curve (AUC={roc_auc:.4f})', fontsize=12, fontweight='bold', color='white')
    ax4.grid(True, alpha=0.3)
    
    # ─ Panel 5: Residual Histogram ─
    ax5 = fig.add_subplot(gs[1, 1])
    residuals = preds - targets
    ax5.hist(residuals, bins=50, color=COLORS['purple'], alpha=0.8, edgecolor='#30363d')
    ax5.axvline(0, color=COLORS['accent'], lw=2, linestyle='--')
    ax5.set_xlabel('Residual')
    ax5.set_title(f'Residuals (μ={np.mean(residuals):.5f})', fontsize=12, fontweight='bold', color='white')
    ax5.grid(True, alpha=0.3)
    
    # ─ Panel 6: Precision-Recall ─
    ax6 = fig.add_subplot(gs[1, 2])
    precision, recall, _ = precision_recall_curve(y_true_cls, preds)
    ax6.plot(recall, precision, color=COLORS['green'], lw=2.5)
    ax6.fill_between(recall, precision, alpha=0.1, color=COLORS['green'])
    ax6.set_xlabel('Recall')
    ax6.set_ylabel('Precision')
    ax6.set_title(f'PR Curve (AP={ap:.4f})', fontsize=12, fontweight='bold', color='white')
    ax6.grid(True, alpha=0.3)
    
    fig.suptitle('DELAMINATION ML FRAMEWORK — Evaluation Dashboard',
                 fontsize=18, fontweight='bold', color='white', y=1.01)
    
    path = OUTPUT_DIR / "evaluation_dashboard.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved {path.name}")


# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("\n" + "#" * 60)
    print("# COMPREHENSIVE METRICS & VISUALIZATION GENERATOR")
    print("#" * 60)
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 60)
    
    print("\n[1/4] Loading model and data...")
    model, loader, device = load_model_and_data()
    
    print("\n[2/4] Running inference...")
    results = run_inference(model, loader, device)
    targets = results['targets']
    preds = results['preds']
    growth = results['growth']
    
    # Regression metrics
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    
    print(f"\n  Regression: RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:+.4f}")
    
    print("\n[3/4] Generating plots...")
    
    # Individual plots
    cm, y_true, y_pred = plot_confusion_matrix(targets, preds)
    roc_auc = plot_roc_curve(targets, preds)
    ap = plot_precision_recall(targets, preds)
    plot_pred_vs_actual(targets, preds)
    plot_residuals(targets, preds)
    plot_multi_threshold_cm(targets, preds)
    
    # Dashboard
    plot_dashboard(targets, preds, growth, roc_auc, ap)
    
    # Classification report
    threshold = 0.05
    y_true_cls = (targets > threshold).astype(int)
    y_pred_cls = (preds > threshold).astype(int)
    
    report = classification_report(y_true_cls, y_pred_cls,
                                   target_names=['Safe', 'Delaminated'],
                                   output_dict=True)
    report_str = classification_report(y_true_cls, y_pred_cls,
                                       target_names=['Safe', 'Delaminated'])
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT (threshold = 5%)")
    print('='*60)
    print(report_str)
    
    # Save all metrics to JSON
    print("[4/4] Saving metrics...")
    all_metrics = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': int(len(targets)),
        'regression': {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
        },
        'classification_5pct': {
            'accuracy': float(accuracy_score(y_true_cls, y_pred_cls)),
            'f1': float(f1_score(y_true_cls, y_pred_cls, zero_division=0)),
            'roc_auc': float(roc_auc),
            'avg_precision': float(ap),
            'confusion_matrix': cm.tolist(),
            'report': report
        },
        'residuals': {
            'mean': float(np.mean(preds - targets)),
            'std': float(np.std(preds - targets)),
            'max_abs': float(np.max(np.abs(preds - targets))),
        },
        'plots_saved': [
            str(p.relative_to(PROJECT_ROOT)) for p in OUTPUT_DIR.glob("*.png")
        ]
    }
    
    metrics_path = OUTPUT_DIR / "all_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("COMPLETE — All metrics and plots saved")
    print('='*60)
    print(f"  Output dir: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    for p in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"    📊 {p.name}")
    print(f"    📋 all_metrics.json")
    print('='*60)


if __name__ == "__main__":
    main()
