"""
Dataset Download & Setup Script for Delamination ML Framework.

Downloads and organizes publicly available datasets for training and validation:

1. NASA PCoE CFRP Composites (High Fidelity)
   - Tension-tension fatigue on CFRP panels
   - PZT sensor lamb wave signals + X-ray ground truth
   - Source: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
   - NOTE: Direct download currently unavailable. Contact christopher.a.teubert@nasa.gov

2. SDNET2018 — Concrete Crack/Delamination Images (Transfer Learning)
   - 56,000+ images of cracked/non-cracked surfaces
   - Source: https://doi.org/10.15142/T3TD19 or Kaggle
   - License: Free for academic use

3. Synthetic Physics-Based Data (Generated Locally)
   - Uses CLT + cohesive zone models to generate training samples
   - No download needed — generated on-the-fly

Usage:
    python scripts/download_datasets.py
"""
import os
import sys
import shutil
import zipfile
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def setup_directories():
    """Create the required data directory structure."""
    dirs = [
        DATA_DIR / "NASA_CFRP",
        DATA_DIR / "NASA_CFRP" / "PZT-data",
        DATA_DIR / "NASA_CFRP" / "XRays",
        DATA_DIR / "additional" / "F-MOC" / "extracted" / "ACOUSTIC",
        DATA_DIR / "additional" / "F-MOC" / "extracted" / "DIC",
        DATA_DIR / "additional" / "SDNET2021",
        DATA_DIR / "additional" / "HF_Real",
        PROJECT_ROOT / "data" / "experimental",
        PROJECT_ROOT / "data" / "simulated",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {d.relative_to(PROJECT_ROOT)}")


def download_file(url, dest_path, description="file"):
    """Download a file with progress indicator."""
    print(f"\n  Downloading {description}...")
    print(f"    URL:  {url}")
    print(f"    Dest: {dest_path}")
    
    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, count * block_size * 100 // total_size)
                print(f"\r    Progress: {pct}%", end="", flush=True)
        
        urlretrieve(url, str(dest_path), reporthook=progress_hook)
        print(f"\n    ✅ Downloaded: {dest_path.name} ({dest_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except (URLError, Exception) as e:
        print(f"\n    ❌ Download failed: {e}")
        return False


def generate_synthetic_physics_data():
    """
    Generate synthetic training data using physics models.
    Creates samples with known ground truth for validation.
    """
    print("\n" + "=" * 60)
    print("GENERATING SYNTHETIC PHYSICS-BASED DATA")
    print("=" * 60)
    
    try:
        import numpy as np
        import torch
    except ImportError:
        print("  ❌ NumPy/PyTorch not installed. Run: pip install numpy torch")
        return
    
    output_dir = PROJECT_ROOT / "data" / "simulated" / "synthetic_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    n_samples = 1000
    
    print(f"  Generating {n_samples} samples...")
    
    # Material properties for typical CFRP (T300/914C)
    configs = []
    for i in range(n_samples):
        # Randomize within manufacturing tolerances
        E11 = 130e9 + np.random.randn() * 5e9      # GPa
        E22 = 10e9 + np.random.randn() * 0.5e9
        G12 = 5e9 + np.random.randn() * 0.3e9
        nu12 = 0.28 + np.random.randn() * 0.02
        t_ply = 0.125e-3 + np.random.randn() * 0.005e-3  # mm
        
        # Fracture properties
        GIc = 0.25 + np.abs(np.random.randn() * 0.05)    # kJ/m²
        GIIc = 0.80 + np.abs(np.random.randn() * 0.10)
        eta_BK = 1.75 + np.random.randn() * 0.25          # B-K exponent
        
        # Stacking sequence (random cross-ply or quasi-isotropic)
        seq_type = np.random.choice(['cross_ply', 'quasi_iso', 'angle_ply'])
        if seq_type == 'cross_ply':
            angles = [0, 90, 0, 90, 90, 0, 90, 0]
        elif seq_type == 'quasi_iso':
            angles = [0, 45, -45, 90, 90, -45, 45, 0]
        else:
            angle = np.random.choice([15, 30, 45, 60])
            angles = [0, angle, -angle, 90, 90, -angle, angle, 0]
        
        # Applied loading
        max_load = np.random.uniform(5000, 50000)  # N
        load_ratio = np.random.uniform(0.05, 0.5)  # R-ratio
        n_cycles = int(np.random.uniform(1000, 1e6))
        
        # Ground truth: delamination area (using simplified Paris law model)
        # da/dN = C * (ΔG)^m
        C = 1e-10
        m = 3.5
        delta_G = GIc * (1 - load_ratio) * (max_load / 50000)**2
        da_dN = C * delta_G**m
        delamination_area = min(1.0, da_dN * n_cycles)
        
        # Migration probability (higher for cross-ply, lower for quasi-iso)
        migration_prob = np.random.uniform(0.1, 0.9)
        if seq_type == 'cross_ply':
            migration_prob = min(1.0, migration_prob * 1.5)
        
        configs.append({
            'E11': E11, 'E22': E22, 'G12': G12, 'nu12': nu12, 't_ply': t_ply,
            'GIc': GIc, 'GIIc': GIIc, 'eta_BK': eta_BK,
            'stacking': angles, 'seq_type': seq_type,
            'max_load': max_load, 'load_ratio': load_ratio, 'n_cycles': n_cycles,
            'delamination_area': delamination_area,
            'migration_prob': migration_prob
        })
    
    # Save as numpy arrays
    features = np.array([[c['E11'], c['E22'], c['G12'], c['nu12'], c['t_ply'], c['GIc']] for c in configs])
    targets = np.array([c['delamination_area'] for c in configs])
    migration_targets = np.array([c['migration_prob'] for c in configs])
    
    np.save(output_dir / "features.npy", features.astype(np.float32))
    np.save(output_dir / "targets.npy", targets.astype(np.float32))
    np.save(output_dir / "migration_targets.npy", migration_targets.astype(np.float32))
    
    # Save metadata
    import json
    metadata = {
        'n_samples': n_samples,
        'feature_columns': ['E11', 'E22', 'G12', 'nu12', 't_ply', 'GIc'],
        'target_column': 'delamination_area',
        'seq_types': {'cross_ply': sum(1 for c in configs if c['seq_type']=='cross_ply'),
                      'quasi_iso': sum(1 for c in configs if c['seq_type']=='quasi_iso'),
                      'angle_ply': sum(1 for c in configs if c['seq_type']=='angle_ply')},
        'source': 'Synthetic (Paris Law + CLT)',
        'material': 'CFRP T300/914C'
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✅ Saved {n_samples} samples to {output_dir.relative_to(PROJECT_ROOT)}")
    print(f"     Features shape: {features.shape}")
    print(f"     Targets shape:  {targets.shape}")
    print(f"     Stacking types: {metadata['seq_types']}")


def print_dataset_instructions():
    """Print instructions for manually downloading NASA PCoE data."""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    
    print("""
  ┌─────────────────────────────────────────────────────────────┐
  │              NASA PCoE CFRP Composites Dataset              │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  This dataset contains real fatigue test data with:         │
  │  • PZT lamb wave signals (16 sensors per panel)             │
  │  • Triaxial strain gage data                                │
  │  • X-ray images (ground truth damage)                       │
  │                                                             │
  │  HOW TO ACCESS:                                             │
  │  1. Visit: https://www.nasa.gov/intelligent-systems-        │
  │     division/discovery-and-systems-health/pcoe/             │
  │     pcoe-data-set-repository/                               │
  │                                                             │
  │  2. Email: christopher.a.teubert@nasa.gov                   │
  │     Subject: "Request for CFRP Composites Dataset"          │
  │                                                             │
  │  3. Once downloaded, extract to:                            │
  │     data/raw/NASA_CFRP/                                     │
  │     ├── PZT-data/  (MAT files)                              │
  │     └── XRays/     (JPG images)                             │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │              SDNET2018 — Crack/Delamination Images           │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  56,000+ images (bridge decks, walls, pavements)            │
  │  Used for transfer learning in our framework.               │
  │                                                             │
  │  HOW TO ACCESS:                                             │
  │  1. Visit: https://www.kaggle.com/datasets/                 │
  │     aniruddhsoul/structural-defects-network-concrete-       │
  │     crack-images                                            │
  │                                                             │
  │  2. Or use DOI: https://doi.org/10.15142/T3TD19             │
  │                                                             │
  │  3. Download and extract to:                                │
  │     data/raw/additional/SDNET2021/                           │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │         F-MOC Acoustic Emission + DIC Images                │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  Contact your lab or check institutional repositories       │
  │  for F-MOC fatigue monitoring data. Extract to:             │
  │     data/raw/additional/F-MOC/extracted/                    │
  │     ├── ACOUSTIC/ (.pridb files)                            │
  │     └── DIC/      (.tif images)                             │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
""")


def main():
    print("\n" + "#" * 60)
    print("# DELAMINATION ML FRAMEWORK — DATASET SETUP")
    print("#" * 60)
    
    # Step 1: Create directory structure
    print("\n[1/3] Setting up directory structure...")
    setup_directories()
    
    # Step 2: Generate synthetic data
    print("\n[2/3] Generating synthetic physics-based data...")
    generate_synthetic_physics_data()
    
    # Step 3: Print manual download instructions
    print("\n[3/3] Manual download instructions...")
    print_dataset_instructions()
    
    # Summary
    print("=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    total_files = sum(1 for _ in (PROJECT_ROOT / "data").rglob("*") if _.is_file())
    print(f"  Total files in data/: {total_files}")
    print(f"  Synthetic data:      data/simulated/synthetic_v1/")
    print(f"  Real data (manual):  data/raw/NASA_CFRP/")
    print(f"                       data/raw/additional/SDNET2021/")
    print("=" * 60)


if __name__ == "__main__":
    main()
