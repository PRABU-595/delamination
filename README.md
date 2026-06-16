# Delamination ML Framework

**Multiscale Machine Learning-Associated Nonlocal Modeling of Delamination in Composites**

A hybrid physics-informed deep learning framework with adaptive nonlocal influence and delamination migration tracking. This repository contains the reference implementation of the research framework described in the "Complete Research Document".

## 1. Project Overview

This framework addresses the critical challenge of predicting delamination migration and growth in composite materials by integrating three novel features:

- **Feature 1: SNPI-Net** (Stochastic Nonlocal Peridynamic-Informed Neural Network)
  - Adaptive nonlocal kernel learning for damage modeling.
  - Uncertainty quantification (Aleatoric & Epistemic).
  - Multi-fidelity Bayesian training.

- **Feature 2: CAD-Former** (Cross-Scale Attentive Delamination Transformer)
  - Hierarchical multi-scale feature extraction (Micro-Meso-Macro).
  - Spatial-temporal attention for delamination migration prediction.
  - Physics-embedded positional encoding.

- **Feature 3: AL-VTFD** (Active Learning-Guided Virtual Testing Framework)
  - Multi-objective acquisition function (Information Gain, Exploration, Exploitation, Cost).
  - Adaptive virtual testing strategy to minimize experimental cost.
  - Multi-fidelity Gaussian Process surrogate.

## 2. Repository Structure (Appendix D)

```
delamination-ml-project/
├── src/
│   ├── models/
│   │   ├── snpi_net/           # SNPI-Net implementation
│   │   ├── cad_former/         # CAD-Former implementation
│   │   └── al_vtfd/            #AL-VTFD implementation
│   ├── data/
│   │   ├── preprocessing.py    # Data preparation and cleaning
│   │   └── augmentation.py     # Physics-informed data augmentation
│   ├── training/
│   │   ├── train_integrated.py # Main training pipeline
│   │   ├── train_cad.py        # CAD-Former specific training
│   │   ├── train_snpi.py       # SNPI-Net specific training
│   │   └── active_learning.py  # AL-VTFD main loop
│   └── utils/
│       ├── visualization.py    # Plotting and analysis tools
│       └── metrics.py          # RMSE, R2, Migration Accuracy
├── experiments/                # Benchmark scripts and results
├── config/                     # Configuration YAML files
├── docs/                       # Documentation and guides
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 3. Installation

**Prerequisites:** Python 3.8+, PyTorch 1.10+, CUDA (optional but recommended for training).

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/delam-ml.git
cd delam-ml

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 4. Usage

### Training
To train the fully integrated framework:
```bash
python src/training/train_integrated.py --config config/model_config.yaml
```

To run the Active Learning loop (Feature 3):
```bash
python src/training/active_learning.py
```

### Inference & Benchmarking
To run the full benchmark suite and generate paper results (Section 9.1):
```bash
python experiments/run_all_benchmarks.py
```

To predict on a single custom image or sample:
```bash
python experiments/predict_custom_image.py --image_path path/to/image.jpg
```

## 5. Validation Strategy

The framework has been validated against experimental data (NASA/F-MOC) and literature benchmarks.
- **Accuracy Target**: RMSE < 0.08, R² > 0.92
- **Test Reduction**: ~70% fewer physical tests required via AL-VTFD.

## 6. Citation

If you use this code, please cite the associated research document:
> "Multiscale Machine Learning-Associated Nonlocal Modeling of Delamination in Composites", 2026.
