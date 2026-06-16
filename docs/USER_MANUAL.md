# Delamination Framework - User Manual

## Overview
The Delamination ML Framework is a hybrid physics-informed deep learning system for predicting interlaminar damage in composite structures.

## Core Modules
1. **SNPI-Net**: Stochastic Nonlocal Peridynamic-Informed Neural Network. Handles nonlocal interactions and uncertainty.
2. **CAD-Former**: Cross-Scale Attentive Delamination Transformer. Tracks delamination migration across interfaces.
3. **AL-VTFD**: Active Learning-Guided Virtual Testing. Minimizes experimental requirements.

## Usage
### 1. Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Inference
Run the interactive prediction script to explore the championship model:
```bash
python predict.py
```

### 3. Training
To start a new mega-run on local data:
```bash
python src/training/train_mega.py
```

### 4. Visualization
Generate publication-quality migration plots:
```bash
python experiments/visualization_paper.py
```
