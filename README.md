# Phase-Congruency Guided Image Denoising

Official PyTorch implementation for **Phase-Congruency Guided Image Denoising**, featuring **PG-NAFNet** (Phase-Gated Nonlinear Activation-Free Network) and **PC-SSFlow v2** (Phase-Congruency Structure-Coupled Rectified Flow) across sRGB and RAW sensor domains.

---

## Overview

This repository provides a modular research codebase for phase-aware image restoration:

- **Dual-Tree Complex Wavelet Transform (DTCWT) & Fourier Phase Congruency (`src/wavelets/`)**: Extracts contrast- and illumination-invariant structural phase congruency maps ($PC \in [0, 1]$) with automatic Rayleigh noise floor compensation.
- **PG-NAFNet (`src/models/pg_nafnet.py`)**: A lightweight, single-channel input (`in_ch=1`) Nonlinear Activation-Free Network equipped with zero-initialized **Phase-Gated SimpleGates (`PG-Gate`)** that modulate feature transmission across depth using structural phase congruency with minimal parameter overhead (+384 parameters across 12 blocks).
- **PC-SSFlow v2 (`src/models/pc_ssflow.py`)**: Combines noise-correlation-adaptive spectral masking (`src/masking/`), structure-guided Hilbert curve warping (`src/backbone/`), and physics-calibrated rectified flow transport (`src/flow/`).

---

## Repository Structure

```text
├── configs/                        # Experiment and ablation YAML configurations
│   ├── default_srgb.yaml           # Default sRGB denoising configuration
│   ├── default_raw.yaml            # Default RAW Bayer sensor configuration
│   └── ablation_*.yaml             # Component-level ablation configurations
├── src/                            # Core library implementation
│   ├── backbone/                   # Hilbert curve indexing, PC warping, and 2D SSM blocks
│   ├── flow/                       # Rectified flow ODE solver, physics schedules, proximal step
│   ├── masking/                    # MAD noise floor estimator and adaptive spectral masking
│   ├── models/                     # PG-NAFNet, PC-SSFlow v2, and self-supervised losses
│   ├── utils/                      # Device management, PSNR/SSIM metrics, and Bayer RAW packing
│   └── wavelets/                   # 2D DTCWT decomposition and Kovesi Phase Congruency
├── scripts/                        # Training, evaluation, and ablation scripts
│   ├── train_pg_nafnet.py          # Train PG-NAFNet vs. NAFNet baseline
│   ├── eval_pg_nafnet.py           # Evaluate PG-NAFNet on SIDD / benchmark sets
│   ├── train_phase_nafnet_raw.py   # Train & evaluate on 4-channel RAW sensor patches
│   ├── train.py                    # Train PC-SSFlow v2 (sRGB)
│   ├── train_raw.py                # Train PC-SSFlow v2 (RAW sensor domain)
│   ├── evaluate.py                 # Standard benchmark evaluation (SIDD, BSD68, Urban100)
│   ├── demo_denoise.py             # Single-image visual denoising demo
│   └── ...                         # Ablation sweeps, cross-dataset & hardware profiling scripts
├── tests/                          # Unit and integration test suite (pytest)
├── download_eval_datasets.py       # Automated downloader for BSD68, Urban100, and SIDD validation
├── download_dnd.py                 # Automated downloader for the DND benchmark dataset
├── pytest.ini                      # Pytest configuration
└── requirements.txt                # Python package dependencies
```

---

## Installation

1. **Create and activate a virtual environment** (Python 3.10+ recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   *Note: Hardware acceleration is automatically detected across NVIDIA CUDA (`cuda`), Apple Silicon Metal (`mps`), and CPU via `src/utils/device.py`.*

---

## Dataset Preparation

Download the standard evaluation benchmarks (BSD68, Urban100, and SIDD validation blocks) into `data/`:

```bash
python download_eval_datasets.py
```

To download the Darmstadt Noise Dataset (DND) using your [DND portal](https://noise.visinf.tu-darmstadt.de/) credentials:

```bash
python download_dnd.py --username <DND_USER> --password <DND_PASSWORD>
```

---

## Usage

### 1. Training & Evaluating PG-NAFNet
Train the flagship `PG-NAFNet` architecture and evaluate on the SIDD validation benchmark:

```bash
python scripts/train_pg_nafnet.py
python scripts/eval_pg_nafnet.py
```

For RAW 4-channel Bayer (`R-G1-G2-B`) training and evaluation under realistic Poisson-Gaussian read/shot noise:

```bash
python scripts/train_and_eval_phase_nafnet_raw.py
```

### 2. Training & Evaluating PC-SSFlow v2
Train `PC-SSFlow v2` using a YAML configuration file:

```bash
# sRGB domain
python scripts/train.py --config configs/default_srgb.yaml

# RAW sensor domain
python scripts/train_raw.py --config configs/default_raw.yaml
```

Evaluate a trained checkpoint on BSD68, Urban100, or SIDD:

```bash
python scripts/evaluate.py \
    --config configs/default_srgb.yaml \
    --dataset bsd68 \
    --data_dir data/BSD68 \
    --checkpoint checkpoints/pcssflow_v2.pt
```

### 3. Quick Demo
Run single-image denoising and save a side-by-side comparison (`Noisy | Phase Congruency | Denoised | Residual`):

```bash
python scripts/demo_denoise.py \
    --image data/BSD68/test001.png \
    --output results/demo_comparison.png \
    --sigma 0.08
```

### 4. Reproducing Ablation Studies & Profiling
- **Component & Architecture Ablations**:
  ```bash
  python scripts/run_ablations.py
  python scripts/eval_scaled_ablations.py
  ```
- **Correlated Spatial Noise & Depth Pareto Sweeps**:
  ```bash
  python scripts/benchmark_correlated_noise.py
  python scripts/benchmark_depth_pareto.py
  ```
- **Hardware Latency, Throughput & Memory Profiling**:
  ```bash
  python scripts/profile_hardware.py
  ```

---

## Running Tests

Run the unit test suite via `pytest` to verify the DTCWT filterbanks, phase congruency invariance, adaptive spectral masking, Hilbert curve warping, rectified flow solvers, and `PG-NAFNet` zero-initialization identity:

```bash
pytest
```
