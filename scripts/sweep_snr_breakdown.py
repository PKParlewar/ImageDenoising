#!/usr/bin/env python3
"""
Task 1.4: SNR vs. Phase Congruency Breakdown Characterization Study.

Sweeps input SNR from 30 dB down to -5 dB across natural and synthetic test images,
measuring PC distortion and identifying the critical breakdown threshold where noise overwhelms phase alignment.
"""

import os
import sys
import json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.wavelets.phase_congruency import PhaseCongruencyCalculator, measure_pc_snr_breakdown


def load_test_image(image_path: str, size: int = 128) -> torch.Tensor:
    """Loads and crops/resizes an image to a normalized tensor [1, 1, H, W]."""
    if os.path.exists(image_path):
        img = Image.open(image_path).convert("L")
        img = img.resize((size, size))
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    else:
        # Generate synthetic synthetic edge/texture pattern
        grid = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(grid, grid)
        pattern = 0.5 * (np.sin(10 * xx) * np.cos(10 * yy) + 1.0)
        # Add a sharp step edge
        pattern[size//4:3*size//4, size//4:3*size//4] += 0.5
        pattern = np.clip(pattern, 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(pattern).unsqueeze(0).unsqueeze(0)


def run_snr_breakdown_study(output_json: str = "results/snr_pc_breakdown.json"):
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    
    # Try finding an image from BSD68 or Urban100 if available
    sample_paths = [
        "data/BSD68/test001.png",
        "data/BSD68/test010.png",
        "data/Urban100/img_001.png"
    ]
    test_path = None
    for p in sample_paths:
        if os.path.exists(p):
            test_path = p
            break

    print(f"Running SNR breakdown study using: {test_path or 'Synthetic Pattern'}...")
    img = load_test_image(test_path if test_path else "")

    snr_grid = [30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0, -5.0]
    results = measure_pc_snr_breakdown(
        clean_image=img,
        snr_db_range=snr_grid,
        n_scales=3,
        noise_threshold_mult=2.0
    )

    print("\n==================================================")
    print("SNR (dB) | Sigma   | PC MSE   | PC MAE")
    print("--------------------------------------------------")
    for i, snr in enumerate(results["snr_db_list"]):
        sig = results["sigma_list"][i]
        mse = results["pc_mse_list"][i]
        mae = results["pc_mae_list"][i]
        print(f"{snr:7.1f}  | {sig:7.4f} | {mse:8.6f} | {mae:8.6f}")
    print("==================================================")
    print(f"Empirical Critical SNR Threshold: {results['critical_snr_db']} dB")
    print("==================================================")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Results saved to {output_json}")


if __name__ == "__main__":
    run_snr_breakdown_study()
