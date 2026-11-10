#!/usr/bin/env python3
"""
Phase 1 Empirical Validation:
1. High-SNR Dynamic Residual Bypass Gate: alpha(sigma)
2. Exposure-Calibrated Velocity Scaling: gamma_low(sigma)
3. Benchmark on all 40 SIDD paired scenes.
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.pc_ssflow import PCSSFlowV2
from src.utils.metrics import compute_image_metrics
from scripts.evaluate import load_sidd_paired_scenes
from src.utils.device import get_default_device

def estimate_sigma_mad(img: torch.Tensor) -> float:
    """Robust Donoho MAD noise standard deviation estimator."""
    lowpass = F.avg_pool2d(F.pad(img, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
    highpass = img - lowpass
    mad = torch.median(torch.abs(highpass - torch.median(highpass)))
    return (mad / 0.6745).item()

def main():
    dev = get_default_device("auto")
    print(f"Testing Phase 1 Improvements on {dev}...")

    base_m = PCSSFlowV2(in_channels=1, hidden_dim=128, n_scales=2, physics_mode="srgb").to(dev)
    base_m.load_state_dict(torch.load("checkpoints/pcssflow_v2_sidd_gpu.pt", map_location=dev))
    base_m.eval()

    reflow_m = PCSSFlowV2(in_channels=1, hidden_dim=128, n_scales=2, physics_mode="srgb").to(dev)
    reflow_m.load_state_dict(torch.load("checkpoints/pcssflow_v2_reflow.pt", map_location=dev))
    reflow_m.eval()

    val_pairs = load_sidd_paired_scenes("data/SIDD/train/train", max_scenes=40, crop_size=128)
    print(f"Loaded {len(val_pairs)} scenes.")

    # Sweep parameters for High-SNR bypass and low-SNR velocity boosting
    # Baseline: fixed gamma_low = 0.75, no bypass
    # Strategy 1: Dynamic Bypass Gate alone
    # Strategy 2: Exposure-Calibrated gamma_low alone
    # Strategy 3: Joint Bypass + Exposure-Calibrated gamma_low

    configs = [
        {"name": "Baseline Cascaded Reflow", "use_bypass": False, "adaptive_gamma": False},
        {"name": "Bypass Gate (sigma_floor=0.010, tau=0.005)", "use_bypass": True, "sigma_floor": 0.010, "tau": 0.005, "adaptive_gamma": False},
        {"name": "Adaptive Gamma (Low-SNR boost)", "use_bypass": False, "adaptive_gamma": True},
        {"name": "Joint Bypass + Adaptive Gamma", "use_bypass": True, "sigma_floor": 0.010, "tau": 0.005, "adaptive_gamma": True},
        {"name": "Joint Bypass + Adaptive Gamma (sigma_floor=0.008)", "use_bypass": True, "sigma_floor": 0.008, "tau": 0.004, "adaptive_gamma": True},
    ]

    for cfg in configs:
        psnr_l, ssim_l = [], []
        with torch.no_grad():
            for scene_name, noisy_t, gt_t in val_pairs:
                noisy = noisy_t.to(dev)
                clean = gt_t.to(dev)

                sigma = estimate_sigma_mad(noisy)

                if cfg["adaptive_gamma"]:
                    # Scale gamma_low with estimated noise: higher noise requires stronger lowpass cleanup
                    # sigma in [0.010, 0.080] -> g_low in [0.70, 1.80]
                    g_low = float(np.clip(0.65 + (sigma - 0.010) * 16.0, 0.65, 1.80))
                else:
                    g_low = 0.75

                gammas = {"low": g_low, "scales": [1.15, 0.85]}

                # Pass 1: 4-step Heun
                out1 = base_m.rectified_flow.denoise_sample(
                    noisy, num_steps=4, solver="heun", use_spectral_sizing=True, spectral_gammas=gammas
                )
                # Pass 2: 2-step Heun
                out2 = reflow_m.rectified_flow.denoise_sample(
                    out1, num_steps=2, solver="heun", use_spectral_sizing=True, spectral_gammas=gammas
                )
                out_flow = torch.clamp(out2, 0.0, 1.0)

                if cfg["use_bypass"]:
                    # Soft bypass gate: if sigma is very small, blend towards input y
                    s_floor = cfg["sigma_floor"]
                    tau = cfg["tau"]
                    alpha = 1.0 / (1.0 + np.exp(-(sigma - s_floor) / tau))
                    out = alpha * out_flow + (1.0 - alpha) * noisy
                else:
                    out = out_flow

                out = torch.clamp(out, 0.0, 1.0)
                m = compute_image_metrics(out, clean)
                psnr_l.append(m["psnr"])
                ssim_l.append(m["ssim"])

        print(f"{cfg['name']:<55s} | PSNR: {np.mean(psnr_l):6.3f} dB | SSIM: {np.mean(ssim_l):6.4f}")

if __name__ == "__main__":
    main()
