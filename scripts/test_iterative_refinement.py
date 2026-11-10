#!/usr/bin/env python3
"""
Systematic Investigation of Iterative Phase-Domain Refinement:
Analyzes multi-pass self-conditioned flow on 40 SIDD scenes.
Explores:
1. Pass 2 solver (Euler vs Heun) and step count (1, 2, 4)
2. Residual blending factor alpha
3. Pass 3 convergence / over-smoothing limit
4. Combination with 8-fold TTA
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.pc_ssflow import PCSSFlowV2
from src.utils.metrics import compute_image_metrics
from scripts.evaluate import load_sidd_paired_scenes
from src.utils.device import get_default_device

def main():
    dev = get_default_device("auto")
    print(f"Loading base PC-SSFlow model on {dev}...")
    model = PCSSFlowV2(in_channels=1, hidden_dim=128, n_scales=2, physics_mode="srgb").to(dev)
    model.load_state_dict(torch.load("checkpoints/pcssflow_v2_sidd_gpu.pt", map_location=dev))
    model.eval()

    val_pairs = load_sidd_paired_scenes("data/SIDD/train/train", max_scenes=40, crop_size=128)
    print(f"Loaded {len(val_pairs)} SIDD validation scenes.")

    configs = [
        {"name": "Pass 1 (Baseline)", "p2_steps": 0, "p2_solver": "none", "alpha": 1.0},
        {"name": "Pass 2 (1-step Euler, alpha=1.0)", "p2_steps": 1, "p2_solver": "euler", "alpha": 1.0},
        {"name": "Pass 2 (2-step Euler, alpha=1.0)", "p2_steps": 2, "p2_solver": "euler", "alpha": 1.0},
        {"name": "Pass 2 (1-step Heun,  alpha=1.0)", "p2_steps": 1, "p2_solver": "heun",  "alpha": 1.0},
        {"name": "Pass 2 (2-step Heun,  alpha=1.0)", "p2_steps": 2, "p2_solver": "heun",  "alpha": 1.0},
        {"name": "Pass 2 (1-step Euler, alpha=0.85)", "p2_steps": 1, "p2_solver": "euler", "alpha": 0.85},
        {"name": "Pass 2 (2-step Euler, alpha=0.85)", "p2_steps": 2, "p2_solver": "euler", "alpha": 0.85},
    ]

    for cfg in configs:
        psnr_l, ssim_l = [], []
        with torch.no_grad():
            for _, noisy_t, gt_t in val_pairs:
                noisy = noisy_t.to(dev)
                clean = gt_t.to(dev)

                out1 = model.rectified_flow.denoise_sample(
                    y_noisy=noisy,
                    num_steps=4,
                    solver="heun",
                    use_spectral_sizing=True
                )

                if cfg["p2_steps"] == 0:
                    out = out1
                else:
                    out2 = model.rectified_flow.denoise_sample(
                        y_noisy=out1,
                        num_steps=cfg["p2_steps"],
                        solver=cfg["p2_solver"],
                        use_spectral_sizing=True
                    )
                    alpha = cfg["alpha"]
                    out = alpha * out2 + (1.0 - alpha) * out1

                out = torch.clamp(out, 0.0, 1.0)
                m = compute_image_metrics(out, clean)
                psnr_l.append(m["psnr"])
                ssim_l.append(m["ssim"])

        print(f"{cfg['name']:<35s} | PSNR: {np.mean(psnr_l):6.3f} dB | SSIM: {np.mean(ssim_l):6.4f}")

    print("\n--- TESTING 3-PASS & TTA COMBINATIONS ---")
    for pass3_steps in [0, 1]:
        for use_tta in [False, True]:
            psnr_l, ssim_l = [], []
            with torch.no_grad():
                for _, noisy_t, gt_t in val_pairs:
                    noisy = noisy_t.to(dev)
                    clean = gt_t.to(dev)

                    if use_tta:
                        outs = []
                        for k in [0, 1, 2, 3]:
                            rx = torch.rot90(noisy, k, [-2, -1])
                            orot = model.rectified_flow.denoise_sample(rx, num_steps=4, solver="heun", use_spectral_sizing=True)
                            outs.append(torch.rot90(orot, -k, [-2, -1]))
                            fx = torch.flip(rx, dims=[-1])
                            oflip = model.rectified_flow.denoise_sample(fx, num_steps=4, solver="heun", use_spectral_sizing=True)
                            outs.append(torch.rot90(torch.flip(oflip, dims=[-1]), -k, [-2, -1]))
                        out1 = torch.stack(outs, dim=0).mean(dim=0)
                    else:
                        out1 = model.rectified_flow.denoise_sample(noisy, num_steps=4, solver="heun", use_spectral_sizing=True)

                    out2 = model.rectified_flow.denoise_sample(out1, num_steps=2, solver="euler", use_spectral_sizing=True)

                    if pass3_steps > 0:
                        out3 = model.rectified_flow.denoise_sample(out2, num_steps=pass3_steps, solver="euler", use_spectral_sizing=True)
                        out = out3
                    else:
                        out = out2

                    out = torch.clamp(out, 0.0, 1.0)
                    m = compute_image_metrics(out, clean)
                    psnr_l.append(m["psnr"])
                    ssim_l.append(m["ssim"])

            tag = f"Pass 2 (2-step Euler) + Pass 3 ({pass3_steps}-step)" + (" + 8-fold TTA" if use_tta else "")
            print(f"{tag:<50s} | PSNR: {np.mean(psnr_l):6.3f} dB | SSIM: {np.mean(ssim_l):6.4f}")

if __name__ == "__main__":
    main()
