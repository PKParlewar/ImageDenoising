#!/usr/bin/env python3
"""
Diagnostic evaluation script to compare:
1. Hybrid (with SSM, w=8): 2-step Euler, 4-step Heun
2. Pruned PC-Flow (without SSM): 2-step Euler, 4-step Heun, 4-step Heun + TTA
Across 40 SIDD benchmark paired scenes, plus latency measurement.
"""

import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.pc_ssflow import PCSSFlowV2
from src.utils.metrics import compute_image_metrics
from scripts.evaluate import load_sidd_paired_scenes
from src.utils.device import get_default_device

def main():
    dev = get_default_device("auto")
    print(f"Device: {dev}")
    
    ckpt_path = "checkpoints/pcssflow_v2_sidd_gpu.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/pcssflow_v2_sidd.pt"
    
    print(f"Loading checkpoint: {ckpt_path}")
    model = PCSSFlowV2(in_channels=1, hidden_dim=128, n_scales=2, physics_mode="srgb").to(dev)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()

    pairs = load_sidd_paired_scenes("data/SIDD/train/train", max_scenes=40, crop_size=128)
    print(f"Loaded {len(pairs)} SIDD scenes.")

    # Helper function to run inference with or without SSM
    def run_eval(use_ssm: bool, solver: str, steps: int, use_tta: bool = False):
        psnr_list, ssim_list = [], []
        
        for name, noisy_t, gt_t in pairs:
            noisy = noisy_t.to(dev)
            clean = gt_t.to(dev)
            
            with torch.no_grad():
                if not use_tta:
                    # Single channel
                    pc = model.pc_calc(noisy)
                    if use_ssm:
                        ssm_feat = model.ssm_backbone(noisy, pc, window_size=model.warping_window)
                    else:
                        ssm_feat = pc
                    denoised = model.rectified_flow.denoise_sample(
                        y_noisy=noisy,
                        num_steps=steps,
                        ssm_feat=ssm_feat,
                        solver=solver
                    )
                else:
                    # TTA with or without SSM
                    outs = []
                    for k in [0, 1, 2, 3]:
                        rot_x = torch.rot90(noisy, k, [-2, -1])
                        pc_rot = model.pc_calc(rot_x)
                        if use_ssm:
                            ssm_rot = model.ssm_backbone(rot_x, pc_rot, window_size=model.warping_window)
                        else:
                            ssm_rot = pc_rot
                        out_rot = model.rectified_flow.denoise_sample(
                            y_noisy=rot_x,
                            num_steps=steps,
                            ssm_feat=ssm_rot,
                            solver=solver
                        )
                        outs.append(torch.rot90(out_rot, -k, [-2, -1]))
                        
                        flip_x = torch.flip(rot_x, dims=[-1])
                        pc_flip = model.pc_calc(flip_x)
                        if use_ssm:
                            ssm_flip = model.ssm_backbone(flip_x, pc_flip, window_size=model.warping_window)
                        else:
                            ssm_flip = pc_flip
                        out_flip = model.rectified_flow.denoise_sample(
                            y_noisy=flip_x,
                            num_steps=steps,
                            ssm_feat=ssm_flip,
                            solver=solver
                        )
                        outs.append(torch.rot90(torch.flip(out_flip, dims=[-1]), -k, [-2, -1]))
                    denoised = torch.stack(outs, dim=0).mean(dim=0)
                
                denoised = torch.clamp(denoised, 0.0, 1.0)
                m = compute_image_metrics(denoised, clean)
                psnr_list.append(m["psnr"])
                ssim_list.append(m["ssim"])
                
        mean_psnr = float(np.mean(psnr_list))
        mean_ssim = float(np.mean(ssim_list))
        return mean_psnr, mean_ssim

    # Latency benchmarking
    dummy_input = torch.randn(1, 1, 128, 128, device=dev)
    def measure_latency(use_ssm: bool, solver: str, steps: int, runs: int = 50):
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                pc = model.pc_calc(dummy_input)
                ssm_feat = model.ssm_backbone(dummy_input, pc, window_size=model.warping_window) if use_ssm else pc
                _ = model.rectified_flow.denoise_sample(dummy_input, num_steps=steps, ssm_feat=ssm_feat, solver=solver)
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize()
            
        t0 = time.perf_counter()
        for _ in range(runs):
            with torch.no_grad():
                pc = model.pc_calc(dummy_input)
                ssm_feat = model.ssm_backbone(dummy_input, pc, window_size=model.warping_window) if use_ssm else pc
                _ = model.rectified_flow.denoise_sample(dummy_input, num_steps=steps, ssm_feat=ssm_feat, solver=solver)
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) / runs * 1000.0  # ms

    configs = [
        ("Hybrid (with SSM)", True, "euler", 2, False),
        ("Hybrid (with SSM)", True, "heun", 4, False),
        ("Hybrid (with SSM)", True, "heun", 4, True),
        ("Pruned (without SSM)", False, "euler", 2, False),
        ("Pruned (without SSM)", False, "heun", 4, False),
        ("Pruned (without SSM)", False, "heun", 4, True),
    ]

    print("\n--- RESULTS ON 40 SIDD SCENES ---")
    for name, use_ssm, solver, steps, use_tta in configs:
        psnr, ssim = run_eval(use_ssm, solver, steps, use_tta)
        if not use_tta:
            lat = measure_latency(use_ssm, solver, steps)
            lat_str = f"{lat:5.2f} ms"
        else:
            lat = measure_latency(use_ssm, solver, steps) * 8
            lat_str = f"{lat:5.2f} ms (8x TTA)"
        print(f"{name:22s} | Solver: {solver:5s} | Steps: {steps:2d} | TTA: {str(use_tta):5s} | PSNR: {psnr:6.3f} dB | SSIM: {ssim:6.4f} | Latency: {lat_str}")

if __name__ == "__main__":
    main()
