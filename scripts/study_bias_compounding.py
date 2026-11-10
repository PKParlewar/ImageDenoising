#!/usr/bin/env python3
"""
Task 5.2: Pseudo-Clean Target Bias Compounding Study.

Compares:
1. Model A: Rectified Flow trained with Ground-Truth Clean pairs (y, x_clean) [Upper Bound].
2. Model B: Rectified Flow trained with Self-Supervised Pseudo-Clean pairs (y, x_pseudo) [Bootstrap Method].

Quantifies residual bias E[x_pseudo - x_clean], PSNR / SSIM gap, and error compounding metrics.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.pc_ssflow import PCSSFlowV2
from src.models.loss import PCSSFlowLoss
from src.utils.metrics import compute_image_metrics
from src.utils.device import get_default_device


def run_bias_compounding_study(
    num_train_iters: int = 40,
    device: str = "auto",
    output_json: str = "results/bias_compounding_study.json"
):
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    dev = get_default_device(device)

    print("==========================================================================")
    print(f"Executing Task 5.2: Pseudo-Clean Target Bias Compounding Study (Device: {dev})")
    print("==========================================================================")

    # 1. Create synthetic / benchmark ground-truth test pairs
    # Use real image if available
    img_path = "data/BSD68/test001.png"
    if os.path.exists(img_path):
        img_raw = Image.open(img_path).convert("L").resize((64, 64))
        clean_img = torch.from_numpy(np.array(img_raw, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
    else:
        clean_img = torch.rand(1, 1, 64, 64, device=dev)

    # Correlated noise generator
    noise_sigma = 0.08
    noisy_img = torch.clamp(clean_img + torch.randn_like(clean_img) * noise_sigma, 0.0, 1.0)

    # 2. Stage 1: Generate Self-Supervised Pseudo-Clean Target
    print("\n[1/3] Training Self-Supervised Branch to generate pseudo-clean target...")
    model_self = PCSSFlowV2(in_channels=1, hidden_dim=32, n_scales=2, physics_mode="srgb").to(dev)
    loss_fn = PCSSFlowLoss()
    optimizer_self = torch.optim.Adam(
        list(model_self.mag_denoiser.parameters()) + list(model_self.ssm_backbone.parameters()),
        lr=0.005
    )

    for it in range(num_train_iters):
        optimizer_self.zero_grad()
        out = model_self.forward_self_supervised_step(noisy_img)
        l_dict = loss_fn.self_supervised_loss(
            rec_mags=out["rec_mags"],
            orig_mags=out["orig_mags"],
            masks=out["masks"],
            pc_pred=out["pc_pred"],
            pc_orig=out["pc_orig"],
            denoised_img=out["pseudo_clean"],
            y_noisy=noisy_img
        )
        l_dict["loss_total"].backward()
        optimizer_self.step()

    with torch.no_grad():
        pseudo_clean = model_self.forward_self_supervised_step(noisy_img)["pseudo_clean"].detach()
        pseudo_clean = torch.clamp(pseudo_clean, 0.0, 1.0)

    # Measure initial pseudo-target bias
    bias_map = pseudo_clean - clean_img
    mean_bias = bias_map.mean().item()
    std_bias = bias_map.std().item()
    pseudo_metrics = compute_image_metrics(pseudo_clean, clean_img)

    print(f"Self-Supervised Pseudo-Clean Initial PSNR: {pseudo_metrics['psnr']:.2f} dB, SSIM: {pseudo_metrics['ssim']:.4f}")
    print(f"Pseudo Target Bias: Mean = {mean_bias:+.6f}, Std = {std_bias:.6f}")

    # 3. Model A: Train Flow with Ground Truth Clean Pairs (Upper Bound)
    print("\n[2/3] Training Model A: Rectified Flow on Ground-Truth Pairs (Upper Bound)...")
    flow_gt = PCSSFlowV2(in_channels=1, hidden_dim=32, n_scales=2, physics_mode="srgb").to(dev)
    opt_gt = torch.optim.Adam(flow_gt.rectified_flow.velocity_net.parameters(), lr=0.005)

    for it in range(num_train_iters):
        opt_gt.zero_grad()
        l_flow = flow_gt.rectified_flow.compute_flow_loss(noisy_img, clean_img)["loss"]
        l_flow.backward()
        opt_gt.step()

    with torch.no_grad():
        denoised_gt = flow_gt.denoise(noisy_img, num_flow_steps=2)
        metrics_gt = compute_image_metrics(denoised_gt, clean_img)

    # 4. Model B: Train Flow with Self-Supervised Pseudo-Clean Pairs
    print("\n[3/3] Training Model B: Rectified Flow on Pseudo-Clean Pairs (Bootstrap)...")
    flow_pseudo = PCSSFlowV2(in_channels=1, hidden_dim=32, n_scales=2, physics_mode="srgb").to(dev)
    opt_pseudo = torch.optim.Adam(flow_pseudo.rectified_flow.velocity_net.parameters(), lr=0.005)

    for it in range(num_train_iters):
        opt_pseudo.zero_grad()
        l_flow = flow_pseudo.rectified_flow.compute_flow_loss(noisy_img, pseudo_clean)["loss"]
        l_flow.backward()
        opt_pseudo.step()

    with torch.no_grad():
        denoised_pseudo = flow_pseudo.denoise(noisy_img, num_flow_steps=2)
        metrics_pseudo = compute_image_metrics(denoised_pseudo, clean_img)

    # 5. Summary and Comparison
    psnr_gap = metrics_gt["psnr"] - metrics_pseudo["psnr"]
    ssim_gap = metrics_gt["ssim"] - metrics_pseudo["ssim"]

    summary = {
        "pseudo_target_bias": {
            "mean_bias": mean_bias,
            "std_bias": std_bias,
            "psnr": pseudo_metrics["psnr"],
            "ssim": pseudo_metrics["ssim"]
        },
        "model_A_ground_truth_flow": metrics_gt,
        "model_B_pseudo_bootstrap_flow": metrics_pseudo,
        "psnr_gap_db": psnr_gap,
        "ssim_gap": ssim_gap
    }

    print("\n==========================================================================")
    print("BIAS COMPOUNDING STUDY RESULTS")
    print("--------------------------------------------------------------------------")
    print(f"Model A (Ground Truth Flow):      PSNR = {metrics_gt['psnr']:6.2f} dB, SSIM = {metrics_gt['ssim']:.4f}")
    print(f"Model B (Pseudo Bootstrap Flow):  PSNR = {metrics_pseudo['psnr']:6.2f} dB, SSIM = {metrics_pseudo['ssim']:.4f}")
    print(f"Performance Gap:                  PSNR Gap = {psnr_gap:+.2f} dB, SSIM Gap = {ssim_gap:+.4f}")
    print(f"Residual Target Bias:             Mean Bias = {mean_bias:+.6f}")
    print("==========================================================================")
    print(f"[✓] Bias study report saved to {output_json}")

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo-Clean Target Bias Compounding Study")
    parser.add_argument("--num-train-iters", type=int, default=40)
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, mps, cuda, cpu)")
    parser.add_argument("--output-json", type=str, default="results/bias_compounding_study.json")
    args = parser.parse_args()

    run_bias_compounding_study(
        num_train_iters=args.num_train_iters,
        device=args.device,
        output_json=args.output_json
    )
