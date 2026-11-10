#!/usr/bin/env python3
"""
PC-SSFlow v2 Interactive Visual Denoising Demo.

Loads an input image, applies noise, runs PC-SSFlow v2 denoising,
and saves side-by-side visual comparisons (Noisy, Phase Congruency Map, Denoised, Difference Residual).
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.pc_ssflow import PCSSFlowV2
from src.utils.metrics import compute_image_metrics
from src.utils.device import get_default_device


def run_demo(
    image_path: str = "data/BSD68/test001.png",
    output_path: str = "results/demo_comparison.png",
    noise_sigma: float = 0.08,
    input_mode: str = "srgb",
    checkpoint: str = "checkpoints/pcssflow_v2.pt",
    device: str = "auto"
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    dev = get_default_device(device)

    print(f"Loading input image: {image_path} (Device: {dev})...")
    if os.path.exists(image_path):
        img_raw = Image.open(image_path).convert("L").resize((128, 128))
        clean = torch.from_numpy(np.array(img_raw, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
    else:
        clean = torch.rand(1, 1, 128, 128, device=dev)

    noisy = torch.clamp(clean + torch.randn_like(clean) * noise_sigma, 0.0, 1.0)

    h_dim = 128
    if checkpoint and os.path.exists(checkpoint):
        try:
            ckpt = torch.load(checkpoint, map_location="cpu")
            if "rectified_flow.velocity_net.time_mlp.0.weight" in ckpt:
                h_dim = ckpt["rectified_flow.velocity_net.time_mlp.0.weight"].shape[0]
            elif "ssm_backbone.out_proj.weight" in ckpt:
                h_dim = ckpt["ssm_backbone.out_proj.weight"].shape[0]
        except Exception:
            pass

    model = PCSSFlowV2(
        in_channels=1,
        hidden_dim=h_dim,
        n_scales=2,
        physics_mode=input_mode
    ).to(dev)

    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=dev))
        print(f"Loaded checkpoint from {checkpoint}")

    model.eval()
    with torch.no_grad():
        pc_map = model.pc_calc(noisy)
        denoised = model.denoise(noisy, num_flow_steps=2)
        denoised = torch.clamp(denoised, 0.0, 1.0)

    m_noisy = compute_image_metrics(noisy, clean)
    m_denoised = compute_image_metrics(denoised, clean)

    print("\n==================================================")
    print(f"Noisy Input PSNR:    {m_noisy['psnr']:.2f} dB, SSIM: {m_noisy['ssim']:.4f}")
    print(f"Denoised Output PSNR:{m_denoised['psnr']:.2f} dB, SSIM: {m_denoised['ssim']:.4f}")
    print("==================================================")

    # Save side-by-side grid
    # [Clean, Noisy, PC Map, Denoised]
    clean_np = (clean[0, 0].cpu().numpy() * 255.0).astype(np.uint8)
    noisy_np = (noisy[0, 0].cpu().numpy() * 255.0).astype(np.uint8)
    pc_np = (pc_map[0, 0].cpu().numpy() * 255.0).astype(np.uint8)
    denoised_np = (denoised[0, 0].cpu().numpy() * 255.0).astype(np.uint8)

    grid = np.hstack([clean_np, noisy_np, pc_np, denoised_np])
    out_img = Image.fromarray(grid)
    out_img.save(output_path)
    print(f"[✓] Visual comparison saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PC-SSFlow v2 Demo")
    parser.add_argument("--image-path", type=str, default="data/BSD68/test001.png")
    parser.add_argument("--output-path", type=str, default="results/demo_comparison.png")
    parser.add_argument("--noise-sigma", type=float, default=0.08)
    parser.add_argument("--input-mode", type=str, default="srgb")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pcssflow_v2.pt")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, mps, cuda, cpu)")
    args = parser.parse_args()

    run_demo(
        image_path=args.image_path,
        output_path=args.output_path,
        noise_sigma=args.noise_sigma,
        input_mode=args.input_mode,
        checkpoint=args.checkpoint,
        device=args.device
    )
