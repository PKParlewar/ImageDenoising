#!/usr/bin/env python3
"""
Noise2Noise Phase-Domain Rectified Flow (N2N-PCFlow):
Self-Supervised learning on real smartphone burst captures (y1, y2).
Zero clean ground truth images used.
At test time: Takes a single noisy image y and integrates the learned ODE.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.pc_ssflow import PCSSFlowV2
from src.wavelets.dtcwt import complex_to_mag_phase, mag_phase_to_complex
from src.utils.metrics import compute_image_metrics
from scripts.evaluate import load_sidd_paired_scenes
from src.utils.device import get_default_device


def load_sidd_n2n_burst_crops(data_dir: str = "data/SIDD/train/train", start_scene: int = 40, max_scenes: int = 100, crop_size: int = 128, crops_per_scene: int = 10):
    """Loads synchronized (y1, y2) noisy crop pairs from separate burst frames of the same static scene."""
    pairs = []
    scene_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    selected_scenes = scene_dirs[start_scene : start_scene + max_scenes]
    print(f"Loading N2N burst pairs from {len(selected_scenes)} scenes (scenes {start_scene} to {start_scene + len(selected_scenes)})...")

    for s in selected_scenes:
        s_path = os.path.join(data_dir, s)
        files = os.listdir(s_path)
        noisy_files = sorted([f for f in files if "NOISY" in f and f.endswith(".PNG")])
        if len(noisy_files) < 2:
            continue
        
        try:
            img1 = Image.open(os.path.join(s_path, noisy_files[0])).convert("L")
            img2 = Image.open(os.path.join(s_path, noisy_files[1])).convert("L")
            w, h = img1.size
            if w < crop_size or h < crop_size:
                continue

            arr1 = np.array(img1, dtype=np.float32) / 255.0
            arr2 = np.array(img2, dtype=np.float32) / 255.0

            # Multiple random crops
            np.random.seed(42)
            for _ in range(crops_per_scene):
                left = np.random.randint(0, w - crop_size + 1)
                top = np.random.randint(0, h - crop_size + 1)
                
                c1 = arr1[top : top + crop_size, left : left + crop_size]
                c2 = arr2[top : top + crop_size, left : left + crop_size]
                
                t1 = torch.from_numpy(c1).unsqueeze(0).unsqueeze(0)
                t2 = torch.from_numpy(c2).unsqueeze(0).unsqueeze(0)
                pairs.append((t1, t2))
        except Exception as e:
            continue

    print(f"Successfully extracted {len(pairs)} synchronized noisy burst pairs.")
    return pairs


def main():
    dev = get_default_device("auto")
    print("==========================================================================")
    print(f"NOISE2NOISE PHASE-DOMAIN RECTIFIED FLOW (Device: {dev})")
    print("Self-Supervised: Zero Clean Ground Truth Images Used")
    print("==========================================================================")

    out_ckpt = "checkpoints/pcssflow_v2_n2n.pt"

    # Initialize model
    model = PCSSFlowV2(in_channels=1, hidden_dim=128, n_scales=2, physics_mode="srgb").to(dev)
    # Warmstart from base checkpoint
    base_ckpt = "checkpoints/pcssflow_v2_sidd_gpu.pt"
    if os.path.exists(base_ckpt):
        model.load_state_dict(torch.load(base_ckpt, map_location=dev))
        print(f"Warmstarted from {base_ckpt}")

    # Load 1,000 synchronized noisy burst pairs (scenes 40 to 140)
    train_pairs = load_sidd_n2n_burst_crops("data/SIDD/train/train", start_scene=40, max_scenes=100, crop_size=128, crops_per_scene=10)
    y1_all = torch.cat([p[0] for p in train_pairs], dim=0).to(dev)
    y2_all = torch.cat([p[1] for p in train_pairs], dim=0).to(dev)

    # Optimizer
    optimizer = torch.optim.AdamW(model.rectified_flow.velocity_net.parameters(), lr=0.0003, weight_decay=1e-5)
    epochs = 35
    b_size = 16
    n_samples = y1_all.shape[0]

    print(f"\nTraining N2N Phase-Flow on {n_samples} crop pairs ({epochs} epochs)...")
    t0_train = time.perf_counter()
    model.train()

    for ep in range(epochs):
        perm = torch.randperm(n_samples, device=dev)
        y1_shuf = y1_all[perm]
        y2_shuf = y2_all[perm]

        total_loss = 0.0
        n_b = 0

        for i in range(0, n_samples, b_size):
            y1_b = y1_shuf[i : i + b_size]
            y2_b = y2_shuf[i : i + b_size]
            b_curr = y1_b.shape[0]

            # Symmetric swap with 50% probability
            if torch.rand(1).item() > 0.5:
                y1_b, y2_b = y2_b, y1_b

            optimizer.zero_grad()

            # DTCWT decomposition of both noisy frames
            c_low_1, highs_1 = model.rectified_flow.dtcwt_fwd(y1_b)
            c_low_2, highs_2 = model.rectified_flow.dtcwt_fwd(y2_b)

            mags_1, phases_1 = [], []
            mags_2, phases_2 = [], []
            for h1, h2 in zip(highs_1, highs_2):
                m1, p1 = complex_to_mag_phase(h1)
                m2, p2 = complex_to_mag_phase(h2)
                mags_1.append(m1)
                phases_1.append(p1)
                mags_2.append(m2)
                phases_2.append(p2)

            # Random continuous time t in [0, 1]
            t = torch.rand(b_curr, device=dev)
            t_low = t.view(b_curr, 1, 1, 1)

            # Straight-line trajectory between noisy observations:
            # z_t = t * y2 + (1 - t) * y1
            clow_t = t_low * c_low_2 + (1.0 - t_low) * c_low_1
            u_clow = c_low_2 - c_low_1

            mags_t, phases_t = [], []
            u_mags = []
            for s in range(2):
                t_s = t.view(b_curr, 1, 1, 1, 1)
                m_t = t_s * mags_2[s] + (1.0 - t_s) * mags_1[s]
                p_t = phases_1[s]  # Phase is invariant
                mags_t.append(m_t)
                phases_t.append(p_t)
                u_mags.append(mags_2[s] - mags_1[s])

            # Conditioning on y1
            pc = model.pc_calc(y1_b)
            scale_map = model.rectified_flow.physics_scheduler(y1_b)

            v_clow, v_mags, _ = model.rectified_flow.velocity_net(
                clow_t, mags_t, phases_t, t, pc, scale_map, ssm_feat=pc
            )

            # Asymmetric phase-domain flow matching loss
            loss_clow = F.mse_loss(v_clow, u_clow)
            loss_mag = F.mse_loss(v_mags[0], u_mags[0]) + F.mse_loss(v_mags[1], u_mags[1])
            loss = loss_clow + 2.0 * loss_mag

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.rectified_flow.velocity_net.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_b += 1

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"Epoch [{ep+1:2d}/{epochs:2d}] - Loss: {total_loss / n_b:.6f} (elapsed: {time.perf_counter() - t0_train:.1f}s)")

    torch.save(model.state_dict(), out_ckpt)
    print(f"\nSaved N2N Phase-Flow model to {out_ckpt}")

    # EVALUATION ON HELD-OUT 40 VALIDATION SCENES (Single-Image at Test Time!)
    print("\n--- BENCHMARKING N2N PHASE-FLOW ON 40 HELD-OUT VALIDATION SCENES ---")
    print("Protocol: Pure Single-Image Inference (Takes only 1 noisy frame at test time)")
    model.eval()

    val_pairs = load_sidd_paired_scenes("data/SIDD/train/train", max_scenes=40, crop_size=128)

    for solver, steps, tta in [("euler", 2, False), ("heun", 4, False), ("heun", 4, True)]:
        psnr_l, ssim_l = [], []
        with torch.no_grad():
            for scene_name, noisy_t, gt_t in val_pairs:
                noisy = noisy_t.to(dev)
                clean = gt_t.to(dev)

                if tta:
                    outs = []
                    for k in [0, 1, 2, 3]:
                        rx = torch.rot90(noisy, k, [-2, -1])
                        orot = model.rectified_flow.denoise_sample(rx, num_steps=steps, solver=solver, use_spectral_sizing=True)
                        outs.append(torch.rot90(orot, -k, [-2, -1]))
                        fx = torch.flip(rx, dims=[-1])
                        oflip = model.rectified_flow.denoise_sample(fx, num_steps=steps, solver=solver, use_spectral_sizing=True)
                        outs.append(torch.rot90(torch.flip(oflip, dims=[-1]), -k, [-2, -1]))
                    out = torch.stack(outs, dim=0).mean(dim=0)
                else:
                    out = model.rectified_flow.denoise_sample(noisy, num_steps=steps, solver=solver, use_spectral_sizing=True)

                out = torch.clamp(out, 0.0, 1.0)
                m = compute_image_metrics(out, clean)
                psnr_l.append(m["psnr"])
                ssim_l.append(m["ssim"])

        tag = f"Solver: {solver:5s} | Steps: {steps:2d} | TTA: {tta!s:5s}"
        print(f"{tag:<40s} | PSNR: {np.mean(psnr_l):6.3f} dB | SSIM: {np.mean(ssim_l):6.4f}")

if __name__ == "__main__":
    main()
