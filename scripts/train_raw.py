#!/usr/bin/env python3
"""
PC-SSFlow v2 High-Performance RAW Bayer Training Entrypoint.

Trains PC-SSFlow v2 directly on linear raw Bayer sensor data with calibrated Poisson-Gaussian noise:
1. Converts paired smartphone scenes into linear Bayer raw mosaic via inverse ISP unprocessing.
2. Trains Stage 1: Self-supervised pretraining (Adaptive Masking + Phase Congruency Invariance).
3. Trains Stage 2: Phase-Domain Rectified Flow matching with physical Poisson-Gaussian scheduling.
4. Saves checkpoint to checkpoints/pcssflow_v2_raw.pt.
"""

import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.pc_ssflow import PCSSFlowV2
from src.models.loss import PCSSFlowLoss
from src.utils.raw_utils import inverse_isp_unprocess, DEFAULT_CAM2SRGB
from src.utils.device import get_default_device


def load_raw_bayer_training_pairs(
    data_dir: str = "data/SIDD/train/train",
    max_scenes: int = 40,
    crop_size: int = 128,
    crops_per_scene: int = 8,
    shot_noise_sigma: float = 0.01,
    read_noise_sigma: float = 0.005
) -> list:
    """
    Loads sRGB scenes from SIDD, unprocesses them to linear sensor RGB,
    mosaics to RGGB Bayer, and injects calibrated Poisson-Gaussian noise.
    """
    pairs = []
    if not os.path.exists(data_dir):
        return pairs

    scenes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    rng = np.random.RandomState(42)

    for scene in scenes[:max_scenes]:
        scene_path = os.path.join(data_dir, scene)
        files = os.listdir(scene_path)
        gt_files = sorted([f for f in files if "GT_SRGB" in f and f.endswith((".PNG", ".png"))])

        if not gt_files:
            continue

        gt_p = os.path.join(scene_path, gt_files[0])
        try:
            gt_img = Image.open(gt_p).convert("RGB")
            w, h = gt_img.size
            if w < crop_size or h < crop_size:
                continue

            for _ in range(crops_per_scene):
                left = int(rng.randint(0, w - crop_size + 1))
                top = int(rng.randint(0, h - crop_size + 1))
                # Ensure even crop coordinates for RGGB Bayer alignment
                left = (left // 2) * 2
                top = (top // 2) * 2

                crop = gt_img.crop((left, top, left + crop_size, top + crop_size))
                arr = np.array(crop, dtype=np.float32) / 255.0

                # Random rotation (0, 90, 180, 270) and flip
                k_rot = rng.randint(0, 4)
                if k_rot > 0:
                    arr = np.rot90(arr, k_rot, axes=(0, 1))
                if rng.rand() > 0.5:
                    arr = np.fliplr(arr)

                # Shape [1, 3, H, W]
                t_srgb = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0)

                # Unprocess sRGB to linear Raw Bayer with Poisson-Gaussian sensor noise
                noisy_bayer, clean_bayer = inverse_isp_unprocess(
                    t_srgb,
                    ccm=DEFAULT_CAM2SRGB,
                    shot_noise_sigma=shot_noise_sigma,
                    read_noise_sigma=read_noise_sigma
                )
                pairs.append((noisy_bayer, clean_bayer))
        except Exception:
            continue

    return pairs


def train_raw(
    data_dir: str = "data/SIDD/train/train",
    epochs_stage1: int = 25,
    epochs_stage2: int = 30,
    hidden_dim: int = 64,
    crop_size: int = 128,
    batch_size: int = 32,
    lr: float = 0.003,
    target_mode: str = "paired_gt",
    max_scenes: int = 40,
    crops_per_scene: int = 8,
    shot_noise_sigma: float = 0.01,
    read_noise_sigma: float = 0.005,
    device: str = "auto",
    save_path: str = "checkpoints/pcssflow_v2_raw.pt"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    dev = get_default_device(device)

    print("==========================================================================")
    print(f"   PC-SSFLOW v2 RAW BAYER TRAINING (Poisson-Gaussian Physical Flow)")
    print(f"   Device: {dev} | Crop: {crop_size}x{crop_size} | Batch Size: {batch_size} | Hidden Dim: {hidden_dim}")
    print(f"   Sensor Calibration: Shot Sigma={shot_noise_sigma}, Read Sigma={read_noise_sigma}")
    print("==========================================================================")

    model = PCSSFlowV2(
        in_channels=1,
        hidden_dim=hidden_dim,
        n_scales=2,
        physics_mode="raw",
        shot_noise_sigma=shot_noise_sigma,
        read_noise_sigma=read_noise_sigma
    ).to(dev)

    loss_fn = PCSSFlowLoss()
    opt_stage1 = torch.optim.Adam(
        list(model.mag_denoiser.parameters()) + list(model.ssm_backbone.parameters()),
        lr=lr
    )
    opt_stage2 = torch.optim.Adam(
        list(model.rectified_flow.velocity_net.parameters()) + list(model.ssm_backbone.parameters()),
        lr=lr
    )

    print("Synthesizing calibrated Raw Bayer training dataset...")
    pairs = load_raw_bayer_training_pairs(
        data_dir=data_dir,
        max_scenes=max_scenes,
        crop_size=crop_size,
        crops_per_scene=crops_per_scene,
        shot_noise_sigma=shot_noise_sigma,
        read_noise_sigma=read_noise_sigma
    )
    if not pairs:
        print("Fallback: synthesizing synthetic Bayer training pairs...")
        pairs = [(torch.rand(1, 1, crop_size, crop_size), torch.rand(1, 1, crop_size, crop_size)) for _ in range(64)]

    num_samples = len(pairs)
    print(f"Loaded {num_samples} calibrated Raw Bayer training crops.")

    # Move to GPU memory
    noisy_all = torch.cat([p[0] for p in pairs], dim=0).to(dev)
    gt_all = torch.cat([p[1] for p in pairs], dim=0).to(dev)

    # --- Stage 1: Self-Supervised Pretraining ---
    print(f"\n>>> Stage 1: Self-Supervised Pretraining on Raw Bayer ({epochs_stage1} epochs)...")
    start_t1 = time.perf_counter()
    for ep in range(epochs_stage1):
        perm = torch.randperm(num_samples, device=dev)
        noisy_shuffled = noisy_all[perm]
        total_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch = noisy_shuffled[i : i + batch_size]
            opt_stage1.zero_grad()
            out = model.forward_self_supervised_step(batch)
            l_dict = loss_fn.self_supervised_loss(
                rec_mags=out["rec_mags"],
                orig_mags=out["orig_mags"],
                masks=out["masks"],
                pc_pred=out["pc_pred"],
                pc_orig=out["pc_orig"],
                denoised_img=out["pseudo_clean"],
                y_noisy=batch
            )
            l_dict["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.mag_denoiser.parameters()) + list(model.ssm_backbone.parameters()),
                max_norm=1.0
            )
            opt_stage1.step()
            total_loss += l_dict["loss_total"].item()
            num_batches += 1

        if (ep + 1) % 5 == 0 or ep == epochs_stage1 - 1:
            elapsed = time.perf_counter() - start_t1
            print(f"  [Stage 1] Epoch {ep+1:2d}/{epochs_stage1:2d} | Loss: {total_loss / max(1, num_batches):.5f} | Elapsed: {elapsed:.1f}s")
            sys.stdout.flush()
    print(f"Stage 1 completed in {time.perf_counter() - start_t1:.2f}s.")

    # Target selection for Stage 2 flow matching
    if target_mode == "pseudo_clean":
        print("\n>>> Pre-computing Stage 1 pseudo-clean targets for Raw Flow matching...")
        pseudo_targets = []
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch = noisy_all[i : i + batch_size]
                p_c = model.forward_self_supervised_step(batch)["pseudo_clean"]
                pseudo_targets.append(p_c.clamp(0.0, 1.0))
        target_all = torch.cat(pseudo_targets, dim=0)
    else:
        print("\n>>> Using Ground Truth clean Bayer targets for Flow matching supervision...")
        target_all = gt_all

    # --- Stage 2: Phase-Domain Rectified Flow Matching ---
    print(f"\n>>> Stage 2: Phase-Domain Rectified Flow Matching ({epochs_stage2} epochs)...")
    start_t2 = time.perf_counter()
    for ep in range(epochs_stage2):
        perm = torch.randperm(num_samples, device=dev)
        noisy_shuffled = noisy_all[perm]
        target_shuffled = target_all[perm]
        total_flow_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            b_noisy = noisy_shuffled[i : i + batch_size]
            b_target = target_shuffled[i : i + batch_size]
            opt_stage2.zero_grad()

            pc_b = model.pc_calc(b_noisy)
            b_ssm = model.ssm_backbone(b_noisy, pc_b, window_size=model.warping_window)
            l_flow = model.rectified_flow.compute_flow_loss(b_noisy, b_target, ssm_feat=b_ssm)["loss"]
            l_flow.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.rectified_flow.velocity_net.parameters()) + list(model.ssm_backbone.parameters()),
                max_norm=1.0
            )
            opt_stage2.step()
            total_flow_loss += l_flow.item()
            num_batches += 1

        if (ep + 1) % 5 == 0 or ep == epochs_stage2 - 1:
            elapsed = time.perf_counter() - start_t2
            print(f"  [Stage 2] Epoch {ep+1:2d}/{epochs_stage2:2d} | Flow Loss: {total_flow_loss / max(1, num_batches):.5f} | Elapsed: {elapsed:.1f}s")
            sys.stdout.flush()
    print(f"Stage 2 completed in {time.perf_counter() - start_t2:.2f}s.")

    torch.save(model.state_dict(), save_path)
    print(f"\n[✓] Checkpoint saved successfully to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/SIDD/train/train")
    parser.add_argument("--epochs-stage1", type=int, default=20)
    parser.add_argument("--epochs-stage2", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--target-mode", type=str, default="paired_gt")
    parser.add_argument("--max-scenes", type=int, default=40)
    parser.add_argument("--crops-per-scene", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-path", type=str, default="checkpoints/pcssflow_v2_raw.pt")
    args = parser.parse_args()

    train_raw(
        data_dir=args.data_dir,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        target_mode=args.target_mode,
        max_scenes=args.max_scenes,
        crops_per_scene=args.crops_per_scene,
        device=args.device,
        save_path=args.save_path
    )
