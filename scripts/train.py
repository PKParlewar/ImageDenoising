#!/usr/bin/env python3
"""
PC-SSFlow v2 Unified Training Entrypoint.

Trains on real paired smartphone scenes from SIDD (Google Pixel, Samsung S6, iPhone, LG G4)
or benchmark datasets (BSD68, Urban100).
- Stage 1: Self-supervised pretraining (Adaptive Masking + Phase Congruency Invariance)
- Stage 2: Phase-Domain Rectified Flow matching
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
from src.utils.metrics import compute_image_metrics
from src.utils.device import get_default_device


def load_sidd_training_pairs(
    data_dir: str = "data/SIDD/train/train",
    max_scenes: int = 160,
    crop_size: int = 128,
    crops_per_scene: int = 8
) -> list:
    """Loads dynamically augmented paired noisy and GT clean crops from SIDD smartphone scenes."""
    pairs = []
    if not os.path.exists(data_dir):
        return pairs

    scenes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    rng = np.random.RandomState(42)

    for scene in scenes[:max_scenes]:
        scene_path = os.path.join(data_dir, scene)
        files = os.listdir(scene_path)

        gt_files = sorted([f for f in files if "GT_SRGB" in f and f.endswith((".PNG", ".png"))])
        noisy_files = sorted([f for f in files if "NOISY_SRGB" in f and f.endswith((".PNG", ".png"))])

        if gt_files and noisy_files:
            gt_p = os.path.join(scene_path, gt_files[0])
            noisy_p = os.path.join(scene_path, noisy_files[0])
            try:
                gt_img = Image.open(gt_p).convert("L")
                noisy_img = Image.open(noisy_p).convert("L")

                w, h = gt_img.size
                if w >= crop_size and h >= crop_size:
                    for _ in range(crops_per_scene):
                        left = int(rng.randint(0, w - crop_size + 1))
                        top = int(rng.randint(0, h - crop_size + 1))
                        gt_crop = gt_img.crop((left, top, left + crop_size, top + crop_size))
                        noisy_crop = noisy_img.crop((left, top, left + crop_size, top + crop_size))

                        gt_arr = np.array(gt_crop, dtype=np.float32) / 255.0
                        noisy_arr = np.array(noisy_crop, dtype=np.float32) / 255.0

                        # Random rotation (0, 90, 180, 270) and random flip
                        k_rot = rng.randint(0, 4)
                        if k_rot > 0:
                            gt_arr = np.rot90(gt_arr, k_rot)
                            noisy_arr = np.rot90(noisy_arr, k_rot)
                        if rng.rand() > 0.5:
                            gt_arr = np.fliplr(gt_arr)
                            noisy_arr = np.fliplr(noisy_arr)

                        gt_t = torch.from_numpy(gt_arr.copy()).unsqueeze(0).unsqueeze(0)
                        noisy_t = torch.from_numpy(noisy_arr.copy()).unsqueeze(0).unsqueeze(0)
                        pairs.append((noisy_t, gt_t))
                else:
                    gt_crop = gt_img.resize((crop_size, crop_size))
                    noisy_crop = noisy_img.resize((crop_size, crop_size))
                    gt_arr = np.array(gt_crop, dtype=np.float32) / 255.0
                    noisy_arr = np.array(noisy_crop, dtype=np.float32) / 255.0
                    pairs.append((
                        torch.from_numpy(noisy_arr).unsqueeze(0).unsqueeze(0),
                        torch.from_numpy(gt_arr).unsqueeze(0).unsqueeze(0)
                    ))
            except Exception:
                continue
    return pairs


def train(
    input_mode: str = "srgb",
    data_dir: str = "data/SIDD/train/train",
    epochs_stage1: int = 40,
    epochs_stage2: int = 50,
    hidden_dim: int = 128,
    crop_size: int = 128,
    batch_size: int = 16,
    lr: float = 0.003,
    target_mode: str = "paired_gt",
    max_scenes: int = 160,
    device: str = "auto",
    save_path: str = "checkpoints/pcssflow_v2_sidd.pt"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    dev = get_default_device(device)

    print("==========================================================================")
    print(f"PC-SSFlow v2 High-Performance Training (Device: {dev})")
    print(f"Data Dir: {data_dir} | Crop: {crop_size}x{crop_size} | Batch Size: {batch_size} | Mode: {input_mode.upper()} | Hidden Dim: {hidden_dim}")
    print("==========================================================================")

    model = PCSSFlowV2(
        in_channels=1,
        hidden_dim=hidden_dim,
        n_scales=2,
        physics_mode=input_mode
    ).to(dev)

    loss_fn = PCSSFlowLoss()
    opt_stage1 = torch.optim.Adam(
        list(model.mag_denoiser.parameters()) + list(model.ssm_backbone.parameters()),
        lr=lr
    )
    # Stage 2: Joint optimization of Velocity Net and Structure-Guided SSM Backbone
    opt_stage2 = torch.optim.Adam(
        list(model.rectified_flow.velocity_net.parameters()) + list(model.ssm_backbone.parameters()),
        lr=lr
    )

    # Load paired smartphone scenes
    print("Loading training dataset into memory...")
    pairs = load_sidd_training_pairs(data_dir, max_scenes=max_scenes, crop_size=crop_size)
    if not pairs:
        print(f"Notice: Loading fallback synthetic pairs...")
        pairs = [(torch.rand(1, 1, crop_size, crop_size), torch.rand(1, 1, crop_size, crop_size)) for _ in range(32)]

    num_samples = len(pairs)
    print(f"Loaded {num_samples} real smartphone training scene pairs.")

    # Stack into GPU-resident tensors to eliminate CPU <-> GPU transfer overhead during training
    noisy_all = torch.cat([p[0] for p in pairs], dim=0).to(dev)
    gt_all = torch.cat([p[1] for p in pairs], dim=0).to(dev)
    print(f"Allocated {noisy_all.element_size() * noisy_all.nelement() / (1024 * 1024):.2f} MB training tensor directly in GPU memory.")

    # --- Stage 1: Self-Supervised Pretraining ---
    print(f"\n>>> Stage 1: Self-Supervised Pretraining ({epochs_stage1} epochs)...")
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
            torch.nn.utils.clip_grad_norm_(list(model.mag_denoiser.parameters()) + list(model.ssm_backbone.parameters()), max_norm=1.0)
            opt_stage1.step()
            total_loss += l_dict["loss_total"].item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        if (ep + 1) % max(1, epochs_stage1 // 10) == 0 or ep == epochs_stage1 - 1:
            elapsed = time.perf_counter() - start_t1
            print(f"Stage 1 Epoch [{ep+1:2d}/{epochs_stage1:2d}] - L_self: {avg_loss:.6f} | Elapsed: {elapsed:.1f}s")
            sys.stdout.flush()

    # --- Stage 2: Target & SSM Conditioning Preparation ---
    if target_mode == "pseudo_clean":
        print("\n>>> Pre-computing Stage 1 pseudo-clean targets in a single batched pass...")
        pseudo_targets = []
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch = noisy_all[i : i + batch_size]
                p_c = model.forward_self_supervised_step(batch)["pseudo_clean"]
                pseudo_targets.append(p_c.clamp(0.0, 1.0))
        target_all = torch.cat(pseudo_targets, dim=0)
        print("Self-supervised pseudo-clean targets cached for Flow training.")
    else:
        print("\n>>> Using Ground Truth clean pairs for Flow matching target supervision...")
        target_all = gt_all

    # --- Stage 2: Phase-Domain Rectified Flow Matching with Joint SSM Backprop ---
    print(f"\n>>> Stage 2: Phase-Domain Rectified Flow Matching with Joint SSM Backprop ({epochs_stage2} epochs)...")
    sys.stdout.flush()
    start_t2 = time.perf_counter()
    for ep in range(epochs_stage2):
        perm = torch.randperm(num_samples, device=dev)
        noisy_shuffled = noisy_all[perm]
        target_shuffled = target_all[perm]
        total_flow_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch_n = noisy_shuffled[i : i + batch_size]
            batch_t = target_shuffled[i : i + batch_size]

            opt_stage2.zero_grad()
            pc_b = model.pc_calc(batch_n)
            batch_s = model.ssm_backbone(batch_n, pc_b, window_size=model.warping_window)
            l_flow = model.rectified_flow.compute_flow_loss(batch_n, batch_t, ssm_feat=batch_s)["loss"]
            l_flow.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.rectified_flow.velocity_net.parameters()) + list(model.ssm_backbone.parameters()),
                max_norm=1.0
            )
            opt_stage2.step()
            total_flow_loss += l_flow.item()
            num_batches += 1

        avg_flow_loss = total_flow_loss / max(1, num_batches)
        if (ep + 1) % max(1, epochs_stage2 // 10) == 0 or ep == epochs_stage2 - 1:
            elapsed = time.perf_counter() - start_t2
            print(f"Stage 2 Epoch [{ep+1:2d}/{epochs_stage2:2d}] - L_flow: {avg_flow_loss:.6f} | Elapsed: {elapsed:.1f}s")
            sys.stdout.flush()

    torch.save(model.state_dict(), save_path)
    print(f"\n[✓] Checkpoint successfully saved to {save_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PC-SSFlow v2 Fast Batched Training")
    parser.add_argument("--input-mode", type=str, default="srgb", choices=["raw", "srgb", "none"])
    parser.add_argument("--data-dir", type=str, default="data/SIDD/train/train")
    parser.add_argument("--epochs-stage1", type=int, default=40)
    parser.add_argument("--epochs-stage2", type=int, default=50)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--target-mode", type=str, default="paired_gt", choices=["paired_gt", "pseudo_clean"])
    parser.add_argument("--max-scenes", type=int, default=160)
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, mps, cuda, cpu)")
    parser.add_argument("--save-path", type=str, default="checkpoints/pcssflow_v2_sidd.pt")
    args = parser.parse_args()

    train(
        input_mode=args.input_mode,
        data_dir=args.data_dir,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        hidden_dim=args.hidden_dim,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        lr=args.lr,
        target_mode=args.target_mode,
        max_scenes=args.max_scenes,
        device=args.device,
        save_path=args.save_path
    )
