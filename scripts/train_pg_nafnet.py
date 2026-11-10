#!/usr/bin/env python3
"""
Training Phase-Gated NAFNet (PG-NAFNet):
1. Single-Channel Input (in_ch=1): No naive input concatenation.
2. Phase-Gated SimpleGate (PG-Gate): Dynamically modulates feature conductivity across all 12 blocks.
3. Zero-Initialized Modulation: Warmstarted from base NAFNet with pc_proj initialized to zero.
4. Identical Constraints: 900 patch pairs across 60 scenes, 15 epochs, batch size 24, AdamW lr=5e-4.
"""

import os
import sys
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.pg_nafnet import PG_NAFNet
from scripts.train_phase_nafnet_fast_gpu import load_dataset_with_precomputed_pc
from src.utils.metrics import compute_image_metrics
from src.utils.device import get_default_device


def warmstart_pg_nafnet(model: PG_NAFNet, dev: torch.device, ckpt_candidates=None):
    """
    Transfers base weights from trained NAFNet checkpoint while preserving
    zero-initialization of the Phase-Gated projection layers.
    """
    if ckpt_candidates is None:
        ckpt_candidates = [
            "checkpoints/ablation_no_pc_baseline_fast.pt",
            "checkpoints/phase_nafnet_sidd.pt"
        ]

    chosen_ckpt = None
    for ckpt in ckpt_candidates:
        if os.path.exists(ckpt):
            chosen_ckpt = ckpt
            break

    if chosen_ckpt is None:
        print("No base checkpoint found, training from scratch.", flush=True)
        return model

    state = torch.load(chosen_ckpt, map_location=dev)
    model_state = model.state_dict()

    transferred = 0
    for k, v in state.items():
        # Handle body vs blocks naming if applicable
        target_k = k
        if target_k.startswith("body."):
            target_k = target_k.replace("body.", "blocks.")

        if target_k in model_state:
            if target_k == "intro.weight" and v.shape[1] == 2 and model_state[target_k].shape[1] == 1:
                # Slice channel 0
                model_state[target_k] = v[:, 0:1, :, :].clone()
                transferred += 1
            elif model_state[target_k].shape == v.shape:
                model_state[target_k] = v.clone()
                transferred += 1

    model.load_state_dict(model_state)
    print(f"Successfully warmstarted PG-NAFNet from {chosen_ckpt} ({transferred} tensors loaded)!", flush=True)

    # Double check zero-init of gamma_pc
    for i, block in enumerate(model.blocks):
        assert torch.all(block.gamma_pc == 0), f"Block {i} gamma_pc not zero!"
    print("Verified: all 12 Phase-Gated modulation layers are strictly zero-initialized.", flush=True)
    return model


def main():
    import resource
    dev = get_default_device("auto")
    epochs = 15
    batch_size = 16
    lr = 5e-4
    save_path = "checkpoints/pg_nafnet_sidd.pt"

    print("=" * 80, flush=True)
    print("TRAINING PHASE-GATED NAFNET (PG-NAFNet) - MEMORY EFFICIENT")
    print(f"Device: {dev} | Epochs: {epochs} | Batch Size: {batch_size} | lr: {lr}", flush=True)
    print("Architecture: Single-channel input (in_ch=1) + 12 Phase-Gated SimpleGates", flush=True)
    print("=" * 80, flush=True)

    # 1. Load dataset with precomputed PC (returns y1, pc1, y2)
    y1_all, pc1_all, y2_all = load_dataset_with_precomputed_pc(
        data_dir="data/SIDD/train/train",
        start_scene=40,
        max_scenes=60,
        crop_size=128,
        crops_per_scene=15,
        device=dev
    )
    n_samples = y1_all.shape[0]

    # Precompute PC on y2 as well for symmetric burst swapping
    print("Precomputing Phase Congruency on frame 2 for symmetric burst training...", flush=True)
    from src.wavelets.phase_congruency import PhaseCongruencyCalculator
    pc_calc = PhaseCongruencyCalculator(n_scales=2).to(dev)
    pc2_list = []
    with torch.no_grad():
        for i in range(0, n_samples, 32):
            pc2_list.append(pc_calc(y2_all[i : i + 32]))
    pc2_all = torch.cat(pc2_list, dim=0)
    del pc_calc, pc2_list
    if dev.type == "mps":
        torch.mps.empty_cache()
    print(f"[✓] Both frames with PC cached on {dev}: {n_samples} pairs ready.", flush=True)

    # 2. Instantiate and warmstart model
    model = PG_NAFNet(in_ch=1, c=72, n_blocks=12).to(dev)
    model = warmstart_pg_nafnet(model, dev=dev)

    n_params = sum(p.numel() for p in model.parameters())
    extra_params = sum(p.numel() for block in model.blocks for p in [block.gamma_pc])
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M) | Extra PG params: {extra_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    t0 = time.perf_counter()
    model.train()
    total_batches_per_epoch = math.ceil(n_samples / batch_size)

    print(f"\nBeginning training for {epochs} epochs ({total_batches_per_epoch} batches/epoch)...", flush=True)
    for ep in range(epochs):
        t_ep = time.perf_counter()
        perm = torch.randperm(n_samples, device=dev)
        y1_s = y1_all[perm]
        y2_s = y2_all[perm]
        pc1_s = pc1_all[perm]
        pc2_s = pc2_all[perm]

        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            b1 = y1_s[i : i + batch_size]
            b2 = y2_s[i : i + batch_size]
            pc1 = pc1_s[i : i + batch_size]
            pc2 = pc2_s[i : i + batch_size]

            # Symmetric frame swap with 50% probability
            if torch.rand(1).item() > 0.5:
                b1, b2 = b2, b1
                pc1, pc2 = pc2, pc1

            optimizer.zero_grad()
            pred = model(b1, pc1)
            loss = F.l1_loss(pred, b2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if n_batches % 15 == 0:
                print(f"  [Epoch {ep+1:2d}/{epochs:2d}] Batch {n_batches:2d}/{total_batches_per_epoch:2d} | Step Loss: {loss.item():.6f} | Elapsed: {time.perf_counter()-t0:.1f}s", flush=True)

        scheduler.step()
        if dev.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

        ep_time = time.perf_counter() - t_ep
        avg_loss = total_loss / max(n_batches, 1)
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        print(f"--> Completed Epoch [{ep+1:2d}/{epochs:2d}] ({ep_time:4.1f}s) | Avg Loss: {avg_loss:.6f} | RSS: {rss_mb:.1f} MB | LR: {scheduler.get_last_lr()[0]:.2e} | Total: {time.perf_counter()-t0:.1f}s", flush=True)

    # Save trained checkpoint
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n[✓] Saved PG-NAFNet checkpoint to {save_path} in {time.perf_counter()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
