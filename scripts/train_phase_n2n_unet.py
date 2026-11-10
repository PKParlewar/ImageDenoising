#!/usr/bin/env python3
"""
Hierarchical Phase-Domain Noise2Noise Network (PhaseN2N-UNet)
Self-supervised training on real-world smartphone camera burst captures (y1, y2).
Zero clean ground truth images used during training.
Receptive field > 128px to eliminate 8-16px demosaicing blotches.
Evaluates both Single-Image and 2-Frame Burst modes on 40 held-out validation scenes.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.metrics import compute_image_metrics
from src.utils.device import get_default_device
from src.wavelets.phase_congruency import PhaseCongruencyCalculator


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        res = self.act1(self.norm1(self.conv1(x)))
        res = self.ca(self.norm2(self.conv2(res)))
        return x + res


class PhaseN2NUNet(nn.Module):
    """Hierarchical Multi-Scale Network with Phase Congruency injection."""
    def __init__(self, in_ch=1, base_ch=48):
        super().__init__()
        # Level 1 (128x128)
        self.in_conv = nn.Conv2d(in_ch + 1, base_ch, 3, padding=1)  # +1 for PC map
        self.enc1 = nn.Sequential(ResBlock(base_ch), ResBlock(base_ch))
        
        # Level 2 (64x64)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(ResBlock(base_ch * 2), ResBlock(base_ch * 2))
        
        # Level 3 (32x32) Bottleneck
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            ResBlock(base_ch * 4),
            ResBlock(base_ch * 4),
            ResBlock(base_ch * 4)
        )
        
        # Level 2 Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = nn.Sequential(ResBlock(base_ch * 2), ResBlock(base_ch * 2))
        
        # Level 1 Decoder
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = nn.Sequential(ResBlock(base_ch), ResBlock(base_ch))
        
        # Output residual
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, pc):
        inp = torch.cat([x, pc], dim=1)
        e1 = self.enc1(self.in_conv(inp))
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        
        d2 = self.dec2(self.up2(b) + e2)
        d1 = self.dec1(self.up1(d2) + e1)
        
        res = self.out_conv(d1)
        return torch.clamp(x + res, 0.0, 1.0)


def load_sidd_n2n_burst_dataset(data_dir="data/SIDD/train/train", start_scene=40, max_scenes=120, crop_size=128, crops_per_scene=16):
    pairs = []
    scene_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    selected = scene_dirs[start_scene : start_scene + max_scenes]
    print(f"Loading N2N burst pairs from {len(selected)} scenes ({start_scene} to {start_scene + len(selected)})...", flush=True)

    for s in selected:
        s_path = os.path.join(data_dir, s)
        files = os.listdir(s_path)
        noisy_files = sorted([f for f in files if "NOISY" in f and f.endswith(".PNG")])
        if len(noisy_files) < 2:
            continue
        try:
            im1 = Image.open(os.path.join(s_path, noisy_files[0])).convert("L")
            im2 = Image.open(os.path.join(s_path, noisy_files[1])).convert("L")
            w, h = im1.size
            if w < crop_size or h < crop_size:
                continue
            a1 = np.array(im1, dtype=np.float32) / 255.0
            a2 = np.array(im2, dtype=np.float32) / 255.0

            np.random.seed(42 + len(pairs))
            for _ in range(crops_per_scene):
                left = np.random.randint(0, w - crop_size + 1)
                top = np.random.randint(0, h - crop_size + 1)
                c1 = a1[top : top + crop_size, left : left + crop_size]
                c2 = a2[top : top + crop_size, left : left + crop_size]
                pairs.append((torch.from_numpy(c1).unsqueeze(0).unsqueeze(0),
                              torch.from_numpy(c2).unsqueeze(0).unsqueeze(0)))
        except Exception:
            continue
    print(f"Loaded {len(pairs)} synchronized crop pairs.", flush=True)
    return pairs


def load_validation_scenes(data_dir="data/SIDD/train/train", max_scenes=40, crop_size=128):
    val_scenes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])[:max_scenes]
    val_data = []
    for s in val_scenes:
        s_path = os.path.join(data_dir, s)
        files = os.listdir(s_path)
        noisy_files = sorted([f for f in files if "NOISY" in f and f.endswith(".PNG")])
        gt_files = [f for f in files if "GT" in f and f.endswith(".PNG")]
        if len(noisy_files) >= 2 and len(gt_files) >= 1:
            im1 = Image.open(os.path.join(s_path, noisy_files[0])).convert("L")
            im2 = Image.open(os.path.join(s_path, noisy_files[1])).convert("L")
            imgt = Image.open(os.path.join(s_path, gt_files[0])).convert("L")
            w, h = im1.size
            cx, cy = w // 2, h // 2
            c1 = np.array(im1.crop((cx-crop_size//2, cy-crop_size//2, cx+crop_size//2, cy+crop_size//2)), dtype=np.float32) / 255.0
            c2 = np.array(im2.crop((cx-crop_size//2, cy-crop_size//2, cx+crop_size//2, cy+crop_size//2)), dtype=np.float32) / 255.0
            cg = np.array(imgt.crop((cx-crop_size//2, cy-crop_size//2, cx+crop_size//2, cy+crop_size//2)), dtype=np.float32) / 255.0
            val_data.append((s,
                             torch.from_numpy(c1).unsqueeze(0).unsqueeze(0),
                             torch.from_numpy(c2).unsqueeze(0).unsqueeze(0),
                             torch.from_numpy(cg).unsqueeze(0).unsqueeze(0)))
    return val_data


def run_tta_inference(model, pc_calc, img):
    """8-fold Test-Time Augmentation (4 rotations + horizontal flips)."""
    outs = []
    for k in [0, 1, 2, 3]:
        rx = torch.rot90(img, k, [-2, -1])
        rpc = pc_calc(rx)
        outs.append(torch.rot90(model(rx, rpc), -k, [-2, -1]))
        fx = torch.flip(rx, dims=[-1])
        fpc = pc_calc(fx)
        outs.append(torch.rot90(torch.flip(model(fx, fpc), dims=[-1]), -k, [-2, -1]))
    return torch.stack(outs, dim=0).mean(dim=0).clamp(0.0, 1.0)


def main():
    dev = get_default_device("auto")
    print("=" * 78, flush=True)
    print(f"TRAINING HIERARCHICAL PHASE-DOMAIN N2N UNET (Device: {dev})", flush=True)
    print("Self-Supervised: Zero Clean Ground Truth Images Used", flush=True)
    print("=" * 78, flush=True)

    pc_calc = PhaseCongruencyCalculator(n_scales=2).to(dev)
    model = PhaseN2NUNet(in_ch=1, base_ch=48).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)", flush=True)

    # Load 1,920 burst pairs from 120 scenes
    pairs = load_sidd_n2n_burst_dataset(start_scene=40, max_scenes=120, crop_size=128, crops_per_scene=16)
    y1_all = torch.cat([p[0] for p in pairs], dim=0).to(dev)
    y2_all = torch.cat([p[1] for p in pairs], dim=0).to(dev)
    n_samples = y1_all.shape[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    epochs = 30
    b_size = 16

    print(f"Training for {epochs} epochs on {n_samples} crop pairs...", flush=True)
    t0 = time.perf_counter()
    model.train()

    for ep in range(epochs):
        perm = torch.randperm(n_samples, device=dev)
        y1_s, y2_s = y1_all[perm], y2_all[perm]
        total_loss = 0.0
        n_b = 0

        for i in range(0, n_samples, b_size):
            b1 = y1_s[i : i + b_size]
            b2 = y2_s[i : i + b_size]

            # Symmetric swap with 50% probability
            if torch.rand(1).item() > 0.5:
                b1, b2 = b2, b1

            optimizer.zero_grad()
            with torch.no_grad():
                pc_b1 = pc_calc(b1)

            pred = model(b1, pc_b1)
            # N2N reconstruction loss
            loss_rec = F.l1_loss(pred, b2)
            # Edge consistency loss with b1 phase
            pred_pc = pc_calc(pred)
            loss_pc = F.l1_loss(pred_pc, pc_b1)
            loss = loss_rec + 0.1 * loss_pc

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_b += 1

        scheduler.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"Epoch [{ep+1:2d}/{epochs:2d}] - Loss: {total_loss/n_b:.6f} | LR: {scheduler.get_last_lr()[0]:.2e} | Elapsed: {time.perf_counter()-t0:.1f}s", flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/pcssflow_v2_n2n_unet.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved model checkpoint to {ckpt_path}", flush=True)

    # =========================================================================
    # BENCHMARK EVALUATION ON 40 HELD-OUT SCENES
    # =========================================================================
    print("\n" + "=" * 78, flush=True)
    print("BENCHMARK EVALUATION ON 40 HELD-OUT SIDD VALIDATION SCENES", flush=True)
    print("=" * 78, flush=True)
    model.eval()

    val_data = load_validation_scenes("data/SIDD/train/train", max_scenes=40, crop_size=128)
    print(f"Loaded {len(val_data)} validation scenes.", flush=True)

    m_s_dir = {"psnr": [], "ssim": []}
    m_s_tta = {"psnr": [], "ssim": []}
    m_b_dir = {"psnr": [], "ssim": []}
    m_b_tta = {"psnr": [], "ssim": []}
    m_b_avg = {"psnr": [], "ssim": []}

    with torch.no_grad():
        for s_name, y1_t, y2_t, gt_t in val_data:
            y1 = y1_t.to(dev)
            y2 = y2_t.to(dev)
            gt = gt_t.to(dev)

            pc1 = pc_calc(y1)
            pc2 = pc_calc(y2)

            # Single-Image Direct
            out_s_dir = model(y1, pc1)
            m1 = compute_image_metrics(out_s_dir, gt)
            m_s_dir["psnr"].append(m1["psnr"])
            m_s_dir["ssim"].append(m1["ssim"])

            # Single-Image + 8-fold TTA
            out_s_tta = run_tta_inference(model, pc_calc, y1)
            m2 = compute_image_metrics(out_s_tta, gt)
            m_s_tta["psnr"].append(m2["psnr"])
            m_s_tta["ssim"].append(m2["ssim"])

            # Two-Frame Burst Direct: (f(y1) + f(y2)) / 2
            out_y2_dir = model(y2, pc2)
            out_b_dir = (out_s_dir + out_y2_dir) / 2.0
            m3 = compute_image_metrics(out_b_dir, gt)
            m_b_dir["psnr"].append(m3["psnr"])
            m_b_dir["ssim"].append(m3["ssim"])

            # Two-Frame Burst + 8-fold TTA: (TTA(y1) + TTA(y2)) / 2
            out_y2_tta = run_tta_inference(model, pc_calc, y2)
            out_b_tta = (out_s_tta + out_y2_tta) / 2.0
            m4 = compute_image_metrics(out_b_tta, gt)
            m_b_tta["psnr"].append(m4["psnr"])
            m_b_tta["ssim"].append(m4["ssim"])

            # Averaged-Input Burst: f((y1+y2)/2) + TTA
            y_avg = (y1 + y2) / 2.0
            out_b_avg = run_tta_inference(model, pc_calc, y_avg)
            m5 = compute_image_metrics(out_b_avg, gt)
            m_b_avg["psnr"].append(m5["psnr"])
            m_b_avg["ssim"].append(m5["ssim"])

    print("\nRESULTS SUMMARY (40 Held-Out SIDD Scenes):", flush=True)
    print("-" * 78, flush=True)
    print(f"1. Single-Image Direct (1 Frame):          PSNR = {np.mean(m_s_dir['psnr']):.3f} dB | SSIM = {np.mean(m_s_dir['ssim']):.4f}", flush=True)
    print(f"2. Single-Image + 8-fold TTA (1 Frame):     PSNR = {np.mean(m_s_tta['psnr']):.3f} dB | SSIM = {np.mean(m_s_tta['ssim']):.4f}", flush=True)
    print(f"3. Two-Frame Burst Direct (2 Frames):       PSNR = {np.mean(m_b_dir['psnr']):.3f} dB | SSIM = {np.mean(m_b_dir['ssim']):.4f}", flush=True)
    print(f"4. Two-Frame Burst + 8-fold TTA (2 Frames): PSNR = {np.mean(m_b_tta['psnr']):.3f} dB | SSIM = {np.mean(m_b_tta['ssim']):.4f}", flush=True)
    print(f"5. Averaged-Input Burst + TTA (2 Frames):   PSNR = {np.mean(m_b_avg['psnr']):.3f} dB | SSIM = {np.mean(m_b_avg['ssim']):.4f}", flush=True)
    print("-" * 78, flush=True)


if __name__ == "__main__":
    main()
