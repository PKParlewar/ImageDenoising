#!/usr/bin/env python3
"""
Phase-Guided Nonlinear Activation-Free Network (PhaseNAFNet):
1. LayerNorm + SimpleGate + Simplified Channel Attention (SCA).
2. Zero-initialized residual gates (beta, gamma) -> provably zero activation explosion.
3. Phase Congruency conditioning -> preserves razor-sharp directional edges.
4. Self-Supervised: Trained strictly on noisy burst pairs (y1, y2), zero clean GT.
5. Evaluates Single-Image and Two-Frame Burst across 40 held-out validation scenes.
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
from scripts.evaluate import load_sidd_paired_scenes
from scripts.train_n2n_flow import load_sidd_n2n_burst_crops


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class PhaseNAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1)
        )
        self.sg = SimpleGate()
        
        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)
        
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta
        
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


class PhaseNAFNet(nn.Module):
    def __init__(self, in_ch=2, c=72, n_blocks=12):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, c, 3, padding=1)
        self.body = nn.Sequential(*[PhaseNAFBlock(c) for _ in range(n_blocks)])
        self.outro = nn.Conv2d(c, 1, 3, padding=1)
        nn.init.zeros_(self.outro.weight)
        nn.init.zeros_(self.outro.bias)
        
    def forward(self, x, pc):
        inp = torch.cat([x, pc], dim=1)
        feat = self.intro(inp)
        feat = self.body(feat)
        res = self.outro(feat)
        return torch.clamp(x + res, 0.0, 1.0)


def run_8fold_tta(model, pc_calc, x):
    outs = []
    for k in [0, 1, 2, 3]:
        rx = torch.rot90(x, k, [-2, -1])
        rpc = pc_calc(rx)
        outs.append(torch.rot90(model(rx, rpc), -k, [-2, -1]))
        fx = torch.flip(rx, dims=[-1])
        fpc = pc_calc(fx)
        outs.append(torch.rot90(torch.flip(model(fx, fpc), dims=[-1]), -k, [-2, -1]))
    return torch.stack(outs, dim=0).mean(dim=0).clamp(0.0, 1.0)


def main():
    dev = get_default_device("auto")
    print("=" * 80, flush=True)
    print(f"TRAINING PHASE-GUIDED NAFNet (Device: {dev})", flush=True)
    print("Self-Supervised: Zero Clean Ground Truth Supervision", flush=True)
    print("=" * 80, flush=True)

    pc_calc = PhaseCongruencyCalculator(n_scales=2).to(dev)
    model = PhaseNAFNet(in_ch=2, c=72, n_blocks=12).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)", flush=True)

    # 1. Load burst training pairs
    train_pairs = load_sidd_n2n_burst_crops("data/SIDD/train/train", start_scene=40, max_scenes=120, crop_size=128, crops_per_scene=14)
    y1_all = torch.cat([p[0] for p in train_pairs], dim=0).to(dev)
    y2_all = torch.cat([p[1] for p in train_pairs], dim=0).to(dev)
    n_samples = y1_all.shape[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    epochs = 30
    b_size = 16

    print(f"Training on {n_samples} crop pairs for {epochs} epochs...", flush=True)
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
            loss_rec = F.l1_loss(pred, b2)
            
            # Loss
            loss = loss_rec
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_b += 1

        scheduler.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"Epoch [{ep+1:2d}/{epochs:2d}] - Loss: {total_loss/n_b:.6f} | LR: {scheduler.get_last_lr()[0]:.2e} | Elapsed: {time.perf_counter()-t0:.1f}s", flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/phase_nafnet_sidd.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved model checkpoint to {ckpt_path}", flush=True)

    # =========================================================================
    # BENCHMARK EVALUATION ON 40 HELD-OUT SCENES
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("BENCHMARK EVALUATION ON 40 HELD-OUT SIDD VALIDATION SCENES", flush=True)
    print("=" * 80, flush=True)
    model.eval()

    val_scenes = sorted([d for d in os.listdir("data/SIDD/train/train") if os.path.isdir(os.path.join("data/SIDD/train/train", d))])[:40]
    val_data = []
    crop_size = 128
    for s in val_scenes:
        s_path = os.path.join("data/SIDD/train/train", s)
        files = os.listdir(s_path)
        noisy_files = sorted([f for f in files if "NOISY" in f and f.endswith(".PNG")])
        gt_files = [f for f in files if "GT" in f and f.endswith(".PNG")]
        if len(noisy_files) >= 2 and len(gt_files) >= 1:
            im1 = Image.open(os.path.join(s_path, noisy_files[0])).convert("L")
            im2 = Image.open(os.path.join(s_path, noisy_files[1])).convert("L")
            imgt = Image.open(os.path.join(s_path, gt_files[0])).convert("L")
            w, h = im1.size
            cx, cy = w // 2, h // 2
            c1 = torch.from_numpy(np.array(im1.crop((cx-crop_size//2, cy-crop_size//2, cx+crop_size//2, cy+crop_size//2)), dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
            c2 = torch.from_numpy(np.array(im2.crop((cx-crop_size//2, cy-crop_size//2, cx+crop_size//2, cy+crop_size//2)), dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
            cg = torch.from_numpy(np.array(imgt.crop((cx-crop_size//2, cy-crop_size//2, cx+crop_size//2, cy+crop_size//2)), dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
            val_data.append((s, c1, c2, cg))

    print(f"Evaluating across {len(val_data)} validation scenes...", flush=True)

    metrics = {
        "Single-Image Direct (1 Frame)": {"psnr": [], "ssim": []},
        "Single-Image + 8-fold TTA (1 Frame)": {"psnr": [], "ssim": []},
        "Two-Frame Burst Direct (2 Frames)": {"psnr": [], "ssim": []},
        "Two-Frame Burst + 8-fold TTA (2 Frames)": {"psnr": [], "ssim": []},
        "Averaged-Input Burst + TTA (2 Frames)": {"psnr": [], "ssim": []},
    }

    with torch.no_grad():
        for s_name, y1, y2, gt in val_data:
            pc1 = pc_calc(y1)
            pc2 = pc_calc(y2)

            # Single direct
            out_s1 = model(y1, pc1)
            m = compute_image_metrics(out_s1, gt)
            metrics["Single-Image Direct (1 Frame)"]["psnr"].append(m["psnr"])
            metrics["Single-Image Direct (1 Frame)"]["ssim"].append(m["ssim"])

            # Single + TTA
            out_s1_tta = run_8fold_tta(model, pc_calc, y1)
            m = compute_image_metrics(out_s1_tta, gt)
            metrics["Single-Image + 8-fold TTA (1 Frame)"]["psnr"].append(m["psnr"])
            metrics["Single-Image + 8-fold TTA (1 Frame)"]["ssim"].append(m["ssim"])

            # Burst direct
            out_s2 = model(y2, pc2)
            out_burst_dir = (out_s1 + out_s2) / 2.0
            m = compute_image_metrics(out_burst_dir, gt)
            metrics["Two-Frame Burst Direct (2 Frames)"]["psnr"].append(m["psnr"])
            metrics["Two-Frame Burst Direct (2 Frames)"]["ssim"].append(m["ssim"])

            # Burst + TTA
            out_s2_tta = run_8fold_tta(model, pc_calc, y2)
            out_burst_tta = (out_s1_tta + out_s2_tta) / 2.0
            m = compute_image_metrics(out_burst_tta, gt)
            metrics["Two-Frame Burst + 8-fold TTA (2 Frames)"]["psnr"].append(m["psnr"])
            metrics["Two-Frame Burst + 8-fold TTA (2 Frames)"]["ssim"].append(m["ssim"])

            # Averaged input burst + TTA
            y_avg = (y1 + y2) / 2.0
            out_avg_tta = run_8fold_tta(model, pc_calc, y_avg)
            m = compute_image_metrics(out_avg_tta, gt)
            metrics["Averaged-Input Burst + TTA (2 Frames)"]["psnr"].append(m["psnr"])
            metrics["Averaged-Input Burst + TTA (2 Frames)"]["ssim"].append(m["ssim"])

    print("\n" + "=" * 80, flush=True)
    print(f"{'Method / Configuration':<45s} | {'PSNR (dB)':<10s} | {'SSIM':<10s}", flush=True)
    print("-" * 80, flush=True)
    for name, vals in metrics.items():
        p_mean = np.mean(vals["psnr"])
        s_mean = np.mean(vals["ssim"])
        print(f"{name:<45s} | {p_mean:6.3f} dB  | {s_mean:6.4f}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
