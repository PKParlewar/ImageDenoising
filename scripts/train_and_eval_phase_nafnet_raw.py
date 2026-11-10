#!/usr/bin/env python3
"""
Evaluate PC-Net (PhaseNAFNet) on Linear Raw Bayer Sensor Denoising
Evaluates on all 40 SIDD paired scenes under the exact same protocol as PC-Flow Raw.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.train_phase_nafnet import PhaseNAFNet, run_8fold_tta
from src.wavelets.phase_congruency import PhaseCongruencyCalculator
from src.utils.raw_utils import (
    inverse_isp_unprocess,
    forward_isp,
    DEFAULT_CAM2SRGB,
    GOOGLE_PIXEL_CAM2SRGB,
    SAMSUNG_S6_CAM2SRGB
)
from src.utils.metrics import compute_image_metrics
from src.utils.device import get_default_device


def train_and_eval_phase_nafnet_raw(
    data_dir="data/SIDD/train/train",
    save_path="checkpoints/phase_nafnet_raw.pt",
    max_val_scenes=40,
    crop_size=128,
    epochs=25,
    device="auto"
):
    dev = get_default_device(device)
    print("=" * 78)
    print(f"  PC-NET (PhaseNAFNet) LINEAR RAW BAYER TRAINING & 40-SCENE EVALUATION")
    print(f"  Device: {dev} | Parameters: 0.47M | Cropsize: {crop_size}x{crop_size}")
    print("=" * 78)

    scenes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    # 1. Synthesize training data from scenes 40+ to keep 0..39 strictly held-out
    train_scenes = scenes[40:120]
    val_scenes = scenes[:max_val_scenes]

    pc_calc = PhaseCongruencyCalculator(n_scales=2).to(dev)
    model = PhaseNAFNet(in_ch=2, c=72, n_blocks=12).to(dev)

    if os.path.exists(save_path):
        print(f"Loading cached weights from {save_path}...")
        model.load_state_dict(torch.load(save_path, map_location=dev))
    else:
        print(f"Collecting calibrated Raw Bayer training crops from {len(train_scenes)} scenes...")
        pairs = []
        rng = np.random.RandomState(42)
        for s in train_scenes:
            gt_f = [f for f in os.listdir(os.path.join(data_dir, s)) if "GT_SRGB" in f]
            if not gt_f:
                continue
            gt_p = os.path.join(data_dir, s, gt_f[0])
            try:
                img = Image.open(gt_p).convert("RGB")
                w, h = img.size
                if w < crop_size or h < crop_size:
                    continue
                for _ in range(8):
                    x = int(rng.randint(0, w - crop_size)) // 2 * 2
                    y = int(rng.randint(0, h - crop_size)) // 2 * 2
                    crop = img.crop((x, y, x + crop_size, y + crop_size))
                    arr = np.array(crop, dtype=np.float32) / 255.0
                    t_srgb = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                    nb, cb = inverse_isp_unprocess(t_srgb, shot_noise_sigma=0.008, read_noise_sigma=0.003)
                    pairs.append((nb, cb))
            except Exception:
                continue

        print(f"Synthesized {len(pairs)} raw Bayer training crops.")
        noisy_all = torch.cat([p[0] for p in pairs], dim=0).to(dev)
        clean_all = torch.cat([p[1] for p in pairs], dim=0).to(dev)

        opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        criterion = nn.L1Loss()
        b_size = 16
        N = len(pairs)

        print(f"Training PhaseNAFNet for {epochs} epochs on Raw Bayer...")
        t0 = time.perf_counter()
        model.train()
        for ep in range(epochs):
            perm = torch.randperm(N, device=dev)
            n_s, c_s = noisy_all[perm], clean_all[perm]
            for i in range(0, N, b_size):
                bn = n_s[i : i + b_size]
                bc = c_s[i : i + b_size]
                opt.zero_grad()
                with torch.no_grad():
                    pc = pc_calc(bn)
                pred = model(bn, pc)
                loss = criterion(pred, bc)
                loss.backward()
                opt.step()
            sched.step()
        print(f"Training completed in {time.perf_counter() - t0:.1f}s.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}.")

    # 2. Comprehensive 40-Scene Validation under identical protocol
    model.eval()
    raw_noisy_psnr, raw_denoised_psnr = [], []
    raw_noisy_ssim, raw_denoised_ssim = [], []
    srgb_noisy_psnr, srgb_denoised_psnr = [], []
    srgb_noisy_ssim, srgb_denoised_ssim = [], []

    per_scene = []
    print("\nEvaluating on 40 Held-Out Validation Scenes...")
    print("-" * 78)

    for idx, scene in enumerate(val_scenes):
        scene_p = os.path.join(data_dir, scene)
        gt_files = [f for f in os.listdir(scene_p) if "GT_SRGB" in f]
        if not gt_files:
            continue
        gt_p = os.path.join(scene_p, gt_files[0])
        try:
            img = Image.open(gt_p).convert("RGB")
            w, h = img.size
            if w < crop_size or h < crop_size:
                continue
            # Center crop
            left = ((w - crop_size) // 4) * 2
            top = ((h - crop_size) // 4) * 2
            crop = img.crop((left, top, left + crop_size, top + crop_size))
            arr = np.array(crop, dtype=np.float32) / 255.0
            t_srgb = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev)

            if "GP" in scene:
                ccm = GOOGLE_PIXEL_CAM2SRGB.to(dev)
            elif "S6" in scene:
                ccm = SAMSUNG_S6_CAM2SRGB.to(dev)
            else:
                ccm = DEFAULT_CAM2SRGB.to(dev)

            noisy_bayer, clean_bayer = inverse_isp_unprocess(
                t_srgb, ccm=ccm, shot_noise_sigma=0.008, read_noise_sigma=0.003
            )

            with torch.no_grad():
                pc = pc_calc(noisy_bayer)
                denoised_bayer = model(noisy_bayer, pc)
                denoised_bayer = torch.clamp(denoised_bayer, 0.0, 1.0)

            # Raw metrics
            m_raw_n = compute_image_metrics(noisy_bayer, clean_bayer)
            m_raw_d = compute_image_metrics(denoised_bayer, clean_bayer)
            raw_noisy_psnr.append(m_raw_n["psnr"])
            raw_denoised_psnr.append(m_raw_d["psnr"])
            raw_noisy_ssim.append(m_raw_n["ssim"])
            raw_denoised_ssim.append(m_raw_d["ssim"])

            # Post-ISP metrics
            noisy_srgb = forward_isp(noisy_bayer, ccm=ccm)
            denoised_srgb = forward_isp(denoised_bayer, ccm=ccm)
            gt_srgb_recon = forward_isp(clean_bayer, ccm=ccm)

            m_srgb_n = compute_image_metrics(noisy_srgb, gt_srgb_recon)
            m_srgb_d = compute_image_metrics(denoised_srgb, gt_srgb_recon)
            srgb_noisy_psnr.append(m_srgb_n["psnr"])
            srgb_denoised_psnr.append(m_srgb_d["psnr"])
            srgb_noisy_ssim.append(m_srgb_n["ssim"])
            srgb_denoised_ssim.append(m_srgb_d["ssim"])

            per_scene.append({
                "scene": scene,
                "raw_noisy_psnr": m_raw_n["psnr"],
                "raw_denoised_psnr": m_raw_d["psnr"],
                "raw_noisy_ssim": m_raw_n["ssim"],
                "raw_denoised_ssim": m_raw_d["ssim"],
                "srgb_noisy_psnr": m_srgb_n["psnr"],
                "srgb_denoised_psnr": m_srgb_d["psnr"],
            })

            if (idx + 1) % 5 == 0 or idx == 0:
                print(f"[{idx+1:2d}/40] {scene[:20]:20s} | Raw: {m_raw_n['psnr']:5.2f} -> {m_raw_d['psnr']:5.2f} dB ({m_raw_d['psnr'] - m_raw_n['psnr']:+5.2f} dB) | SSIM: {m_raw_d['ssim']:.4f}")
        except Exception as e:
            continue

    summary = {
        "raw_domain": {
            "mean_noisy_psnr": float(np.mean(raw_noisy_psnr)),
            "mean_denoised_psnr": float(np.mean(raw_denoised_psnr)),
            "net_psnr_gain": float(np.mean(raw_denoised_psnr) - np.mean(raw_noisy_psnr)),
            "peak_scene_psnr": float(np.max(raw_denoised_psnr)),
            "mean_noisy_ssim": float(np.mean(raw_noisy_ssim)),
            "mean_denoised_ssim": float(np.mean(raw_denoised_ssim)),
            "net_ssim_gain": float(np.mean(raw_denoised_ssim) - np.mean(raw_noisy_ssim)),
            "peak_scene_ssim": float(np.max(raw_denoised_ssim)),
        },
        "post_isp_srgb": {
            "mean_noisy_psnr": float(np.mean(srgb_noisy_psnr)),
            "mean_denoised_psnr": float(np.mean(srgb_denoised_psnr)),
            "net_psnr_gain": float(np.mean(srgb_denoised_psnr) - np.mean(srgb_noisy_psnr)),
            "mean_noisy_ssim": float(np.mean(srgb_noisy_ssim)),
            "mean_denoised_ssim": float(np.mean(srgb_denoised_ssim)),
            "net_ssim_gain": float(np.mean(srgb_denoised_ssim) - np.mean(srgb_noisy_ssim)),
        },
        "scenes_evaluated": len(per_scene)
    }

    out_json = "results/raw_pipeline/eval_pc_net_raw_summary.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("      FINAL PC-NET (PhaseNAFNet) LINEAR RAW BAYER BENCHMARK RESULTS")
    print("=" * 78)
    print(f"Mean Raw Noisy PSNR:    {summary['raw_domain']['mean_noisy_psnr']:6.2f} dB  |  SSIM: {summary['raw_domain']['mean_noisy_ssim']:.4f}")
    print(f"Mean Raw Denoised PSNR: {summary['raw_domain']['mean_denoised_psnr']:6.2f} dB  |  SSIM: {summary['raw_domain']['mean_denoised_ssim']:.4f}")
    print(f"Net Raw PSNR Gain:      {summary['raw_domain']['net_psnr_gain']:+6.2f} dB  |  Gain: {summary['raw_domain']['net_ssim_gain']:+.4f}")
    print(f"Peak Scene Raw PSNR:    {summary['raw_domain']['peak_scene_psnr']:6.2f} dB  |  Peak: {summary['raw_domain']['peak_scene_ssim']:.4f}")
    print("-" * 78)
    print(f"Post-ISP sRGB PSNR:     {summary['post_isp_srgb']['mean_denoised_psnr']:6.2f} dB  |  SSIM: {summary['post_isp_srgb']['mean_denoised_ssim']:.4f}")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    train_and_eval_phase_nafnet_raw()
