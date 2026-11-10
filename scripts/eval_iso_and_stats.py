#!/usr/bin/env python3
"""
Phase 4: Statistical Rigor, ISO Stratification & Confidence Intervals
Evaluates PC-Net (PhaseNAFNet) across all 40 held-out SIDD validation scenes:
1. Per-scene ISO sensitivity and camera hardware metadata extraction.
2. Stratification into ISO Regimes: Low (<=100), Moderate (400-800), High (1600), Extreme (3200-6400).
3. Stratification by Camera Hardware (Google Pixel, Samsung S6, Nexus 6, LG G4, iPhone).
4. Calculation of Mean, Standard Deviation (±std), and 95% Confidence Intervals (CI95).
"""

import os
import sys
import json
import math
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.train_phase_nafnet import PhaseNAFNet, run_8fold_tta
from src.wavelets.phase_congruency import PhaseCongruencyCalculator
from src.utils.metrics import compute_image_metrics
from src.utils.device import get_default_device


def parse_sidd_scene_name(scene_dir: str):
    """
    Parses SIDD scene directory name.
    Format example: 0001_001_S6_00100_00060_3200_L
    [SceneID]_[InstanceID]_[Camera]_[ISO]_[ShutterSpeed]_[Brightness]
    """
    parts = scene_dir.split("_")
    camera_code = parts[2] if len(parts) > 2 else "Unknown"
    iso_val = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 100

    cam_map = {
        "S6": "Samsung Galaxy S6",
        "GP": "Google Pixel",
        "N6": "Motorola Nexus 6",
        "G4": "LG G4",
        "IP": "Apple iPhone 7"
    }
    camera_name = cam_map.get(camera_code, f"Camera {camera_code}")
    return camera_code, camera_name, iso_val


def compute_ci95(values):
    """Computes Mean, Std, and 95% Confidence Interval."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, (0.0, 0.0)
    mean = float(np.mean(values))
    std = float(np.std(values))
    margin = 1.96 * (std / math.sqrt(n)) if n > 1 else 0.0
    return mean, std, (mean - margin, mean + margin)


def main():
    dev = get_default_device("auto")
    print("=" * 80, flush=True)
    print("PHASE 4: STATISTICAL RIGOR, ISO STRATIFICATION & 95% CONFIDENCE INTERVALS")
    print(f"Device: {dev} | Checkpoint: checkpoints/phase_nafnet_scaled_sidd.pt")
    print("=" * 80, flush=True)

    # 1. Load trained flagship model
    model = PhaseNAFNet(in_ch=2, c=72, n_blocks=12).to(dev)
    ckpt_path = "checkpoints/phase_nafnet_scaled_sidd.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/phase_nafnet_sidd.pt"
    print(f"Loading checkpoint from {ckpt_path}...", flush=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()

    pc_calc = PhaseCongruencyCalculator(n_scales=2).to(dev)
    data_dir = "data/SIDD/train/train"
    val_scenes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])[:40]

    scene_results = []
    print(f"\nEvaluating {len(val_scenes)} held-out scenes...", flush=True)

    with torch.no_grad():
        for idx, s in enumerate(val_scenes):
            s_path = os.path.join(data_dir, s)
            f_noisy = sorted([f for f in os.listdir(s_path) if "NOISY" in f and f.endswith(".PNG")])
            f_gt = sorted([f for f in os.listdir(s_path) if "GT" in f and f.endswith(".PNG")])

            if len(f_noisy) < 2 or not f_gt:
                continue

            cam_code, cam_name, iso = parse_sidd_scene_name(s)

            im1 = Image.open(os.path.join(s_path, f_noisy[0])).convert("L")
            im2 = Image.open(os.path.join(s_path, f_noisy[1])).convert("L")
            imgt = Image.open(os.path.join(s_path, f_gt[0])).convert("L")

            w, h = im1.size
            cx, cy = w // 2, h // 2
            c1 = torch.from_numpy(np.array(im1.crop((cx-128, cy-128, cx+128, cy+128)), dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
            c2 = torch.from_numpy(np.array(im2.crop((cx-128, cy-128, cx+128, cy+128)), dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
            cg = torch.from_numpy(np.array(imgt.crop((cx-128, cy-128, cx+128, cy+128)), dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)

            # Input noisy metrics
            m_in = compute_image_metrics(c1, cg)

            # 1. Single-Image Direct (Zero-Oracle)
            p1 = pc_calc(c1)
            o1 = model(c1, p1)
            m_single = compute_image_metrics(o1, cg)

            # 2. Two-Frame Burst + 8-Fold TTA (Zero-Oracle)
            o1_tta = run_8fold_tta(model, pc_calc, c1)
            o2_tta = run_8fold_tta(model, pc_calc, c2)
            burst_tta = (o1_tta + o2_tta) / 2.0
            m_burst = compute_image_metrics(burst_tta, cg)

            res_entry = {
                "scene": s,
                "camera_code": cam_code,
                "camera_name": cam_name,
                "iso": iso,
                "noisy_psnr": m_in["psnr"],
                "noisy_ssim": m_in["ssim"],
                "single_psnr": m_single["psnr"],
                "single_ssim": m_single["ssim"],
                "burst_psnr": m_burst["psnr"],
                "burst_ssim": m_burst["ssim"],
            }
            scene_results.append(res_entry)

    # -------------------------------------------------------------------------
    # 2. Global Statistical Metrics (Mean, Std, 95% Confidence Intervals)
    # -------------------------------------------------------------------------
    single_psnrs = [r["single_psnr"] for r in scene_results]
    single_ssims = [r["single_ssim"] for r in scene_results]
    burst_psnrs = [r["burst_psnr"] for r in scene_results]
    burst_ssims = [r["burst_ssim"] for r in scene_results]

    m_s_p, std_s_p, ci_s_p = compute_ci95(single_psnrs)
    m_s_s, std_s_s, ci_s_s = compute_ci95(single_ssims)
    m_b_p, std_b_p, ci_b_p = compute_ci95(burst_psnrs)
    m_b_s, std_b_s, ci_b_s = compute_ci95(burst_ssims)

    print("\n" + "=" * 80, flush=True)
    print(f"GLOBAL BENCHMARK STATISTICS (N={len(scene_results)} SCENES, STRICT ZERO-ORACLE):")
    print("=" * 80, flush=True)
    print(f"1. Single-Image Direct: PSNR = {m_s_p:.2f} ± {std_s_p:.2f} dB (95% CI: [{ci_s_p[0]:.2f}, {ci_s_p[1]:.2f}]) | SSIM = {m_s_s:.4f} ± {std_s_s:.4f} (95% CI: [{ci_s_s[0]:.4f}, {ci_s_s[1]:.4f}])", flush=True)
    print(f"2. Two-Frame Burst:     PSNR = {m_b_p:.2f} ± {std_b_p:.2f} dB (95% CI: [{ci_b_p[0]:.2f}, {ci_b_p[1]:.2f}]) | SSIM = {m_b_s:.4f} ± {std_b_s:.4f} (95% CI: [{ci_b_s[0]:.4f}, {ci_b_s[1]:.4f}])", flush=True)
    print("=" * 80, flush=True)

    # -------------------------------------------------------------------------
    # 3. ISO Regime Stratification
    # -------------------------------------------------------------------------
    iso_bins = {
        "Low ISO (≤ 100)": [r for r in scene_results if r["iso"] <= 100],
        "Moderate ISO (400–800)": [r for r in scene_results if 100 < r["iso"] <= 800],
        "High ISO (1600)": [r for r in scene_results if 800 < r["iso"] <= 1600],
        "Extreme ISO (3200–6400)": [r for r in scene_results if r["iso"] > 1600],
    }

    print("\n" + "=" * 80, flush=True)
    print("ISO REGIME DIFFICULTY STRATIFICATION TABLE:")
    print("=" * 80, flush=True)
    print(f"{'ISO Regime':<26} | {'Scenes':<6} | {'Input Noisy':<12} | {'Single-Image PSNR':<18} | {'Burst PSNR':<18} | {'Burst SSIM':<12}")
    print("-" * 105, flush=True)

    iso_stats = {}
    for regime_name, entries in iso_bins.items():
        if not entries:
            continue
        n_sc = len(entries)
        in_p = float(np.mean([e["noisy_psnr"] for e in entries]))
        s_p, s_std, _ = compute_ci95([e["single_psnr"] for e in entries])
        b_p, b_std, _ = compute_ci95([e["burst_psnr"] for e in entries])
        b_s, b_s_std, _ = compute_ci95([e["burst_ssim"] for e in entries])

        iso_stats[regime_name] = {
            "n_scenes": n_sc,
            "input_noisy_psnr": in_p,
            "single_psnr_mean": s_p,
            "single_psnr_std": s_std,
            "burst_psnr_mean": b_p,
            "burst_psnr_std": b_std,
            "burst_ssim_mean": b_s,
            "burst_ssim_std": b_s_std
        }
        print(f"{regime_name:<26} | {n_sc:<6} | {in_p:5.2f} dB     | {s_p:5.2f} ± {s_std:4.2f} dB   | {b_p:5.2f} ± {b_std:4.2f} dB   | {b_s:.4f} ± {b_s_std:.4f}", flush=True)
    print("=" * 80, flush=True)

    # -------------------------------------------------------------------------
    # 4. Camera Sensor Breakdown
    # -------------------------------------------------------------------------
    cam_groups = {}
    for r in scene_results:
        cam_groups.setdefault(r["camera_name"], []).append(r)

    print("\n" + "=" * 80, flush=True)
    print("CAMERA HARDWARE SENSOR BREAKDOWN TABLE:")
    print("=" * 80, flush=True)
    print(f"{'Camera Sensor Model':<24} | {'Scenes':<6} | {'Input Noisy':<12} | {'Single-Image PSNR':<18} | {'Burst PSNR':<18} | {'Burst SSIM':<12}")
    print("-" * 105, flush=True)

    cam_stats = {}
    for cam_name, entries in sorted(cam_groups.items()):
        n_sc = len(entries)
        in_p = float(np.mean([e["noisy_psnr"] for e in entries]))
        s_p, s_std, _ = compute_ci95([e["single_psnr"] for e in entries])
        b_p, b_std, _ = compute_ci95([e["burst_psnr"] for e in entries])
        b_s, b_s_std, _ = compute_ci95([e["burst_ssim"] for e in entries])

        cam_stats[cam_name] = {
            "n_scenes": n_sc,
            "input_noisy_psnr": in_p,
            "single_psnr_mean": s_p,
            "single_psnr_std": s_std,
            "burst_psnr_mean": b_p,
            "burst_psnr_std": b_std,
            "burst_ssim_mean": b_s,
            "burst_ssim_std": b_s_std
        }
        print(f"{cam_name:<24} | {n_sc:<6} | {in_p:5.2f} dB     | {s_p:5.2f} ± {s_std:4.2f} dB   | {b_p:5.2f} ± {b_std:4.2f} dB   | {b_s:.4f} ± {b_s_std:.4f}", flush=True)
    print("=" * 80, flush=True)

    # Save to JSON
    output_data = {
        "global_statistics": {
            "single_image": {"mean": m_s_p, "std": std_s_p, "ci95": list(ci_s_p), "ssim_mean": m_s_s, "ssim_std": std_s_s, "ssim_ci95": list(ci_s_s)},
            "two_frame_burst": {"mean": m_b_p, "std": std_b_p, "ci95": list(ci_b_p), "ssim_mean": m_b_s, "ssim_std": std_b_s, "ssim_ci95": list(ci_b_s)},
            "total_scenes": len(scene_results)
        },
        "iso_regimes": iso_stats,
        "camera_sensors": cam_stats,
        "per_scene_records": scene_results
    }
    out_file = "results/iso_stratification_and_stats.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[✓] Statistical analysis saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
