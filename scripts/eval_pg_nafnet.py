#!/usr/bin/env python3
"""
Comprehensive Evaluation of Phase-Gated NAFNet (PG-NAFNet):
1. Evaluates on 40 held-out validation scenes (center crop 256x256).
2. Evaluates on full 1,280 official SIDD benchmark blocks (40 scenes x 32 blocks).
3. Evaluates both Single-Image Direct and Two-Frame Burst inference.
4. Generates side-by-side comparative scorecard vs Model A (No-PC), Model B (Sobel), Model C (Flagship Naive-PC).
5. Exports results to results/pg_nafnet_summary.json.
"""

import os
import sys
import json
import math
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.pg_nafnet import PG_NAFNet
from src.wavelets.phase_congruency import PhaseCongruencyCalculator
from src.utils.metrics import compute_image_metrics
from src.utils.device import get_default_device
from scripts.eval_sidd_1280_blocks import extract_32_blocks, compute_ci95


def eval_pg_nafnet_40_scenes(model: PG_NAFNet, pc_calc: PhaseCongruencyCalculator, dev: torch.device, data_dir="data/SIDD/train/train"):
    """Evaluates on 40 held-out validation scenes (center crops)."""
    val_scenes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])[:40]
    psnr_single, ssim_single = [], []
    psnr_burst, ssim_burst = [], []

    t0 = time.perf_counter()
    with torch.no_grad():
        for s in val_scenes:
            s_path = os.path.join(data_dir, s)
            f_noisy = sorted([f for f in os.listdir(s_path) if "NOISY" in f and f.endswith(".PNG")])
            f_gt = sorted([f for f in os.listdir(s_path) if "GT" in f and f.endswith(".PNG")])
            if len(f_noisy) < 2 or not f_gt:
                continue

            im1 = Image.open(os.path.join(s_path, f_noisy[0])).convert("L")
            im2 = Image.open(os.path.join(s_path, f_noisy[1])).convert("L")
            imgt = Image.open(os.path.join(s_path, f_gt[0])).convert("L")
            w, h = im1.size
            cx, cy = w // 2, h // 2
            c1 = torch.from_numpy(np.array(im1.crop((cx-128, cy-128, cx+128, cy+128)), dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
            c2 = torch.from_numpy(np.array(im2.crop((cx-128, cy-128, cx+128, cy+128)), dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
            cg = torch.from_numpy(np.array(imgt.crop((cx-128, cy-128, cx+128, cy+128)), dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)

            pc1 = pc_calc(c1)
            pc2 = pc_calc(c2)

            # Single image
            p1 = model(c1, pc1)
            m1 = compute_image_metrics(p1, cg)
            psnr_single.append(m1["psnr"])
            ssim_single.append(m1["ssim"])

            # Two-frame burst
            p2 = model(c2, pc2)
            p_burst = (p1 + p2) / 2.0
            mb = compute_image_metrics(p_burst, cg)
            psnr_burst.append(mb["psnr"])
            ssim_burst.append(mb["ssim"])

    dt = time.perf_counter() - t0
    m_p1, s_p1, ci_p1 = compute_ci95(psnr_single)
    m_s1, s_s1, ci_s1 = compute_ci95(ssim_single)
    m_pb, s_pb, ci_pb = compute_ci95(psnr_burst)
    m_sb, s_sb, ci_sb = compute_ci95(ssim_burst)

    return {
        "single_image": {
            "psnr_mean": m_p1, "psnr_std": s_p1, "psnr_ci95": list(ci_p1),
            "ssim_mean": m_s1, "ssim_std": s_s1, "ssim_ci95": list(ci_s1),
        },
        "two_frame_burst": {
            "psnr_mean": m_pb, "psnr_std": s_pb, "psnr_ci95": list(ci_pb),
            "ssim_mean": m_sb, "ssim_std": s_sb, "ssim_ci95": list(ci_sb),
        },
        "time_s": dt
    }


def eval_pg_nafnet_1280_blocks(model: PG_NAFNet, pc_calc: PhaseCongruencyCalculator, dev: torch.device, data_dir="data/SIDD/train/train"):
    """Evaluates on full official 1,280 SIDD benchmark blocks (40 scenes x 32 blocks)."""
    val_scenes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])[:40]
    psnr_single, ssim_single = [], []
    psnr_burst, ssim_burst = [], []

    t0 = time.perf_counter()
    with torch.no_grad():
        for s in val_scenes:
            s_path = os.path.join(data_dir, s)
            f_noisy = sorted([f for f in os.listdir(s_path) if "NOISY" in f and f.endswith(".PNG")])
            f_gt = sorted([f for f in os.listdir(s_path) if "GT" in f and f.endswith(".PNG")])
            if len(f_noisy) < 2 or not f_gt:
                continue

            im1_full = Image.open(os.path.join(s_path, f_noisy[0])).convert("L")
            im2_full = Image.open(os.path.join(s_path, f_noisy[1])).convert("L")
            imgt_full = Image.open(os.path.join(s_path, f_gt[0])).convert("L")

            b1_list = extract_32_blocks(im1_full, 256)
            b2_list = extract_32_blocks(im2_full, 256)
            bgt_list = extract_32_blocks(imgt_full, 256)

            for b_idx in range(len(b1_list)):
                c1 = torch.from_numpy(np.array(b1_list[b_idx], dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
                c2 = torch.from_numpy(np.array(b2_list[b_idx], dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
                cg = torch.from_numpy(np.array(bgt_list[b_idx], dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)

                pc1 = pc_calc(c1)
                pc2 = pc_calc(c2)

                p1 = model(c1, pc1)
                m1 = compute_image_metrics(p1, cg)
                psnr_single.append(m1["psnr"])
                ssim_single.append(m1["ssim"])

                p2 = model(c2, pc2)
                p_burst = (p1 + p2) / 2.0
                mb = compute_image_metrics(p_burst, cg)
                psnr_burst.append(mb["psnr"])
                ssim_burst.append(mb["ssim"])

    dt = time.perf_counter() - t0
    m_p1, s_p1, ci_p1 = compute_ci95(psnr_single)
    m_s1, s_s1, ci_s1 = compute_ci95(ssim_single)
    m_pb, s_pb, ci_pb = compute_ci95(psnr_burst)
    m_sb, s_sb, ci_sb = compute_ci95(ssim_burst)

    return {
        "single_image": {
            "psnr_mean": m_p1, "psnr_std": s_p1, "psnr_ci95": list(ci_p1),
            "ssim_mean": m_s1, "ssim_std": s_s1, "ssim_ci95": list(ci_s1),
        },
        "two_frame_burst": {
            "psnr_mean": m_pb, "psnr_std": s_pb, "psnr_ci95": list(ci_pb),
            "ssim_mean": m_sb, "ssim_std": s_sb, "ssim_ci95": list(ci_sb),
        },
        "total_blocks": len(psnr_single),
        "time_s": dt
    }


def main():
    dev = get_default_device("auto")
    ckpt_path = "checkpoints/pg_nafnet_sidd.pt"

    print("=" * 80, flush=True)
    print("COMPREHENSIVE EVALUATION: PHASE-GATED NAFNET (PG-NAFNet)")
    print(f"Device: {dev} | Checkpoint: {ckpt_path}", flush=True)
    print("=" * 80, flush=True)

    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found! Please run train_pg_nafnet.py first.", flush=True)
        sys.exit(1)

    model = PG_NAFNet(in_ch=1, c=72, n_blocks=12).to(dev)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()

    pc_calc = PhaseCongruencyCalculator(n_scales=2).to(dev)

    # 1. 40-scene evaluation
    print("\n[1/2] Evaluating on 40 held-out validation scenes (center crops)...", flush=True)
    res_40 = eval_pg_nafnet_40_scenes(model, pc_calc, dev)
    print(f"  Single-Image: {res_40['single_image']['psnr_mean']:.2f} dB | {res_40['single_image']['ssim_mean']:.4f} SSIM", flush=True)
    print(f"  Burst (2F):   {res_40['two_frame_burst']['psnr_mean']:.2f} dB | {res_40['two_frame_burst']['ssim_mean']:.4f} SSIM", flush=True)

    # 2. 1,280-block evaluation
    print("\n[2/2] Evaluating on full 1,280 official SIDD benchmark blocks...", flush=True)
    res_1280 = eval_pg_nafnet_1280_blocks(model, pc_calc, dev)
    print(f"  Single-Image: {res_1280['single_image']['psnr_mean']:.2f} dB | {res_1280['single_image']['ssim_mean']:.4f} SSIM", flush=True)
    print(f"  Burst (2F):   {res_1280['two_frame_burst']['psnr_mean']:.2f} dB | {res_1280['two_frame_burst']['ssim_mean']:.4f} SSIM", flush=True)

    # 3. Load baseline ablations for comparison
    with open("results/ablation_1280_blocks_summary.json", "r") as f:
        prev_ablations = json.load(f)

    # Compile unified comparison
    summary = {
        "pg_nafnet_40_scenes": res_40,
        "pg_nafnet_1280_blocks": res_1280,
        "comparative_1280_blocks": {
            "Model A (No-PC Baseline, in_ch=1)": {
                "psnr": prev_ablations["model_a_no_pc"]["psnr_mean"],
                "ssim": prev_ablations["model_a_no_pc"]["ssim_mean"],
            },
            "Model B (Sobel-Guided, in_ch=2)": {
                "psnr": prev_ablations["model_b_sobel"]["psnr_mean"],
                "ssim": prev_ablations["model_b_sobel"]["ssim_mean"],
            },
            "Model C (Naive-Input PC, in_ch=2)": {
                "psnr": prev_ablations["model_c_pc_net"]["psnr_mean"],
                "ssim": prev_ablations["model_c_pc_net"]["ssim_mean"],
            },
            "Model D (Phase-Gated PG-NAFNet, in_ch=1)": {
                "psnr": res_1280["single_image"]["psnr_mean"],
                "ssim": res_1280["single_image"]["ssim_mean"],
                "psnr_burst": res_1280["two_frame_burst"]["psnr_mean"],
                "ssim_burst": res_1280["two_frame_burst"]["ssim_mean"],
            }
        }
    }

    os.makedirs("results", exist_ok=True)
    out_json = "results/pg_nafnet_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print("UNIFIED 1,280-BLOCK ABLATION & ARCHITECTURAL SCORECARD")
    print("=" * 80, flush=True)
    print(f"{'Model Architecture':<40} | {'Input':<8} | {'Single PSNR':<12} | {'Single SSIM':<12}")
    print("-" * 80)
    print(f"{'Model A: No-PC Baseline':<40} | {'1-ch':<8} | {summary['comparative_1280_blocks']['Model A (No-PC Baseline, in_ch=1)']['psnr']:.2f} dB     | {summary['comparative_1280_blocks']['Model A (No-PC Baseline, in_ch=1)']['ssim']:.4f}")
    print(f"{'Model B: Sobel-Guided':<40} | {'2-ch':<8} | {summary['comparative_1280_blocks']['Model B (Sobel-Guided, in_ch=2)']['psnr']:.2f} dB     | {summary['comparative_1280_blocks']['Model B (Sobel-Guided, in_ch=2)']['ssim']:.4f}")
    print(f"{'Model C: Naive-Input PC (Flagship)':<40} | {'2-ch':<8} | {summary['comparative_1280_blocks']['Model C (Naive-Input PC, in_ch=2)']['psnr']:.2f} dB     | {summary['comparative_1280_blocks']['Model C (Naive-Input PC, in_ch=2)']['ssim']:.4f}")
    print(f"{'Model D: Phase-Gated PG-NAFNet (Ours)':<40} | {'1-ch':<8} | {res_1280['single_image']['psnr_mean']:.2f} dB     | {res_1280['single_image']['ssim_mean']:.4f}")
    print(f"{'  └─ Two-Frame Burst Mode':<40} | {'1-ch':<8} | {res_1280['two_frame_burst']['psnr_mean']:.2f} dB     | {res_1280['two_frame_burst']['ssim_mean']:.4f}")
    print("=" * 80, flush=True)
    print(f"[✓] Evaluation summary saved to {out_json}!", flush=True)


if __name__ == "__main__":
    main()
