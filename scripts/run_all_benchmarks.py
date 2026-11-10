#!/usr/bin/env python3
"""
Master Benchmark Suite Runner: Executes all evaluation benchmarks, ablations,
visual demos, and diagnostic studies in a single consolidated command.

Executes:
1. SIDD Real Smartphone Paired Benchmark (Clean GT vs Noisy)
2. BSD68 Natural Images Benchmark (All 68 images)
3. Urban100 Structural Architecture Benchmark (All 100 images)
4. 4-Pane Visual Side-by-Side Comparison Generator
5. 5-Point Controlled Ablation Suite (info.md Section 6)
6. Pseudo-Target Bias Compounding Study (Task 5.2)
"""

import os
import sys
import time
import json
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.evaluate import evaluate_benchmark
from scripts.run_ablations import run_full_ablation_matrix
from scripts.study_bias_compounding import run_bias_compounding_study
from scripts.demo_denoise import run_demo
from src.utils.device import get_default_device


def run_master_benchmark_suite(
    checkpoint: str = "checkpoints/pcssflow_v2_sidd_gpu.pt",
    device: str = "auto",
    sidd_dir: str = "data/SIDD/train/train",
    bsd68_dir: str = "data/BSD68",
    urban100_dir: str = "data/Urban100",
    output_dir: str = "results/master_benchmark",
    max_sidd_scenes: int = 40,
    noise_sigma: float = 0.08,
    num_flow_steps: int = 4,
    solver: str = "heun",
    use_tta: bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    dev = get_default_device(device)
    start_total_time = time.perf_counter()

    print("==========================================================================")
    print("       PC-SSFLOW v2 UNIFIED MASTER BENCHMARK & EVALUATION SUITE           ")
    print(f"       Device: {dev} | Checkpoint: {checkpoint} | Steps: {num_flow_steps} | Solver: {solver.upper()} | TTA: {use_tta}")
    print("==========================================================================")

    master_results = {}

    # --- 1. SIDD Real Smartphone Paired Benchmark ---
    print("\n" + "=" * 74)
    print("[1/6] RUNNING REAL SMARTPHONE BENCHMARK (SIDD Paired Scenes)")
    print("=" * 74)
    sidd_json = os.path.join(output_dir, "eval_sidd.json")
    evaluate_benchmark(
        benchmark="sidd_paired",
        data_dir=sidd_dir,
        checkpoint=checkpoint,
        input_mode="srgb",
        num_flow_steps=num_flow_steps,
        solver=solver,
        use_tta=use_tta,
        max_images=max_sidd_scenes,
        crop_size=128,
        device=device,
        output_json=sidd_json
    )
    if os.path.exists(sidd_json):
        with open(sidd_json) as f:
            master_results["sidd"] = json.load(f)

    # --- 2. BSD68 Natural Images Benchmark ---
    print("\n" + "=" * 74)
    print(f"[2/6] RUNNING BSD68 BENCHMARK (All 68 Images | Noise Sigma = {noise_sigma})")
    print("=" * 74)
    bsd68_json = os.path.join(output_dir, "eval_bsd68.json")
    evaluate_benchmark(
        benchmark="bsd68",
        data_dir=bsd68_dir,
        checkpoint=checkpoint,
        input_mode="srgb",
        noise_sigma=noise_sigma,
        num_flow_steps=num_flow_steps,
        solver=solver,
        use_tta=use_tta,
        max_images=68,
        crop_size=128,
        device=device,
        output_json=bsd68_json
    )
    if os.path.exists(bsd68_json):
        with open(bsd68_json) as f:
            master_results["bsd68"] = json.load(f)

    # --- 3. Urban100 Structural Architecture Benchmark ---
    print("\n" + "=" * 74)
    print(f"[3/6] RUNNING URBAN100 BENCHMARK (All 100 Images | Noise Sigma = {noise_sigma})")
    print("=" * 74)
    urban_json = os.path.join(output_dir, "eval_urban100.json")
    evaluate_benchmark(
        benchmark="urban100",
        data_dir=urban100_dir,
        checkpoint=checkpoint,
        input_mode="srgb",
        noise_sigma=noise_sigma,
        num_flow_steps=num_flow_steps,
        solver=solver,
        use_tta=use_tta,
        max_images=100,
        crop_size=128,
        device=device,
        output_json=urban_json
    )
    if os.path.exists(urban_json):
        with open(urban_json) as f:
            master_results["urban100"] = json.load(f)

    # --- 4. 4-Pane Visual Side-by-Side Demo ---
    print("\n" + "=" * 74)
    print("[4/6] GENERATING 4-PANE VISUAL DEMO COMPARISON GRID")
    print("=" * 74)
    demo_img_out = os.path.join(output_dir, "demo_visual_comparison.png")
    sample_img = os.path.join(bsd68_dir, "test001.png")
    run_demo(
        image_path=sample_img,
        output_path=demo_img_out,
        noise_sigma=noise_sigma,
        input_mode="srgb",
        checkpoint=checkpoint,
        device=device
    )
    master_results["demo_image"] = demo_img_out

    # --- 5. 5-Point Controlled Ablation Suite ---
    print("\n" + "=" * 74)
    print("[5/6] EXECUTING 5-POINT CONTROLLED ABLATION MATRIX")
    print("=" * 74)
    ablation_json = os.path.join(output_dir, "ablation_matrix_results.json")
    run_full_ablation_matrix(
        data_path=sample_img,
        device=device,
        output_json=ablation_json
    )
    if os.path.exists(ablation_json):
        with open(ablation_json) as f:
            master_results["ablations"] = json.load(f)

    # --- 6. Pseudo-Target Bias Compounding Study ---
    print("\n" + "=" * 74)
    print("[6/6] EXECUTING PSEUDO-TARGET BIAS COMPOUNDING STUDY (TASK 5.2)")
    print("=" * 74)
    bias_json = os.path.join(output_dir, "bias_compounding_study.json")
    run_bias_compounding_study(
        num_train_iters=40,
        device=device,
        output_json=bias_json
    )
    if os.path.exists(bias_json):
        with open(bias_json) as f:
            master_results["bias_study"] = json.load(f)

    total_elapsed = time.perf_counter() - start_total_time

    # --- Print Unified Final Summary Table ---
    print("\n" + "═" * 78)
    print("               UNIFIED MASTER BENCHMARK SCOREBOARD RESULTS            ")
    print("═" * 78)
    print(f"{'Benchmark / Dataset':<28} | {'Noisy PSNR':<11} | {'Denoised PSNR':<14} | {'Net PSNR Gain':<13} | {'Net SSIM Gain'}")
    print("─" * 78)

    for bench_key, title in [("sidd", "SIDD Smartphone Captures"), ("bsd68", "BSD68 (All 68 Images)"), ("urban100", "Urban100 (All 100 Images)")]:
        if bench_key in master_results:
            res = master_results[bench_key]
            m_noisy_p = res.get("mean_noisy_psnr_db", res.get("mean_noisy_psnr", 0.0))
            m_den_p = res.get("mean_denoised_psnr_db", res.get("mean_denoised_psnr", 0.0))
            m_delta_p = res.get("mean_psnr_improvement_db", res.get("net_psnr_improvement_db", 0.0))
            m_delta_s = res.get("mean_ssim_improvement", res.get("net_ssim_gain", 0.0))
            print(f"{title:<28} | {m_noisy_p:6.2f} dB   | {m_den_p:6.2f} dB      | {m_delta_p:+6.2f} dB      | {m_delta_s:+.4f}")

    print("═" * 78)
    print(f"Total Suite Execution Time: {total_elapsed:.2f} seconds")
    print(f"All reports, json summaries, and visual grids saved to directory: {output_dir}/")
    print("═" * 78)

    summary_file = os.path.join(output_dir, "master_benchmark_summary.json")
    with open(summary_file, "w") as f:
        json.dump(master_results, f, indent=2)
    print(f"[✓] Consolidated Master Report written to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PC-SSFlow v2 Master Benchmark Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pcssflow_v2_sidd_gpu.pt")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, mps, cuda, cpu)")
    parser.add_argument("--sidd-dir", type=str, default="data/SIDD/train/train")
    parser.add_argument("--bsd68-dir", type=str, default="data/BSD68")
    parser.add_argument("--urban100-dir", type=str, default="data/Urban100")
    parser.add_argument("--output-dir", type=str, default="results/master_benchmark")
    parser.add_argument("--max-sidd-scenes", type=int, default=40)
    parser.add_argument("--noise-sigma", type=float, default=0.08)
    parser.add_argument("--num-flow-steps", type=int, default=4)
    parser.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--use-tta", action="store_true", help="Enable 8-fold test-time augmentation")
    args = parser.parse_args()

    run_master_benchmark_suite(
        checkpoint=args.checkpoint,
        device=args.device,
        sidd_dir=args.sidd_dir,
        bsd68_dir=args.bsd68_dir,
        urban100_dir=args.urban100_dir,
        output_dir=args.output_dir,
        max_sidd_scenes=args.max_sidd_scenes,
        noise_sigma=args.noise_sigma,
        num_flow_steps=args.num_flow_steps,
        solver=args.solver,
        use_tta=args.use_tta
    )
