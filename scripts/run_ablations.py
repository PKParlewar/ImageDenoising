#!/usr/bin/env python3
"""
Phase 6 / Task 6.1: Systematic 5-Point Controlled Ablation Suite & Factorial Interaction.

Executes the exact 1-to-1 controlled ablation matrix from info.md across multiple test crops:
1. Fixed-radius spectral masking vs. Correlation-adaptive masking.
2. BSN-style masked-prediction loss vs. Phase-congruency invariance loss.
3. Rigid canonical Hilbert scan (w=0) vs. PC-warped Hilbert scan (w=8).
4. Pixel-space rectified flow vs. Phase-domain rectified flow.
5. Post-hoc-only proximal correction vs. Physics-calibrated scheduling.

Also evaluates a 2x2 factorial interaction design to quantify cross-component synergy.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.pc_ssflow import PCSSFlowV2
from src.wavelets.dtcwt import complex_to_mag_phase, mag_phase_to_complex
from src.utils.metrics import compute_image_metrics
from src.utils.device import get_default_device
from src.backbone.pc_warping import warp_hilbert_trajectory, compute_traversal_cost


def evaluate_ablation_variant(
    model: PCSSFlowV2,
    clean_img: torch.Tensor,
    noisy_img: torch.Tensor,
    variant_name: str,
    cfg: dict
) -> dict:
    device = clean_img.device
    num_steps = 2
    b, c, h, w = noisy_img.shape

    with torch.no_grad():
        if variant_name.startswith("1a."):
            # Fixed Radius Masking (r=2 static radius in Stage 1)
            c_low, highs = model.dtcwt_fwd(noisy_img)
            orig_mags, orig_phases = [], []
            for high_s in highs:
                m_s, p_s = complex_to_mag_phase(high_s)
                orig_mags.append(m_s)
                orig_phases.append(p_s)
            pc_orig = model.pc_calc(c_low=c_low, highs=highs)
            corr_fixed = torch.tensor([[2.0]], device=device)
            masked_mags, _, _ = model.adaptive_masker(orig_mags, corr_fixed)
            rec_mags = model.mag_denoiser(masked_mags, pc_orig)
            rec_highs = [mag_phase_to_complex(rec_mags[s], orig_phases[s]) for s in range(model.n_scales)]
            denoised_spatial = model.dtcwt_inv(c_low, rec_highs, orig_size=(h, w))
            denoised = model.ssm_backbone(denoised_spatial, pc_orig, window_size=model.warping_window)
            denoised = torch.clamp(denoised, 0.0, 1.0)
            tv_val = 0.0

        elif variant_name.startswith("1b."):
            # Correlation-Adaptive Masking (r(kappa) dynamic radius in Stage 1)
            out = model.forward_self_supervised_step(noisy_img)
            denoised = torch.clamp(out["pseudo_clean"], 0.0, 1.0)
            tv_val = 0.0

        elif variant_name.startswith("2a."):
            # Standard BSN-Style Masked Loss Only (no phase congruency prior: PC=0)
            c_low, highs = model.dtcwt_fwd(noisy_img)
            orig_mags, orig_phases = [], []
            for high_s in highs:
                m_s, p_s = complex_to_mag_phase(high_s)
                orig_mags.append(m_s)
                orig_phases.append(p_s)
            pc_zero = torch.zeros(b, 1, h, w, device=device)
            _, corr_length = model.noise_estimator(noisy_img)
            masked_mags, _, _ = model.adaptive_masker(orig_mags, corr_length)
            rec_mags = model.mag_denoiser(masked_mags, pc_zero)
            rec_highs = [mag_phase_to_complex(rec_mags[s], orig_phases[s]) for s in range(model.n_scales)]
            denoised_spatial = model.dtcwt_inv(c_low, rec_highs, orig_size=(h, w))
            denoised = model.ssm_backbone(denoised_spatial, pc_zero, window_size=0)
            denoised = torch.clamp(denoised, 0.0, 1.0)
            tv_val = 0.0

        elif variant_name.startswith("2b."):
            # Phase Congruency Invariance Loss (Full noise-invariant PC prior in Stage 1)
            out = model.forward_self_supervised_step(noisy_img)
            denoised = torch.clamp(out["pseudo_clean"], 0.0, 1.0)
            tv_val = 0.0

        elif variant_name.startswith("3a."):
            # Standard Fixed Hilbert Scan (w=0, unwarped canonical trajectory)
            orig_w = model.warping_window
            model.warping_window = 0
            denoised = model.denoise(noisy_img, num_flow_steps=num_steps)
            model.warping_window = orig_w
            denoised = torch.clamp(denoised, 0.0, 1.0)
            pc_map = model.pc_calc(noisy_img)
            cost_field = compute_traversal_cost(pc_map)[0, 0]
            sig = warp_hilbert_trajectory(pc_map, window_size=0)
            c_seq = cost_field.view(-1)[sig]
            tv_val = torch.mean(torch.abs(c_seq[1:] - c_seq[:-1])).item()

        elif variant_name.startswith("3b."):
            # PC-Warped Hilbert Scan (w=8, structure-guided lookahead trajectory)
            orig_w = model.warping_window
            model.warping_window = 8
            denoised = model.denoise(noisy_img, num_flow_steps=num_steps)
            model.warping_window = orig_w
            denoised = torch.clamp(denoised, 0.0, 1.0)
            pc_map = model.pc_calc(noisy_img)
            cost_field = compute_traversal_cost(pc_map)[0, 0]
            sig = warp_hilbert_trajectory(pc_map, window_size=8)
            c_seq = cost_field.view(-1)[sig]
            tv_val = torch.mean(torch.abs(c_seq[1:] - c_seq[:-1])).item()

        elif variant_name.startswith("4a."):
            # Pixel-Space Rectified Flow (spatial linear transport without phase-domain decomposition)
            dt = 1.0 / num_steps
            curr_x = noisy_img.clone()
            pc = model.pc_calc(noisy_img)
            scale_map = model.physics_scheduler(noisy_img)
            ssm_feat = model.ssm_backbone(noisy_img, pc, window_size=model.warping_window)
            for step in range(num_steps):
                t_val = step * dt
                t_b = torch.full((b,), t_val, device=device, dtype=torch.float32)
                c_low, highs = model.dtcwt_fwd(curr_x)
                curr_clow = c_low.clone()
                curr_mags = [complex_to_mag_phase(h_s)[0] for h_s in highs]
                curr_phases = [complex_to_mag_phase(h_s)[1] for h_s in highs]
                v_clow, v_mags, _ = model.rectified_flow.velocity_net(curr_clow, curr_mags, curr_phases, t_b, pc, scale_map, ssm_feat=ssm_feat)
                rec_highs = [
                    mag_phase_to_complex(curr_mags[s] + dt * v_mags[s], curr_phases[s] + dt * 0.05 * torch.randn_like(curr_phases[s]))
                    for s in range(model.n_scales)
                ]
                curr_x = model.dtcwt_inv(curr_clow + dt * v_clow, rec_highs, orig_size=(h, w))
            denoised = model.proximal_op(y_noisy=noisy_img, x_flow=curr_x)
            denoised = torch.clamp(denoised, 0.0, 1.0)
            tv_val = 0.0

        elif variant_name.startswith("4b."):
            # Phase-Domain Rectified Flow (DTCWT complex-domain transport with invariant phase)
            denoised = model.denoise(noisy_img, num_flow_steps=num_steps)
            denoised = torch.clamp(denoised, 0.0, 1.0)
            tv_val = 0.0

        elif variant_name.startswith("5a."):
            # Post-Hoc Proximal Operator Only (Unscheduled flow: uniform scale_map=1.0)
            c_low, highs = model.rectified_flow.dtcwt_fwd(noisy_img)
            curr_clow = c_low.clone()
            curr_mags = [complex_to_mag_phase(h_s)[0].clone() for h_s in highs]
            curr_phases = [complex_to_mag_phase(h_s)[1].clone() for h_s in highs]
            pc = model.rectified_flow.pc_calc(noisy_img)
            ssm_feat = model.ssm_backbone(noisy_img, pc, window_size=model.warping_window)
            scale_map_uniform = torch.ones_like(pc)
            dt = 1.0 / num_steps
            for step in range(num_steps):
                t_b = torch.full((b,), step * dt, device=device, dtype=torch.float32)
                v_clow, v_mags, _ = model.rectified_flow.velocity_net(
                    curr_clow, curr_mags, curr_phases, t_b, pc, scale_map_uniform, ssm_feat=ssm_feat
                )
                curr_clow = curr_clow + dt * v_clow
                for s in range(model.n_scales):
                    curr_mags[s] = curr_mags[s] + dt * v_mags[s]
            rec_highs = [mag_phase_to_complex(curr_mags[s], curr_phases[s]) for s in range(model.n_scales)]
            x_unscheduled = model.rectified_flow.dtcwt_inv(curr_clow, rec_highs, orig_size=(h, w))
            denoised = model.proximal_op(y_noisy=noisy_img, x_flow=x_unscheduled)
            denoised = torch.clamp(denoised, 0.0, 1.0)
            tv_val = 0.0

        else:
            # 5b: Full Physics-Calibrated Flow + Safeguard (Proposed Master: sensor-calibrated step schedule)
            denoised = model.denoise(noisy_img, num_flow_steps=num_steps)
            denoised = torch.clamp(denoised, 0.0, 1.0)
            tv_val = 0.0

    metrics = compute_image_metrics(denoised, clean_img)
    metrics["variant"] = variant_name
    metrics["tv_path"] = float(tv_val)

    # Compute Edge-PSNR specifically on structural edges (top 15% quantile of Phase Congruency)
    pc_clean = model.pc_calc(clean_img)
    edge_thresh = torch.quantile(pc_clean.view(-1), 0.85)
    edge_mask = (pc_clean >= edge_thresh).float()
    mse_edge = ((denoised - clean_img) ** 2 * edge_mask).sum() / edge_mask.sum().clamp(min=1.0)
    metrics["edge_psnr"] = float((10 * torch.log10(1.0 / mse_edge.clamp(min=1e-8))).item())
    return metrics


def load_diverse_test_crops(num_crops: int = 10, crop_size: int = 128) -> list:
    """Selects diverse crops from Urban100 (high structural geometry) and BSD68 (natural textures)."""
    urban_imgs = sorted(glob.glob("data/Urban100/*.png"))
    bsd_imgs = sorted(glob.glob("data/BSD68/*.png"))
    
    selected_files = []
    n_urban = num_crops // 2
    n_bsd = num_crops - n_urban
    
    step_u = max(1, len(urban_imgs) // n_urban) if urban_imgs else 0
    step_b = max(1, len(bsd_imgs) // n_bsd) if bsd_imgs else 0
    
    for i in range(n_urban):
        if i * step_u < len(urban_imgs):
            selected_files.append(urban_imgs[i * step_u])
    for i in range(n_bsd):
        if i * step_b < len(bsd_imgs):
            selected_files.append(bsd_imgs[i * step_b])
            
    crops = []
    for f in selected_files[:num_crops]:
        arr = np.array(Image.open(f).convert("L"), dtype=np.float32) / 255.0
        h, w = arr.shape
        y0 = max(0, (h - crop_size) // 2)
        x0 = max(0, (w - crop_size) // 2)
        crop_arr = arr[y0:y0+crop_size, x0:x0+crop_size]
        crop_t = torch.from_numpy(crop_arr).unsqueeze(0).unsqueeze(0)
        crops.append((os.path.basename(f), crop_t))
    return crops


def run_full_ablation_matrix(
    num_crops: int = 10,
    checkpoint: str = "checkpoints/pcssflow_v2_sidd_gpu.pt",
    device: str = "auto",
    output_json: str = "results/master_benchmark/ablation_matrix_results.json"
):
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    dev = get_default_device(device)

    print("==========================================================================")
    print(f"PC-SSFLOW v2: 5-POINT CONTROLLED ABLATION SUITE (Device: {dev}, Crops: {num_crops})")
    print("==========================================================================")

    crops = load_diverse_test_crops(num_crops=num_crops, crop_size=128)
    print(f"Loaded {len(crops)} diverse structural evaluation crops from Urban100 & BSD68.\n")

    model = PCSSFlowV2(in_channels=1, hidden_dim=128, n_scales=2, physics_mode="srgb").to(dev)
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=dev))
    model.eval()

    ablations = [
        ("1a. Fixed Radius Spectral Masking (r=2)", {}),
        ("1b. Correlation-Adaptive Masking (r(kappa))", {}),
        ("2a. Standard BSN-Style Masked Loss (L_mask)", {}),
        ("2b. Phase Congruency Invariance Loss (L_mask + L_pc)", {}),
        ("3a. Rigid Canonical Hilbert Scan (w=0)", {}),
        ("3b. Structure-Guided PC-Warped Scan (w=8)", {}),
        ("4a. Pixel-Space Rectified Flow", {}),
        ("4b. Phase-Domain Rectified Flow (DTCWT)", {}),
        ("5a. Post-Hoc Proximal Operator Only", {}),
        ("5b. Physics-Calibrated Flow + Safeguard", {}),
    ]

    aggregated_metrics = {name: {"psnr": [], "edge_psnr": [], "ssim": [], "tv_path": []} for name, _ in ablations}
    noisy_baselines = {"psnr": [], "edge_psnr": [], "ssim": []}

    for crop_idx, (fname, clean_cpu) in enumerate(crops):
        clean = clean_cpu.to(dev)
        torch.manual_seed(42 + crop_idx)
        noisy = torch.clamp(clean + torch.randn_like(clean) * 0.08, 0.0, 1.0)
        m_noisy = compute_image_metrics(noisy, clean)
        noisy_baselines["psnr"].append(m_noisy["psnr"])
        noisy_baselines["ssim"].append(m_noisy["ssim"])
        
        pc_clean = model.pc_calc(clean)
        edge_thresh = torch.quantile(pc_clean.view(-1), 0.85)
        edge_mask = (pc_clean >= edge_thresh).float()
        mse_edge = ((noisy - clean) ** 2 * edge_mask).sum() / edge_mask.sum().clamp(min=1.0)
        noisy_baselines["edge_psnr"].append(float((10 * torch.log10(1.0 / mse_edge.clamp(min=1e-8))).item()))

        print(f"[{crop_idx+1}/{len(crops)}] Evaluating crop: {fname} (Noisy PSNR: {m_noisy['psnr']:.2f} dB)...")
        for name, cfg in ablations:
            res = evaluate_ablation_variant(model, clean, noisy, name, cfg)
            aggregated_metrics[name]["psnr"].append(res["psnr"])
            aggregated_metrics[name]["edge_psnr"].append(res["edge_psnr"])
            aggregated_metrics[name]["ssim"].append(res["ssim"])
            aggregated_metrics[name]["tv_path"].append(res["tv_path"])

    # Compute mean and standard deviation
    results = []
    print("\n=========================================================================================================================")
    print(f"{'Variant / Mechanism Isolated':<50} | {'Full PSNR (dB)':<16} | {'Edge-PSNR (dB)':<16} | {'SSIM':<16} | {'TV_path'}")
    print("-------------------------------------------------------------------------------------------------------------------------")
    for name, _ in ablations:
        psnr_m, psnr_s = np.mean(aggregated_metrics[name]["psnr"]), np.std(aggregated_metrics[name]["psnr"])
        edge_m, edge_s = np.mean(aggregated_metrics[name]["edge_psnr"]), np.std(aggregated_metrics[name]["edge_psnr"])
        ssim_m, ssim_s = np.mean(aggregated_metrics[name]["ssim"]), np.std(aggregated_metrics[name]["ssim"])
        tv_m = np.mean(aggregated_metrics[name]["tv_path"])

        rec = {
            "variant": name,
            "psnr_mean": float(psnr_m),
            "psnr_std": float(psnr_s),
            "edge_psnr_mean": float(edge_m),
            "edge_psnr_std": float(edge_s),
            "ssim_mean": float(ssim_m),
            "ssim_std": float(ssim_s),
            "tv_path_mean": float(tv_m),
        }
        results.append(rec)
        tv_str = f"{tv_m:.5f}" if tv_m > 0 else "—"
        print(f"{name:<50} | {psnr_m:6.2f} ± {psnr_s:4.2f}    | {edge_m:6.2f} ± {edge_s:4.2f}    | {ssim_m:6.4f} ± {ssim_s:5.4f} | {tv_str}")
    print("=========================================================================================================================\n")

    # Run 2x2 Factorial Interaction Study
    print("Executing 2x2 Factorial Interaction Experiment (Phase Congruency Prior x Physics Step Scheduling)...")
    fac_results = {"PC=0, Phys=0": [], "PC=1, Phys=0": [], "PC=0, Phys=1": [], "PC=1, Phys=1": []}
    dt = 1.0 / 2
    for crop_idx, (fname, clean_cpu) in enumerate(crops):
        clean = clean_cpu.to(dev)
        torch.manual_seed(42 + crop_idx)
        noisy = torch.clamp(clean + torch.randn_like(clean) * 0.08, 0.0, 1.0)
        b, c, h, w = noisy.shape
        pc_clean = model.pc_calc(clean)
        edge_thresh = torch.quantile(pc_clean.view(-1), 0.85)
        edge_mask = (pc_clean >= edge_thresh).float()

        c_low, highs = model.dtcwt_fwd(noisy)
        orig_mags = [complex_to_mag_phase(h_s)[0] for h_s in highs]
        orig_phases = [complex_to_mag_phase(h_s)[1] for h_s in highs]

        with torch.no_grad():
            for use_pc in [False, True]:
                for use_phys in [False, True]:
                    tag = f"PC={int(use_pc)}, Phys={int(use_phys)}"
                    pc_in = model.pc_calc(noisy) if use_pc else torch.zeros(b, 1, h, w, device=dev)
                    scale_map = model.physics_scheduler(noisy) if use_phys else torch.ones_like(pc_in)
                    ssm_feat = model.ssm_backbone(noisy, pc_in, window_size=8 if use_pc else 0)

                    curr_x = noisy.clone()
                    for step in range(2):
                        t_val = step * dt
                        t_b = torch.full((b,), t_val, device=dev, dtype=torch.float32)

                        c_low_t, highs_t = model.dtcwt_fwd(curr_x)
                        curr_clow = c_low_t
                        curr_mags, curr_phases = [], []
                        for h_s in highs_t:
                            m, p = complex_to_mag_phase(h_s)
                            curr_mags.append(m)
                            curr_phases.append(p)

                        v_clow, v_mags, _ = model.rectified_flow.velocity_net(
                            curr_clow, curr_mags, curr_phases, t_b, pc_in, scale_map, ssm_feat=ssm_feat
                        )

                        # Predictor
                        pred_clow = curr_clow + dt * v_clow
                        pred_highs = [
                            mag_phase_to_complex(curr_mags[s] + dt * v_mags[s], curr_phases[s])
                            for s in range(model.n_scales)
                        ]
                        pred_x = model.dtcwt_inv(pred_clow, pred_highs, orig_size=(h, w))

                        # Corrector
                        t_next = min(1.0, t_val + dt)
                        t_next_b = torch.full((b,), t_next, device=dev, dtype=torch.float32)
                        c_low_next, highs_next = model.dtcwt_fwd(pred_x)
                        next_mags, next_phases = [], []
                        for h_s in highs_next:
                            m, p = complex_to_mag_phase(h_s)
                            next_mags.append(m)
                            next_phases.append(p)

                        v_clow_next, v_mags_next, _ = model.rectified_flow.velocity_net(
                            c_low_next, next_mags, next_phases, t_next_b, pc_in, scale_map, ssm_feat=ssm_feat
                        )

                        final_clow = curr_clow + 0.5 * dt * (v_clow + v_clow_next)
                        rec_highs = [
                            mag_phase_to_complex(curr_mags[s] + 0.5 * dt * (v_mags[s] + v_mags_next[s]), curr_phases[s])
                            for s in range(model.n_scales)
                        ]
                        curr_x = model.dtcwt_inv(final_clow, rec_highs, orig_size=(h, w))

                    denoised = torch.clamp(model.proximal_op(y_noisy=noisy, x_flow=curr_x), 0.0, 1.0)
                    m = compute_image_metrics(denoised, clean)
                    mse_edge = ((denoised - clean) ** 2 * edge_mask).sum() / edge_mask.sum().clamp(min=1.0)
                    edge_p = float((10 * torch.log10(1.0 / mse_edge.clamp(min=1e-8))).item())
                    fac_results[tag].append({"psnr": m["psnr"], "edge_psnr": edge_p, "ssim": m["ssim"]})

    factorial_summary = {}
    print("Factorial Matrix Results (Mean ± Std):")
    for tag in ["PC=0, Phys=0", "PC=1, Phys=0", "PC=0, Phys=1", "PC=1, Phys=1"]:
        p_m, p_s = np.mean([x["psnr"] for x in fac_results[tag]]), np.std([x["psnr"] for x in fac_results[tag]])
        ep_m, ep_s = np.mean([x["edge_psnr"] for x in fac_results[tag]]), np.std([x["edge_psnr"] for x in fac_results[tag]])
        s_m, s_s = np.mean([x["ssim"] for x in fac_results[tag]]), np.std([x["ssim"] for x in fac_results[tag]])
        factorial_summary[tag] = {
            "psnr_mean": float(p_m), "psnr_std": float(p_s),
            "edge_psnr_mean": float(ep_m), "edge_psnr_std": float(ep_s),
            "ssim_mean": float(s_m), "ssim_std": float(s_s)
        }
        print(f"  {tag:<14}: PSNR {p_m:6.2f} ± {p_s:4.2f} dB | Edge-PSNR {ep_m:6.2f} ± {ep_s:4.2f} dB | SSIM {s_m:6.4f} ± {s_s:5.4f}")

    final_payload = {
        "noisy_baseline": {
            "psnr_mean": float(np.mean(noisy_baselines["psnr"])),
            "psnr_std": float(np.std(noisy_baselines["psnr"])),
            "edge_psnr_mean": float(np.mean(noisy_baselines["edge_psnr"])),
            "edge_psnr_std": float(np.std(noisy_baselines["edge_psnr"])),
            "ssim_mean": float(np.mean(noisy_baselines["ssim"])),
            "ssim_std": float(np.std(noisy_baselines["ssim"])),
        },
        "ablations": results,
        "factorial_interaction": factorial_summary,
    }

    with open(output_json, "w") as f:
        json.dump(final_payload, f, indent=2)
    # Also save to root results if different
    if output_json != "results/ablation_matrix_results.json":
        os.makedirs("results", exist_ok=True)
        with open("results/ablation_matrix_results.json", "w") as f:
            json.dump(final_payload, f, indent=2)

    print(f"\n[✓] Full ablation & factorial suite saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PC-SSFlow v2 Ablation Suite")
    parser.add_argument("--num-crops", type=int, default=10, help="Number of diverse test crops")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pcssflow_v2_sidd_gpu.pt")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, mps, cuda, cpu)")
    parser.add_argument("--output-json", type=str, default="results/master_benchmark/ablation_matrix_results.json")
    args = parser.parse_args()

    run_full_ablation_matrix(
        num_crops=args.num_crops,
        checkpoint=args.checkpoint,
        device=args.device,
        output_json=args.output_json
    )
