#!/usr/bin/env python3
"""
Test Dynamic Phase Refinement & Spectral Step Modulation on 40 SIDD benchmark scenes.
Baseline: 35.717 dB / 0.8749 SSIM (4-step Heun, pruned PC-Flow).
"""

import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.pc_ssflow import PCSSFlowV2
from src.wavelets.dtcwt import complex_to_mag_phase, mag_phase_to_complex
from src.utils.metrics import compute_image_metrics
from scripts.evaluate import load_sidd_paired_scenes
from src.utils.device import get_default_device

def main():
    dev = get_default_device("auto")
    print(f"Device: {dev}")

    ckpt_path = "checkpoints/pcssflow_v2_sidd_gpu.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/pcssflow_v2_sidd.pt"

    model = PCSSFlowV2(in_channels=1, hidden_dim=128, n_scales=2, physics_mode="srgb").to(dev)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()

    pairs = load_sidd_paired_scenes("data/SIDD/train/train", max_scenes=40, crop_size=128)
    print(f"Loaded {len(pairs)} SIDD scenes.")

    def run_dynamic_sample(
        noisy: torch.Tensor,
        num_steps: int = 4,
        dynamic_pc: bool = False,
        pc_blend_power: float = 1.0,
        gamma_low: float = 1.0,
        gamma_s0: float = 1.0,
        gamma_s1: float = 1.0,
    ) -> torch.Tensor:
        b, c, h, w = noisy.shape
        curr_x = noisy.clone()
        pc_0 = model.pc_calc(noisy)
        scale_map = model.rectified_flow.physics_scheduler(noisy)
        dt = 1.0 / num_steps

        gammas = [gamma_s0, gamma_s1]

        for step in range(num_steps):
            t_val = step * dt
            t_batch = torch.full((b,), t_val, device=dev, dtype=torch.float32)

            # Dynamic Phase Congruency Refinement
            if dynamic_pc and step > 0:
                pc_curr = model.pc_calc(curr_x)
                lambda_t = (t_val ** pc_blend_power)
                pc = (1.0 - lambda_t) * pc_0 + lambda_t * pc_curr
            else:
                pc = pc_0

            c_low, highs = model.rectified_flow.dtcwt_fwd(curr_x)
            curr_clow = c_low
            curr_mags, curr_phases = [], []
            for h_s in highs:
                m, p = complex_to_mag_phase(h_s)
                curr_mags.append(m)
                curr_phases.append(p)

            v_clow, v_mags, _ = model.rectified_flow.velocity_net(
                curr_clow, curr_mags, curr_phases, t_batch, pc, scale_map, ssm_feat=pc
            )

            # Predictor step
            pred_clow = curr_clow + dt * v_clow * gamma_low
            pred_highs = [
                mag_phase_to_complex(curr_mags[s] + dt * v_mags[s] * gammas[s], curr_phases[s])
                for s in range(model.rectified_flow.n_scales)
            ]
            pred_x = model.rectified_flow.dtcwt_inv(pred_clow, pred_highs, orig_size=(h, w))

            # Corrector step
            t_next = min(1.0, t_val + dt)
            t_next_batch = torch.full((b,), t_next, device=dev, dtype=torch.float32)

            if dynamic_pc:
                pc_pred = model.pc_calc(pred_x)
                lambda_next = (t_next ** pc_blend_power)
                pc_c = (1.0 - lambda_next) * pc_0 + lambda_next * pc_pred
            else:
                pc_c = pc_0

            c_low_next, highs_next = model.rectified_flow.dtcwt_fwd(pred_x)
            next_mags, next_phases = [], []
            for h_s in highs_next:
                m, p = complex_to_mag_phase(h_s)
                next_mags.append(m)
                next_phases.append(p)

            v_clow_next, v_mags_next, _ = model.rectified_flow.velocity_net(
                c_low_next, next_mags, next_phases, t_next_batch, pc_c, scale_map, ssm_feat=pc_c
            )

            # Heun 2nd-order trapezoidal integration
            final_clow = curr_clow + 0.5 * dt * (v_clow + v_clow_next) * gamma_low
            rec_highs = [
                mag_phase_to_complex(
                    curr_mags[s] + 0.5 * dt * (v_mags[s] + v_mags_next[s]) * gammas[s],
                    curr_phases[s]
                )
                for s in range(model.rectified_flow.n_scales)
            ]
            curr_x = model.rectified_flow.dtcwt_inv(final_clow, rec_highs, orig_size=(h, w))

        return model.rectified_flow.proximal_op(y_noisy=noisy, x_flow=curr_x)

    experiments = [
        ("Baseline (Static PC, Uniform Steps)", False, 1.0, 1.0, 1.0, 1.0),
        ("Dynamic PC (Linear Blend, power=1.0)", True, 1.0, 1.0, 1.0, 1.0),
        ("Dynamic PC (Conservative, power=1.5)", True, 1.5, 1.0, 1.0, 1.0),
        ("Dynamic PC (Aggressive, power=0.5)",   True, 0.5, 1.0, 1.0, 1.0),
        ("Spectral Sizing (low=0.9, s0=1.05, s1=0.95)", False, 1.0, 0.9, 1.05, 0.95),
        ("Combined: Dynamic PC + Spectral Sizing", True, 1.0, 0.95, 1.05, 0.98),
    ]

    print("\n--- EXPERIMENT RESULTS ON 40 SIDD SCENES (4-Step Heun) ---")
    for name, dyn_pc, blend_p, g_low, g_s0, g_s1 in experiments:
        psnr_list, ssim_list = [], []
        t0 = time.perf_counter()
        for _, noisy_t, gt_t in pairs:
            noisy = noisy_t.to(dev)
            clean = gt_t.to(dev)
            with torch.no_grad():
                out = run_dynamic_sample(
                    noisy,
                    num_steps=4,
                    dynamic_pc=dyn_pc,
                    pc_blend_power=blend_p,
                    gamma_low=g_low,
                    gamma_s0=g_s0,
                    gamma_s1=g_s1,
                )
                out = torch.clamp(out, 0.0, 1.0)
                m = compute_image_metrics(out, clean)
                psnr_list.append(m["psnr"])
                ssim_list.append(m["ssim"])
        elapsed = time.perf_counter() - t0
        mean_psnr = float(np.mean(psnr_list))
        mean_ssim = float(np.mean(ssim_list))
        delta_p = mean_psnr - 35.717
        delta_s = mean_ssim - 0.8749
        print(f"{name:45s} | PSNR: {mean_psnr:6.3f} dB ({delta_p:+6.3f}) | SSIM: {mean_ssim:6.4f} ({delta_s:+.4f}) | Time: {elapsed:5.1f}s")

if __name__ == "__main__":
    main()
