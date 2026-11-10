#!/usr/bin/env python3
"""
Grid sweep of Spectral Step Modulation on 40 SIDD scenes.
Explores:
gamma_low: [0.80, 0.85, 0.90, 0.95, 1.0]
gamma_s0 (scale 1 fine): [1.00, 1.05, 1.10, 1.15]
gamma_s1 (scale 2 coarse): [0.85, 0.90, 0.95, 1.00]
"""

import os
import sys
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
    ckpt_path = "checkpoints/pcssflow_v2_sidd_gpu.pt"
    model = PCSSFlowV2(in_channels=1, hidden_dim=128, n_scales=2, physics_mode="srgb").to(dev)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()

    pairs = load_sidd_paired_scenes("data/SIDD/train/train", max_scenes=40, crop_size=128)

    def run_sample(noisy: torch.Tensor, g_low: float, g_s0: float, g_s1: float):
        b, c, h, w = noisy.shape
        curr_x = noisy.clone()
        pc = model.pc_calc(noisy)
        scale_map = model.rectified_flow.physics_scheduler(noisy)
        dt = 0.25  # 4 steps
        gammas = [g_s0, g_s1]

        for step in range(4):
            t_val = step * dt
            t_batch = torch.full((b,), t_val, device=dev, dtype=torch.float32)

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
            pred_clow = curr_clow + dt * v_clow * g_low
            pred_highs = [
                mag_phase_to_complex(curr_mags[s] + dt * v_mags[s] * gammas[s], curr_phases[s])
                for s in range(2)
            ]
            pred_x = model.rectified_flow.dtcwt_inv(pred_clow, pred_highs, orig_size=(h, w))

            # Corrector step
            t_next = min(1.0, t_val + dt)
            t_next_batch = torch.full((b,), t_next, device=dev, dtype=torch.float32)

            c_low_next, highs_next = model.rectified_flow.dtcwt_fwd(pred_x)
            next_mags, next_phases = [], []
            for h_s in highs_next:
                m, p = complex_to_mag_phase(h_s)
                next_mags.append(m)
                next_phases.append(p)

            v_clow_next, v_mags_next, _ = model.rectified_flow.velocity_net(
                c_low_next, next_mags, next_phases, t_next_batch, pc, scale_map, ssm_feat=pc
            )

            final_clow = curr_clow + 0.5 * dt * (v_clow + v_clow_next) * g_low
            rec_highs = [
                mag_phase_to_complex(
                    curr_mags[s] + 0.5 * dt * (v_mags[s] + v_mags_next[s]) * gammas[s],
                    curr_phases[s]
                )
                for s in range(2)
            ]
            curr_x = model.rectified_flow.dtcwt_inv(final_clow, rec_highs, orig_size=(h, w))

        return model.rectified_flow.proximal_op(y_noisy=noisy, x_flow=curr_x)

    # Coarse sweep candidates
    candidates = [
        (1.0, 1.0, 1.0),
        (0.90, 1.05, 0.95),
        (0.85, 1.08, 0.92),
        (0.80, 1.10, 0.90),
        (0.85, 1.12, 0.90),
        (0.80, 1.15, 0.85),
        (0.75, 1.15, 0.85),
        (0.85, 1.10, 0.95),
        (0.90, 1.10, 0.95),
    ]

    print("--- SWEEPING SPECTRAL SIZING (g_low, g_s0, g_s1) ---")
    best_psnr, best_cfg = 0.0, None
    for g_low, g_s0, g_s1 in candidates:
        psnr_list, ssim_list = [], []
        for _, noisy_t, gt_t in pairs:
            noisy = noisy_t.to(dev)
            clean = gt_t.to(dev)
            with torch.no_grad():
                out = torch.clamp(run_sample(noisy, g_low, g_s0, g_s1), 0.0, 1.0)
                m = compute_image_metrics(out, clean)
                psnr_list.append(m["psnr"])
                ssim_list.append(m["ssim"])
        m_psnr = float(np.mean(psnr_list))
        m_ssim = float(np.mean(ssim_list))
        delta = m_psnr - 35.717
        print(f"low={g_low:.2f}, s0={g_s0:.2f}, s1={g_s1:.2f} | PSNR: {m_psnr:6.3f} dB ({delta:+6.3f}) | SSIM: {m_ssim:6.4f}")
        if m_psnr > best_psnr:
            best_psnr = m_psnr
            best_cfg = (g_low, g_s0, g_s1)

    print(f"\nBest Config: low={best_cfg[0]}, s0={best_cfg[1]}, s1={best_cfg[2]} -> PSNR: {best_psnr:.3f} dB")

if __name__ == "__main__":
    main()
