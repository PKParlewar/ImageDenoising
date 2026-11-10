#!/usr/bin/env python3
"""
Self-Supervised Phase-Domain Reflow (2-Rectified Flow Matching):
1. Loads current 1-Rectified Flow checkpoint (PC-Flow v1).
2. Generates self-supervised pseudo-endpoints x_hat using 4-step Heun with Spectral Sizing on training crops.
3. Retrains/fine-tunes the velocity net on straight trajectories:
     z_t = t * x_hat + (1 - t) * y_noisy
     u_target = x_hat - y_noisy
   under Phase Congruency and physics scaling conditioning.
4. Evaluates whether 2-step and 4-step solvers achieve higher PSNR and straight-line efficiency on 40 SIDD scenes.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.pc_ssflow import PCSSFlowV2
from src.wavelets.dtcwt import complex_to_mag_phase, mag_phase_to_complex
from src.utils.metrics import compute_image_metrics
from scripts.evaluate import load_sidd_paired_scenes
from scripts.train import load_sidd_training_pairs
from src.utils.device import get_default_device

def main():
    dev = get_default_device("auto")
    print("==========================================================================")
    print(f"SELF-SUPERVISED PHASE-DOMAIN REFLOW (Device: {dev})")
    print("==========================================================================")

    base_ckpt = "checkpoints/pcssflow_v2_sidd_gpu.pt"
    reflow_ckpt = "checkpoints/pcssflow_v2_reflow.pt"

    model = PCSSFlowV2(in_channels=1, hidden_dim=128, n_scales=2, physics_mode="srgb").to(dev)
    model.load_state_dict(torch.load(base_ckpt, map_location=dev))
    model.eval()

    # 1. Load training crops
    print("Loading 400 training crops from SIDD scenes...")
    train_pairs = load_sidd_training_pairs("data/SIDD/train/train", max_scenes=50, crop_size=128, crops_per_scene=8)
    noisy_train = torch.cat([p[0] for p in train_pairs], dim=0).to(dev)
    print(f"Loaded {noisy_train.shape[0]} training crops.")

    # 2. Generate self-supervised endpoints x_hat using 4-step Heun with Spectral Sizing
    print("Generating self-supervised reflow endpoints (x_hat) using current PC-Flow...")
    endpoints = []
    g_low, g_s0, g_s1 = 0.75, 1.15, 0.85
    gammas = [g_s0, g_s1]
    dt = 0.25
    b_size = 16

    t0_gen = time.perf_counter()
    with torch.no_grad():
        for i in range(0, noisy_train.shape[0], b_size):
            batch = noisy_train[i : i + b_size]
            b_len = batch.shape[0]
            curr_x = batch.clone()
            pc = model.pc_calc(batch)
            scale_map = model.rectified_flow.physics_scheduler(batch)

            for step in range(4):
                t_val = step * dt
                t_batch = torch.full((b_len,), t_val, device=dev, dtype=torch.float32)

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

                pred_clow = curr_clow + dt * v_clow * g_low
                pred_highs = [
                    mag_phase_to_complex(curr_mags[s] + dt * v_mags[s] * gammas[s], curr_phases[s])
                    for s in range(2)
                ]
                pred_x = model.rectified_flow.dtcwt_inv(pred_clow, pred_highs, orig_size=(128, 128))

                t_next = min(1.0, t_val + dt)
                t_next_batch = torch.full((b_len,), t_next, device=dev, dtype=torch.float32)
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
                curr_x = model.rectified_flow.dtcwt_inv(final_clow, rec_highs, orig_size=(128, 128))

            x_out = model.rectified_flow.proximal_op(y_noisy=batch, x_flow=curr_x)
            endpoints.append(torch.clamp(x_out, 0.0, 1.0))

    x_hat_train = torch.cat(endpoints, dim=0)
    print(f"Generated {x_hat_train.shape[0]} endpoints in {time.perf_counter() - t0_gen:.1f}s.")

    # 3. Train Reflow (Straight-Path Matching)
    print("\nTraining Phase-Domain Reflow on straight trajectories (25 epochs)...")
    optimizer = torch.optim.Adam(model.rectified_flow.velocity_net.parameters(), lr=0.0005)
    model.train()

    epochs = 25
    n_samples = noisy_train.shape[0]

    for ep in range(epochs):
        perm = torch.randperm(n_samples, device=dev)
        noisy_shuf = noisy_train[perm]
        target_shuf = x_hat_train[perm]
        total_loss = 0.0
        n_b = 0

        for i in range(0, n_samples, b_size):
            y_b = noisy_shuf[i : i + b_size]
            x_b = target_shuf[i : i + b_size]
            b_curr = y_b.shape[0]

            optimizer.zero_grad()

            # DTCWT decomposition of endpoints
            c_low_0, highs_0 = model.rectified_flow.dtcwt_fwd(y_b)
            c_low_1, highs_1 = model.rectified_flow.dtcwt_fwd(x_b)

            mags_0, phases_0 = [], []
            mags_1, phases_1 = [], []
            for h0, h1 in zip(highs_0, highs_1):
                m0, p0 = complex_to_mag_phase(h0)
                m1, p1 = complex_to_mag_phase(h1)
                mags_0.append(m0)
                phases_0.append(p0)
                mags_1.append(m1)
                phases_1.append(p1)

            # Random continuous time t
            t = torch.rand(b_curr, device=dev)
            t_low = t.view(b_curr, 1, 1, 1)

            # Straight line interpolation
            clow_t = t_low * c_low_1 + (1.0 - t_low) * c_low_0
            u_clow = c_low_1 - c_low_0

            mags_t, phases_t = [], []
            u_mags = []
            for s in range(2):
                t_s = t.view(b_curr, 1, 1, 1, 1)
                m_t = t_s * mags_1[s] + (1.0 - t_s) * mags_0[s]
                p_t = phases_0[s]  # Phase is invariant
                mags_t.append(m_t)
                phases_t.append(p_t)
                u_mags.append(mags_1[s] - mags_0[s])

            pc = model.pc_calc(y_b)
            scale_map = model.rectified_flow.physics_scheduler(y_b)

            v_clow, v_mags, _ = model.rectified_flow.velocity_net(
                clow_t, mags_t, phases_t, t, pc, scale_map, ssm_feat=pc
            )

            loss_clow = F.mse_loss(v_clow, u_clow)
            loss_mag = F.mse_loss(v_mags[0], u_mags[0]) + F.mse_loss(v_mags[1], u_mags[1])
            loss = loss_clow + 2.0 * loss_mag

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.rectified_flow.velocity_net.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_b += 1

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"Reflow Epoch [{ep+1:2d}/{epochs:2d}] - Loss: {total_loss / n_b:.6f}")

    torch.save(model.state_dict(), reflow_ckpt)
    print(f"\nSaved Reflow checkpoint to {reflow_ckpt}")

    # 4. Evaluate on 40 SIDD Scenes!
    print("\n--- EVALUATING REFLOW MODEL ON 40 SIDD SCENES ---")
    model.eval()
    val_pairs = load_sidd_paired_scenes("data/SIDD/train/train", max_scenes=40, crop_size=128)

    def eval_solver(steps: int, solver: str):
        psnr_l, ssim_l = [], []
        dt_s = 1.0 / steps
        for _, noisy_t, gt_t in val_pairs:
            noisy = noisy_t.to(dev)
            clean = gt_t.to(dev)
            b = noisy.shape[0]
            curr_x = noisy.clone()
            pc = model.pc_calc(noisy)
            scale_map = model.rectified_flow.physics_scheduler(noisy)

            with torch.no_grad():
                for step in range(steps):
                    t_val = step * dt_s
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

                    if solver == "heun":
                        pred_clow = curr_clow + dt_s * v_clow * g_low
                        pred_highs = [
                            mag_phase_to_complex(curr_mags[s] + dt_s * v_mags[s] * gammas[s], curr_phases[s])
                            for s in range(2)
                        ]
                        pred_x = model.rectified_flow.dtcwt_inv(pred_clow, pred_highs, orig_size=(128, 128))
                        t_next = min(1.0, t_val + dt_s)
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
                        final_clow = curr_clow + 0.5 * dt_s * (v_clow + v_clow_next) * g_low
                        rec_highs = [
                            mag_phase_to_complex(
                                curr_mags[s] + 0.5 * dt_s * (v_mags[s] + v_mags_next[s]) * gammas[s],
                                curr_phases[s]
                            )
                            for s in range(2)
                        ]
                        curr_x = model.rectified_flow.dtcwt_inv(final_clow, rec_highs, orig_size=(128, 128))
                    else:
                        # 1st-order Euler
                        final_clow = curr_clow + dt_s * v_clow * g_low
                        rec_highs = [
                            mag_phase_to_complex(curr_mags[s] + dt_s * v_mags[s] * gammas[s], curr_phases[s])
                            for s in range(2)
                        ]
                        curr_x = model.rectified_flow.dtcwt_inv(final_clow, rec_highs, orig_size=(128, 128))

                out = model.rectified_flow.proximal_op(y_noisy=noisy, x_flow=curr_x)
                out = torch.clamp(out, 0.0, 1.0)
                m = compute_image_metrics(out, clean)
                psnr_l.append(m["psnr"])
                ssim_l.append(m["ssim"])

        return float(np.mean(psnr_l)), float(np.mean(ssim_l))

    for solver, steps in [("euler", 2), ("euler", 4), ("heun", 2), ("heun", 4)]:
        psnr_val, ssim_val = eval_solver(steps, solver)
        print(f"Reflow Model | Solver: {solver:5s} | Steps: {steps:2d} | PSNR: {psnr_val:6.3f} dB | SSIM: {ssim_val:6.4f}")

if __name__ == "__main__":
    main()
