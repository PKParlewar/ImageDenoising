"""
Phase-Domain Rectified Flow with Asymmetric Channel Transport and ODE Samplers.

Section 5 & 6 of info.md:
Operates over DTCWT (magnitude A, phase phi) representation:
    z_t = (A_t, phi_t) = (t * A_1 + (1-t) * A_0, t * phi_1 + (1-t) * phi_0)
Asymmetric loss:
    L_Flow = E_t[ w_A * || v_theta^A - (A_1 - A_0) ||^2 + w_phi * || v_theta^phi - (phi_1 - phi_0) ||^2 ]
where w_A >> w_phi.
"""

from typing import Tuple, List, Optional, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.wavelets.dtcwt import (
    DTCWTForward,
    DTCWTInverse,
    complex_to_mag_phase,
    mag_phase_to_complex,
)
from src.wavelets.phase_congruency import PhaseCongruencyCalculator
from .physics_schedule import PhysicsScheduler
from .proximal import PhysicsProximalOperator


class AsymmetricVelocityNet(nn.Module):
    """
    Velocity field neural network predicting (v^A, v^phi) at time t.
    Conditioned on Phase Congruency map and physics scale map.
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 128,
        n_scales: int = 3,
        time_emb_dim: int = 32
    ):
        super().__init__()
        self.n_scales = n_scales
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Time step embedding: Sinusoidal positional embedding + MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_emb_dim = time_emb_dim

        # Condition encoder (PC map + physics scale map + SSM global structural feature map)
        # in_channels * 2 + 1 = 3 channels for grayscale (PC, scale_map, ssm_feat)
        cond_in_dim = in_channels * 2 + 1
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_in_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
        )

        # Low-frequency approximation velocity block with dilated convolutions (receptive field 31x31)
        last_clow_conv = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(last_clow_conv.weight)
        nn.init.zeros_(last_clow_conv.bias)
        self.clow_block = nn.Sequential(
            nn.Conv2d(in_channels + hidden_dim // 2 + hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.LeakyReLU(0.2, inplace=True),
            last_clow_conv,
        )

        # Scale velocity blocks for magnitude per DTCWT scale with multi-scale dilation
        self.scale_blocks = nn.ModuleList()
        for s in range(n_scales):
            in_dim = in_channels * 6 + hidden_dim // 2 + hidden_dim
            last_mag_conv = nn.Conv2d(hidden_dim, in_channels * 6, kernel_size=3, padding=1)
            nn.init.zeros_(last_mag_conv.weight)
            nn.init.zeros_(last_mag_conv.bias)
            block = nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
                nn.LeakyReLU(0.2, inplace=True),
                last_mag_conv,
            )
            self.scale_blocks.append(block)

    def _get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding."""
        half_dim = self.time_emb_dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_factor)
        args = t[:, None] * freqs[None, :]
        time_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_mlp(time_emb)

    def forward(
        self,
        clow_t: torch.Tensor,
        mags_t: List[torch.Tensor],
        phases_t: List[torch.Tensor],
        t: torch.Tensor,
        pc: torch.Tensor,
        scale_map: torch.Tensor,
        ssm_feat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            clow_t: Lowpass approximation subband [B, C, H_low, W_low]
            mags_t: List of S magnitude tensors [B, C, 6, H_s, W_s]
            phases_t: List of S phase tensors [B, C, 6, H_s, W_s]
            t: [B] scalar time values in [0, 1]
            pc: [B, C, H, W] Phase Congruency map
            scale_map: [B, 1, H, W] physics scale map
            ssm_feat: Optional [B, C, H, W] structure-guided SSM global feature map
            
        Returns:
            (v_clow, v_mags, v_phases): Predicted velocity tensors
        """
        b = t.shape[0]
        t_emb = self._get_time_embedding(t)  # [B, hidden_dim]
        
        if ssm_feat is None:
            ssm_feat = pc
        cond_input = torch.cat([pc, scale_map, ssm_feat], dim=1)
        cond_feat = self.cond_encoder(cond_input)  # [B, hidden_dim // 2, H, W]

        # 1. Lowpass approximation velocity
        h_low, w_low = clow_t.shape[-2:]
        cond_low = F.interpolate(cond_feat, size=(h_low, w_low), mode="bilinear", align_corners=False)
        t_low = t_emb.view(b, self.hidden_dim, 1, 1).repeat(1, 1, h_low, w_low)
        inp_low = torch.cat([clow_t, cond_low, t_low], dim=1)
        v_clow = self.clow_block(inp_low)

        # 2. Highpass magnitude velocities (phase is invariant)
        v_mags = []
        v_phases = []

        for s_idx in range(self.n_scales):
            mag_s = mags_t[s_idx]
            phase_s = phases_t[s_idx]
            _, c, n_dirs, h_s, w_s = mag_s.shape

            mag_flat = mag_s.view(b, c * n_dirs, h_s, w_s)

            cond_s = F.interpolate(cond_feat, size=(h_s, w_s), mode="bilinear", align_corners=False)
            t_s = t_emb.view(b, self.hidden_dim, 1, 1).repeat(1, 1, h_s, w_s)

            inp = torch.cat([mag_flat, cond_s, t_s], dim=1)
            v_mag_flat = self.scale_blocks[s_idx](inp)

            v_mags.append(v_mag_flat.view(b, c, n_dirs, h_s, w_s))
            v_phases.append(torch.zeros_like(phase_s))

        return v_clow, v_mags, v_phases


class DTCWTRectifiedFlow(nn.Module):
    """
    End-to-End DTCWT Phase-Domain Rectified Flow framework.
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 128,
        n_scales: int = 3,
        weight_mag: float = 2.0,
        weight_phase: float = 0.05,
        physics_mode: str = "raw"
    ):
        super().__init__()
        self.n_scales = n_scales
        self.weight_mag = weight_mag
        self.weight_phase = weight_phase

        self.dtcwt_fwd = DTCWTForward(n_scales=n_scales)
        self.dtcwt_inv = DTCWTInverse()
        self.pc_calc = PhaseCongruencyCalculator(n_scales=n_scales)
        self.physics_scheduler = PhysicsScheduler(mode=physics_mode)
        self.proximal_op = PhysicsProximalOperator(mode=physics_mode)

        self.velocity_net = AsymmetricVelocityNet(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            n_scales=n_scales
        )

    def compute_flow_loss(
        self,
        y_noisy: torch.Tensor,
        x_clean: torch.Tensor,
        ssm_feat: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the asymmetric rectified flow matching loss with spatial consistency.
        """
        b, c, h, w = y_noisy.shape
        device = y_noisy.device

        # Decompose noisy (z_0) and clean (z_1) into DTCWT representation
        c_low_0, highs_0 = self.dtcwt_fwd(y_noisy)
        c_low_1, highs_1 = self.dtcwt_fwd(x_clean)

        mags_0, phases_0 = [], []
        mags_1, phases_1 = [], []

        for h0, h1 in zip(highs_0, highs_1):
            m0, p0 = complex_to_mag_phase(h0)
            m1, p1 = complex_to_mag_phase(h1)
            mags_0.append(m0)
            phases_0.append(p0)
            mags_1.append(m1)
            phases_1.append(p1)

        # Sample random continuous time t in [0, 1]
        t = torch.rand(b, device=device)

        # Linear trajectory interpolation: z_t = t * z_1 + (1-t) * z_0
        t_low = t.view(b, 1, 1, 1)
        clow_t = t_low * c_low_1 + (1.0 - t_low) * c_low_0
        u_clow = c_low_1 - c_low_0

        mags_t, phases_t = [], []
        u_mags, u_phases = [], []  # Ground truth velocity vectors

        for s in range(self.n_scales):
            t_s = t.view(b, 1, 1, 1, 1)
            m_t = t_s * mags_1[s] + (1.0 - t_s) * mags_0[s]
            p_t = t_s * phases_1[s] + (1.0 - t_s) * phases_0[s]
            mags_t.append(m_t)
            phases_t.append(p_t)

            u_mags.append(mags_1[s] - mags_0[s])
            u_phases.append(phases_1[s] - phases_0[s])

        # Compute conditioning priors
        pc = self.pc_calc(y_noisy)
        scale_map = self.physics_scheduler(y_noisy)

        # Predict velocity field
        v_clow, v_mags, v_phases = self.velocity_net(clow_t, mags_t, phases_t, t, pc, scale_map, ssm_feat=ssm_feat)

        # Asymmetric loss computation
        loss_clow = F.mse_loss(v_clow, u_clow)
        loss_mag = torch.tensor(0.0, device=device)
        loss_phase = torch.tensor(0.0, device=device)

        for s in range(self.n_scales):
            loss_mag = loss_mag + F.mse_loss(v_mags[s], u_mags[s])
            loss_phase = loss_phase + F.mse_loss(v_phases[s], u_phases[s])

        total_loss = loss_clow + self.weight_mag * loss_mag + self.weight_phase * loss_phase

        return {
            "loss": total_loss,
            "loss_clow": loss_clow,
            "loss_mag": loss_mag,
            "loss_phase": loss_phase
        }

    @torch.no_grad()
    def denoise_sample(
        self,
        y_noisy: torch.Tensor,
        num_steps: int = 2,
        ssm_feat: Optional[torch.Tensor] = None,
        solver: str = "heun",
        use_spectral_sizing: bool = False,
        spectral_gammas: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Few-step ODE sampling from t=0 to t=1 in DTCWT domain.
        Supports 1st-order Euler and 2nd-order Heun predictor-corrector.
        Optionally applies Scale-Decoupled Spectral Physics Sizing to modulate
        subband velocities (decoupling lowpass drift from high-frequency wavelet edges).
        """
        b, c, h, w = y_noisy.shape
        device = y_noisy.device

        curr_x = y_noisy.clone()
        pc = self.pc_calc(y_noisy)
        scale_map = self.physics_scheduler(y_noisy)

        # Scale-decoupled spectral physics sizing factors
        if use_spectral_sizing or spectral_gammas is not None:
            if spectral_gammas is None:
                g_low = 0.75
                gammas = [1.15, 0.85] + [1.0] * max(0, self.n_scales - 2)
            else:
                g_low = spectral_gammas.get("low", 0.75)
                gammas = spectral_gammas.get("scales", [1.15, 0.85] + [1.0] * max(0, self.n_scales - 2))
        else:
            g_low = 1.0
            gammas = [1.0] * self.n_scales

        dt = 1.0 / num_steps

        # Multi-step Phase-Domain ODE Integration with phase re-analysis
        for step in range(num_steps):
            t_val = step * dt
            t_batch = torch.full((b,), t_val, device=device, dtype=torch.float32)

            c_low, highs = self.dtcwt_fwd(curr_x)
            curr_clow = c_low
            curr_mags, curr_phases = [], []
            for h_s in highs:
                m, p = complex_to_mag_phase(h_s)
                curr_mags.append(m)
                curr_phases.append(p)

            v_clow, v_mags, _ = self.velocity_net(curr_clow, curr_mags, curr_phases, t_batch, pc, scale_map, ssm_feat=ssm_feat)

            if solver.lower() == "heun":
                # Predictor step: Euler forward estimate at t + dt with spectral sizing
                pred_clow = curr_clow + dt * v_clow * g_low
                pred_highs = [
                    mag_phase_to_complex(curr_mags[s] + dt * v_mags[s] * gammas[s], curr_phases[s])
                    for s in range(self.n_scales)
                ]
                pred_x = self.dtcwt_inv(pred_clow, pred_highs, orig_size=(h, w))

                # Corrector step: Evaluate velocity at predicted state (t + dt)
                t_next = min(1.0, t_val + dt)
                t_next_batch = torch.full((b,), t_next, device=device, dtype=torch.float32)
                c_low_next, highs_next = self.dtcwt_fwd(pred_x)
                next_mags, next_phases = [], []
                for h_s in highs_next:
                    m, p = complex_to_mag_phase(h_s)
                    next_mags.append(m)
                    next_phases.append(p)

                v_clow_next, v_mags_next, _ = self.velocity_net(
                    c_low_next, next_mags, next_phases, t_next_batch, pc, scale_map, ssm_feat=ssm_feat
                )

                # Heun 2nd-order trapezoidal integration with spectral sizing
                final_clow = curr_clow + 0.5 * dt * (v_clow + v_clow_next) * g_low
                rec_highs = [
                    mag_phase_to_complex(
                        curr_mags[s] + 0.5 * dt * (v_mags[s] + v_mags_next[s]) * gammas[s],
                        curr_phases[s]
                    )
                    for s in range(self.n_scales)
                ]
                curr_x = self.dtcwt_inv(final_clow, rec_highs, orig_size=(h, w))
            else:
                # 1st-order Euler integration with spectral sizing
                rec_highs = [
                    mag_phase_to_complex(curr_mags[s] + dt * v_mags[s] * gammas[s], curr_phases[s])
                    for s in range(self.n_scales)
                ]
                curr_x = self.dtcwt_inv(curr_clow + dt * v_clow * g_low, rec_highs, orig_size=(h, w))

        # Apply closed-form physics proximal safeguard
        x_final = self.proximal_op(y_noisy=y_noisy, x_flow=curr_x)
        return x_final
