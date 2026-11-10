"""
PC-SSFlow v2: Full Unified Denoising Framework.

Integrates:
1. DTCWT multi-scale directional decomposition
2. Phase Congruency invariance representation & noise autocorrelation estimation
3. Scale-adaptive spectral masking
4. Structure-guided (PC-warped) Hilbert-SSM backbone
5. Phase-domain rectified flow with physics-calibrated step sizing and proximal safeguard
"""

from typing import Tuple, List, Optional, Dict, Any
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
from src.masking.noise_estimator import NoiseEstimator
from src.masking.adaptive_mask import ScaleAdaptiveMasker, MagnitudeReconstructionNet
from src.backbone.ssm_block import StructureGuidedSSMBackbone
from src.flow.rectified_flow import DTCWTRectifiedFlow
from src.flow.physics_schedule import PhysicsScheduler
from src.flow.proximal import PhysicsProximalOperator


class PCSSFlowV2(nn.Module):
    """
    PC-SSFlow v2 Complete Architecture.
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 128,
        n_scales: int = 2,
        warping_window: int = 8,
        physics_mode: str = "raw",
        shot_noise_sigma: float = 0.01,
        read_noise_sigma: float = 0.005
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_scales = n_scales
        self.warping_window = warping_window
        self.physics_mode = physics_mode

        # 1. Wavelet & Phase Congruency
        self.dtcwt_fwd = DTCWTForward(n_scales=n_scales)
        self.dtcwt_inv = DTCWTInverse()
        self.pc_calc = PhaseCongruencyCalculator(n_scales=n_scales)

        # 2. Noise Autocorrelation & Adaptive Masking
        self.noise_estimator = NoiseEstimator(max_lag=7)
        self.adaptive_masker = ScaleAdaptiveMasker(mask_ratio=0.2)
        self.mag_denoiser = MagnitudeReconstructionNet(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            n_scales=n_scales
        )

        # 3. Structure-Guided (PC-Warped) Hilbert-SSM Backbone
        self.ssm_backbone = StructureGuidedSSMBackbone(
            in_channels=in_channels,
            out_channels=in_channels,
            d_model=hidden_dim,
            d_state=16,
            num_layers=2,
            default_window=warping_window,
            patch_size=4
        )

        # 4. Phase-Domain Rectified Flow
        self.rectified_flow = DTCWTRectifiedFlow(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            n_scales=n_scales,
            physics_mode=physics_mode
        )

        # 5. Physics Scheduler and Proximal Operator
        self.physics_scheduler = PhysicsScheduler(
            mode=physics_mode,
            shot_noise_sigma=shot_noise_sigma,
            read_noise_sigma=read_noise_sigma
        )
        self.proximal_op = PhysicsProximalOperator(
            mode=physics_mode,
            shot_noise_sigma=shot_noise_sigma,
            read_noise_sigma=read_noise_sigma
        )

    def forward_self_supervised_step(
        self,
        y_noisy: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Executes self-supervised stage:
        1. Estimates noise autocorrelation kappa_hat.
        2. Computes Phase Congruency prior.
        3. Applies scale-adaptive masking to magnitude coefficients.
        4. Reconstructs masked magnitude conditioned on PC prior.
        5. Re-synthesizes spatial estimate and enforces PC invariance.
        """
        b, c, h, w = y_noisy.shape

        # Step 1: Noise correlation estimation
        kappa_hat, corr_length = self.noise_estimator(y_noisy)

        # Step 2: DTCWT and Phase Congruency
        c_low, highs = self.dtcwt_fwd(y_noisy)
        pc_orig = self.pc_calc(c_low=c_low, highs=highs)

        # Extract magnitudes & phases
        orig_mags, orig_phases = [], []
        for high_s in highs:
            m_s, p_s = complex_to_mag_phase(high_s)
            orig_mags.append(m_s)
            orig_phases.append(p_s)

        # Step 3: Scale-adaptive masking
        masked_mags, masks, radii = self.adaptive_masker(orig_mags, corr_length)

        # Step 4: Phase-conditioned magnitude reconstruction
        rec_mags = self.mag_denoiser(masked_mags, pc_orig)

        # Step 5: Reconstruct complex subbands & spatial image
        rec_highs = []
        for s in range(self.n_scales):
            rec_h = mag_phase_to_complex(rec_mags[s], orig_phases[s])
            rec_highs.append(rec_h)

        denoised_spatial = self.dtcwt_inv(c_low, rec_highs, orig_size=(h, w))

        # Pass through structure-guided SSM backbone for spatial continuity refinement
        refined_spatial = self.ssm_backbone(denoised_spatial, pc_orig, window_size=self.warping_window)

        # Evaluate reconstructed PC map directly from reconstructed DTCWT subbands
        pc_pred = self.pc_calc(c_low=c_low, highs=rec_highs)

        return {
            "pseudo_clean": refined_spatial,
            "rec_mags": rec_mags,
            "orig_mags": orig_mags,
            "masks": masks,
            "pc_orig": pc_orig,
            "pc_pred": pc_pred,
            "radii": radii,
            "corr_length": corr_length,
        }

    def _denoise_single_channel(
        self,
        y_noisy: torch.Tensor,
        num_flow_steps: int = 2,
        solver: str = "heun",
        use_spectral_sizing: bool = False,
        spectral_gammas: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        # Step 1: Compute Phase Congruency edge prior
        pc = self.pc_calc(y_noisy)

        # Step 2: Extract structure-guided global context features via PC-warped SSM
        ssm_feat = self.ssm_backbone(y_noisy, pc, window_size=self.warping_window)

        # Step 3: Phase-Domain Rectified Flow with SSM global conditioning & physics step scheduling
        x_flow = self.rectified_flow.denoise_sample(
            y_noisy=y_noisy,
            num_steps=num_flow_steps,
            ssm_feat=ssm_feat,
            solver=solver,
            use_spectral_sizing=use_spectral_sizing,
            spectral_gammas=spectral_gammas
        )
        return x_flow

    def _denoise_with_tta(
        self,
        y_noisy: torch.Tensor,
        num_flow_steps: int = 2,
        solver: str = "heun",
        use_spectral_sizing: bool = False,
        spectral_gammas: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        8-fold geometric self-ensemble (TTA):
        Evaluates 4 rotations (0, 90, 180, 270 deg) and horizontal flips,
        inverts transformations, and averages predictions to cancel residual noise.
        """
        outs = []
        for k in [0, 1, 2, 3]:
            # Rotated
            rot_x = torch.rot90(y_noisy, k, [-2, -1])
            out_rot = self._denoise_single_channel(
                rot_x,
                num_flow_steps=num_flow_steps,
                solver=solver,
                use_spectral_sizing=use_spectral_sizing,
                spectral_gammas=spectral_gammas
            )
            outs.append(torch.rot90(out_rot, -k, [-2, -1]))

            # Rotated + Horizontal Flip
            flip_x = torch.flip(rot_x, dims=[-1])
            out_flip = self._denoise_single_channel(
                flip_x,
                num_flow_steps=num_flow_steps,
                solver=solver,
                use_spectral_sizing=use_spectral_sizing,
                spectral_gammas=spectral_gammas
            )
            outs.append(torch.rot90(torch.flip(out_flip, dims=[-1]), -k, [-2, -1]))

        return torch.stack(outs, dim=0).mean(dim=0).clamp(0.0, 1.0)

    def denoise(
        self,
        y_noisy: torch.Tensor,
        num_flow_steps: int = 2,
        solver: str = "heun",
        use_tta: bool = False,
        use_spectral_sizing: bool = False,
        spectral_gammas: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Full inference denoising pipeline:
        Runs Phase-Domain Rectified Flow with Physics step scheduling, SSM structural guidance, and proximal projection.
        Supports 2nd-order Heun predictor-corrector, Scale-Decoupled Spectral Physics Sizing, and 8-fold geometric self-ensemble (TTA).
        """
        denoise_fn = self._denoise_with_tta if use_tta else self._denoise_single_channel

        if y_noisy.shape[1] > self.in_channels and self.in_channels == 1:
            denoised_channels = []
            for c in range(y_noisy.shape[1]):
                ch_out = denoise_fn(
                    y_noisy[:, c:c+1],
                    num_flow_steps=num_flow_steps,
                    solver=solver,
                    use_spectral_sizing=use_spectral_sizing,
                    spectral_gammas=spectral_gammas
                )
                denoised_channels.append(ch_out)
            return torch.cat(denoised_channels, dim=1)
        else:
            return denoise_fn(
                y_noisy,
                num_flow_steps=num_flow_steps,
                solver=solver,
                use_spectral_sizing=use_spectral_sizing,
                spectral_gammas=spectral_gammas
            )
