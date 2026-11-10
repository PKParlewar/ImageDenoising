"""
Differentiable Phase Congruency (PC) and Local Energy Engine in PyTorch.

Phase Congruency measures feature significance via phase alignment across scales
rather than intensity gradient magnitude:
    PC(u, v) = sum_{d, s} E^{s,d}(u, v) / (eps + sum_{d, s} A^{s,d}(u, v))
"""

from typing import List, Tuple, Dict, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dtcwt import DTCWTForward, complex_to_mag_phase


class PhaseCongruencyCalculator(nn.Module):
    """
    Computes normalized 2D Phase Congruency field from DTCWT representation.
    """
    def __init__(
        self,
        n_scales: int = 3,
        noise_threshold_mult: float = 1.5,
        eps: float = 1e-4,
        target_size: Optional[Tuple[int, int]] = None
    ):
        super().__init__()
        self.n_scales = n_scales
        self.noise_threshold_mult = noise_threshold_mult
        self.eps = eps
        self.target_size = target_size
        self.dtcwt_fwd = DTCWTForward(n_scales=n_scales)

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        c_low: Optional[torch.Tensor] = None,
        highs: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Computes Phase Congruency map.
        Accepts either raw input image x: [B, C, H, W] or precomputed DTCWT (c_low, highs).
        Returns:
            pc_map: [B, C, H, W] with values in [0, 1].
        """
        if x is not None:
            c_low, highs = self.dtcwt_fwd(x)
            orig_h, orig_w = x.shape[-2:]
        else:
            if highs is None:
                raise ValueError("Must provide either image x or DTCWT decomposition (c_low, highs).")
            b, c, _, h1, w1, _ = highs[0].shape
            orig_h, orig_w = (h1 * 2, w1 * 2) if self.target_size is None else self.target_size

        batch_size, channels = highs[0].shape[:2]
        
        total_energy = torch.zeros(
            (batch_size, channels, orig_h, orig_w),
            dtype=highs[0].dtype,
            device=highs[0].device
        )
        total_amplitude = torch.zeros(
            (batch_size, channels, orig_h, orig_w),
            dtype=highs[0].dtype,
            device=highs[0].device
        )

        for s_idx, high_s in enumerate(highs):
            # high_s: [B, C, 6, H_s, W_s, 2]
            mag_s, _ = complex_to_mag_phase(high_s, eps=self.eps)

            # Local energy: sum across all 6 directional orientations
            sum_mag_dir = mag_s.sum(dim=2)  # [B, C, H_s, W_s]

            # Upsample scale s to target resolution [orig_h, orig_w]
            sum_mag_upsampled = F.interpolate(
                sum_mag_dir,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False
            )

            # Noise floor estimation for scale s: median amplitude per batch/channel
            noise_est = torch.median(mag_s.reshape(batch_size, channels, -1), dim=-1, keepdim=True).values
            noise_thresh = (self.noise_threshold_mult * noise_est).view(batch_size, channels, 1, 1)
            
            # Energy with noise threshold subtraction (rectified)
            energy_s = torch.clamp(sum_mag_upsampled - noise_thresh, min=0.0)

            total_energy = total_energy + energy_s
            total_amplitude = total_amplitude + sum_mag_upsampled

        # Normalized Phase Congruency map
        pc = total_energy / (total_amplitude + self.eps)
        pc = torch.clamp(pc, min=0.0, max=1.0)
        return pc


def measure_pc_snr_breakdown(
    clean_image: torch.Tensor,
    snr_db_range: List[float] = [30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0, -5.0],
    n_scales: int = 3,
    noise_threshold_mult: float = 1.5
) -> Dict[str, Any]:
    """
    Sweeps noise levels across an SNR grid to characterize where Phase Congruency
    invariance breaks down under high noise.
    """
    calc = PhaseCongruencyCalculator(
        n_scales=n_scales,
        noise_threshold_mult=noise_threshold_mult
    )
    
    clean_pc = calc(clean_image)
    clean_var = torch.var(clean_image).item() + 1e-8

    results = {
        "snr_db_list": snr_db_range,
        "sigma_list": [],
        "pc_mse_list": [],
        "pc_mae_list": [],
        "critical_snr_db": None
    }

    critical_found = False

    for snr_db in snr_db_range:
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_var = clean_var / snr_linear
        sigma = math.sqrt(noise_var)
        results["sigma_list"].append(sigma)

        # Generate noisy image
        noise = torch.randn_like(clean_image) * sigma
        noisy_image = clean_image + noise

        # Calculate PC on noisy image
        noisy_pc = calc(noisy_image)

        # Metric evaluations
        mse = F.mse_loss(noisy_pc, clean_pc).item()
        mae = F.l1_loss(noisy_pc, clean_pc).item()

        results["pc_mse_list"].append(mse)
        results["pc_mae_list"].append(mae)

        if not critical_found and mse > 0.05:
            results["critical_snr_db"] = snr_db
            critical_found = True

    return results
