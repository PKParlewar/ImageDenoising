"""
Physics-Calibrated Step Scheduling with Explicit Dual Modes.

Section 6 & Flagged Requirement:
Mode A: Raw Linear Sensor Mode (Poisson-Gaussian noise variance sigma_s^2 * x + sigma_r^2).
Mode B: sRGB / Grayscale Mode (Empirical residual noise variance, avoiding invalid Poisson-Gaussian assumptions).
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def estimate_raw_sensor_variance(
    x_hat: torch.Tensor,
    shot_noise_sigma: float = 0.01,
    read_noise_sigma: float = 0.005
) -> torch.Tensor:
    """
    Mode A: Evaluates heteroscedastic Poisson-Gaussian noise variance for raw linear sensor data:
        Var(y|x) = sigma_s^2 * x_hat + sigma_r^2
        
    Args:
        x_hat: [B, C, H, W] estimated linear image.
        shot_noise_sigma: sigma_s (shot noise gain).
        read_noise_sigma: sigma_r (read noise standard deviation).
        
    Returns:
        variance_map: [B, C, H, W] per-pixel expected noise variance.
    """
    # Linear sensor signal must be non-negative
    x_clamped = torch.clamp(x_hat, min=0.0)
    var_map = (shot_noise_sigma ** 2) * x_clamped + (read_noise_sigma ** 2)
    return var_map


def estimate_srgb_residual_variance(
    img: torch.Tensor,
    window_size: int = 5
) -> torch.Tensor:
    """
    Mode B: Evaluates empirical spatial residual noise variance for sRGB / Grayscale data
    where nonlinear tone curves and demosaicing invalidate the raw Poisson-Gaussian formula.
    
    Args:
        img: [B, C, H, W] image tensor.
        window_size: Local neighborhood window size.
        
    Returns:
        var_map: [B, 1, H, W] empirical local noise variance map.
    """
    pad = window_size // 2
    local_mean = F.avg_pool2d(
        F.pad(img, (pad, pad, pad, pad), mode="reflect"),
        kernel_size=window_size,
        stride=1
    )
    local_sq_mean = F.avg_pool2d(
        F.pad(img ** 2, (pad, pad, pad, pad), mode="reflect"),
        kernel_size=window_size,
        stride=1
    )
    local_var = torch.clamp(local_sq_mean - local_mean ** 2, min=1e-8)
    
    # Take channel average for multi-channel sRGB
    if local_var.shape[1] > 1:
        local_var = local_var.mean(dim=1, keepdim=True)
    return local_var


class PhysicsScheduler(nn.Module):
    """
    Dispatches step size scaling based on input mode (raw vs sRGB).
    """
    def __init__(
        self,
        mode: str = "raw",
        shot_noise_sigma: float = 0.01,
        read_noise_sigma: float = 0.005,
        base_step_scale: float = 1.0
    ):
        super().__init__()
        assert mode in ["raw", "srgb", "none"], f"Unknown mode: {mode}"
        self.mode = mode
        self.shot_noise_sigma = shot_noise_sigma
        self.read_noise_sigma = read_noise_sigma
        self.base_step_scale = base_step_scale

    def forward(
        self,
        img: torch.Tensor,
        current_estimate: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes per-region conditioning signal scale(u, v) for the flow step size.
        
        Returns:
            scale_map: [B, 1, H, W] normalized transport scale in [0, 1].
        """
        if self.mode == "none":
            return torch.ones((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=img.dtype, device=img.device)

        if self.mode == "raw":
            est = current_estimate if current_estimate is not None else img
            var_map = estimate_raw_sensor_variance(
                est,
                shot_noise_sigma=self.shot_noise_sigma,
                read_noise_sigma=self.read_noise_sigma
            )
            std_map = torch.sqrt(var_map)
            # Normalize to [0, 1] relative scale
            scale_map = std_map / (std_map.max() + 1e-8)
            if scale_map.shape[1] > 1:
                scale_map = scale_map.mean(dim=1, keepdim=True)
            return scale_map * self.base_step_scale

        elif self.mode == "srgb":
            var_map = estimate_srgb_residual_variance(img)
            std_map = torch.sqrt(var_map)
            scale_map = std_map / (std_map.max() + 1e-8)
            return scale_map * self.base_step_scale
