"""
Noise correlation estimation and flat-region autocorrelation detection.

Step 1 of Section 2 in info.md:
Measures the empirical spatial autocorrelation kappa_hat(Delta u, Delta v)
from near-uniform image regions F subset y to determine the exact noise correlation length.
Fast, differentiable FFT-based 2D autocorrelation implementation.
"""

from typing import Tuple, Optional, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def detect_flat_regions(
    img: torch.Tensor,
    patch_size: int = 16,
    variance_quantile: float = 0.20
) -> torch.Tensor:
    """
    Selects near-uniform patches F subset y (e.g. shadow areas, sky, black-level regions)
    based on lowest local spatial variance using fast 2D pooling.
    """
    b, c, h, w = img.shape
    stride = max(1, patch_size // 2)

    mean_i = F.avg_pool2d(img, kernel_size=patch_size, stride=stride)
    mean_i2 = F.avg_pool2d(img ** 2, kernel_size=patch_size, stride=stride)
    local_var = (mean_i2 - mean_i ** 2).clamp(min=0.0).mean(dim=1, keepdim=True)  # [B, 1, H_p, W_p]

    # Find patches with lowest variance per batch element
    var_flat = local_var.view(b, -1)
    var_mean = var_flat.mean(dim=-1, keepdim=True)
    var_std = var_flat.std(dim=-1, keepdim=True)
    
    # If the image is homogeneous (variance is uniform across patches), keep full image
    is_homogeneous = (var_std / (var_mean + 1e-6)) < 0.4
    
    thresh = torch.quantile(var_flat, variance_quantile, dim=-1, keepdim=True).view(b, 1, 1, 1)
    flat_patch_mask = (local_var <= thresh).float()

    # Upsample mask to full resolution
    flat_mask = F.interpolate(flat_patch_mask, size=(h, w), mode="nearest")
    for bi in range(b):
        if is_homogeneous[bi]:
            flat_mask[bi] = 1.0
            
    return flat_mask


def estimate_noise_autocorrelation(
    img: torch.Tensor,
    flat_mask: Optional[torch.Tensor] = None,
    max_lag: int = 7,
    lowpass_kernel_size: int = 5
) -> torch.Tensor:
    """
    Computes empirical spatial autocorrelation using fast 2D FFT Wiener-Khinchin theorem:
        r = (y - lowpass(y)) * flat_mask
        kappa_hat = iFFT2(|FFT2(r)|^2)
    """
    b, c, h, w = img.shape
    device = img.device
    
    if flat_mask is None:
        flat_mask = detect_flat_regions(img)
        
    # Extract high-frequency residual: r = y - lowpass(y)
    pad = lowpass_kernel_size // 2
    lowpass = F.avg_pool2d(
        F.pad(img, (pad, pad, pad, pad), mode="reflect"),
        kernel_size=lowpass_kernel_size,
        stride=1
    )
    residual = (img - lowpass) * flat_mask  # [B, C, H, W]
    
    # Fast 2D FFT Autocorrelation with explicit padding
    fft_h = h + 2 * max_lag
    fft_w = w + 2 * max_lag
    res_padded = F.pad(residual, (0, 2 * max_lag, 0, 2 * max_lag), mode="constant", value=0.0)
    
    r_fft = torch.fft.rfft2(res_padded)
    psd = r_fft.real ** 2 + r_fft.imag ** 2
    autocorr_full = torch.fft.irfft2(psd, s=(fft_h, fft_w))  # [B, C, fft_h, fft_w]
    
    # Center the autocorrelation: shift zero-lag to center
    autocorr_centered = torch.fft.fftshift(autocorr_full, dim=(-2, -1))
    
    # Crop central [2*max_lag + 1, 2*max_lag + 1] window
    center_h = fft_h // 2
    center_w = fft_w // 2
    
    kappa_raw = autocorr_centered[
        :, :,
        center_h - max_lag : center_h + max_lag + 1,
        center_w - max_lag : center_w + max_lag + 1
    ]
    
    # Normalize by zero-lag peak (variance)
    zero_lag = kappa_raw[:, :, max_lag, max_lag].unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
    kappa_hat = kappa_raw / zero_lag
    return kappa_hat


def estimate_correlation_length(
    kappa_hat: torch.Tensor,
    decay_threshold: float = 0.2
) -> torch.Tensor:
    """
    Calculates the effective correlation radius (in pixels) from kappa_hat in a fully vectorized manner on GPU.
    """
    b, c, k, _ = kappa_hat.shape
    center = k // 2
    
    y_coords, x_coords = torch.meshgrid(
        torch.arange(k, device=kappa_hat.device) - center,
        torch.arange(k, device=kappa_hat.device) - center,
        indexing="ij"
    )
    radial_dist = torch.sqrt((x_coords ** 2 + y_coords ** 2).float())  # [K, K]
    
    corr_mask = (kappa_hat >= decay_threshold).float()  # [B, C, K, K]
    masked_rad = radial_dist.unsqueeze(0).unsqueeze(0) * corr_mask  # [B, C, K, K]
    corr_lengths = masked_rad.view(b, c, -1).max(dim=-1).values.clamp(min=1.0)  # [B, C]
    return corr_lengths


class NoiseEstimator(nn.Module):
    """
    End-to-end module that estimates noise autocorrelation and correlation length.
    """
    def __init__(self, max_lag: int = 11, decay_threshold: float = 0.15, lowpass_kernel_size: int = 15):
        super().__init__()
        self.max_lag = max_lag
        self.decay_threshold = decay_threshold
        self.lowpass_kernel_size = lowpass_kernel_size

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flat_mask = detect_flat_regions(img)
        kappa_hat = estimate_noise_autocorrelation(
            img,
            flat_mask=flat_mask,
            max_lag=self.max_lag,
            lowpass_kernel_size=self.lowpass_kernel_size
        )
        corr_length = estimate_correlation_length(kappa_hat, decay_threshold=self.decay_threshold)
        return kappa_hat, corr_length
