"""
Evaluation metrics: PSNR, SSIM, and Composite Metrics in PyTorch.
"""

from typing import Dict, Tuple, Optional
import math
import torch
import torch.nn.functional as F


def calculate_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 1.0,
    crop_border: int = 0
) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in dB.
    """
    if crop_border > 0:
        img1 = img1[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]

    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def calculate_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
    crop_border: int = 0
) -> float:
    """
    Computes Structural Similarity Index Measure (SSIM).
    """
    if crop_border > 0:
        img1 = img1[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # Gaussian window weights
    sigma = 1.5
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    channels = img1.shape[1]
    window = window.repeat(channels, 1, 1, 1)

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


def compute_image_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    crop_border: int = 0
) -> Dict[str, float]:
    """
    Computes all standard image quality metrics.
    """
    psnr_val = calculate_psnr(pred, target, data_range=data_range, crop_border=crop_border)
    ssim_val = calculate_ssim(pred, target, data_range=data_range, crop_border=crop_border)
    mae_val = torch.mean(torch.abs(pred - target)).item()
    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "mae": mae_val,
    }
