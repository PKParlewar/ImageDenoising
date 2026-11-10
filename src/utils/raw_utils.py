"""
Raw Bayer pattern processing, sensor calibration, and differentiable ISP utilities.
Supports:
1. Bayer mosaic packing & unpacking (RGGB, BGGR, GRBG, GBRG).
2. Differentiable GPU demosaicing (Bayer mosaic -> 3-channel linear RGB).
3. Bayer mosaicing (3-channel RGB -> 2D Bayer mosaic).
4. Camera Color Correction Matrices (CCM) for smartphone sensors.
5. End-to-end forward ISP pipeline (Raw Bayer -> sRGB).
6. Inverse ISP unprocessing (sRGB -> Linear Raw Bayer + Poisson-Gaussian noise).
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F


# Calibrated Camera Color Correction Matrices (normalized row sum = 1.0)
DEFAULT_CAM2SRGB = torch.tensor([
    [ 1.66, -0.66,  0.00],
    [-0.18,  1.40, -0.22],
    [-0.04, -0.40,  1.44]
], dtype=torch.float32)

GOOGLE_PIXEL_CAM2SRGB = torch.tensor([
    [ 1.72, -0.68, -0.04],
    [-0.20,  1.42, -0.22],
    [-0.03, -0.42,  1.45]
], dtype=torch.float32)

SAMSUNG_S6_CAM2SRGB = torch.tensor([
    [ 1.64, -0.60, -0.04],
    [-0.15,  1.38, -0.23],
    [-0.02, -0.38,  1.40]
], dtype=torch.float32)


def unpack_raw_bayer(raw_2d: torch.Tensor, pattern: str = "RGGB") -> torch.Tensor:
    """
    Unpacks a 2D single-channel Bayer raw mosaic [B, 1, H, W] into 4-channel sub-sampled tensor [B, 4, H/2, W/2]
    corresponding to R, G1, G2, B channels.
    """
    b, _, h, w = raw_2d.shape
    top_left = raw_2d[:, :, 0::2, 0::2]
    top_right = raw_2d[:, :, 0::2, 1::2]
    bottom_left = raw_2d[:, :, 1::2, 0::2]
    bottom_right = raw_2d[:, :, 1::2, 1::2]

    if pattern == "RGGB":
        r, g1, g2, b_ch = top_left, top_right, bottom_left, bottom_right
    elif pattern == "BGGR":
        b_ch, g1, g2, r = top_left, top_right, bottom_left, bottom_right
    elif pattern == "GRBG":
        g1, r, b_ch, g2 = top_left, top_right, bottom_left, bottom_right
    elif pattern == "GBRG":
        g1, b_ch, r, g2 = top_left, top_right, bottom_left, bottom_right
    else:
        raise ValueError(f"Unknown Bayer pattern: {pattern}")

    return torch.cat([r, g1, g2, b_ch], dim=1)


def pack_raw_bayer(raw_4ch: torch.Tensor, pattern: str = "RGGB") -> torch.Tensor:
    """
    Packs 4-channel sub-sampled tensor [B, 4, H/2, W/2] back into 2D Bayer mosaic [B, 1, H, W].
    """
    b, _, h_half, w_half = raw_4ch.shape
    h, w = h_half * 2, w_half * 2
    out = torch.zeros((b, 1, h, w), dtype=raw_4ch.dtype, device=raw_4ch.device)

    # In unpack: return torch.cat([r, g1, g2, b_ch], dim=1)
    r, g1, g2, b_ch = raw_4ch[:, 0:1], raw_4ch[:, 1:2], raw_4ch[:, 2:3], raw_4ch[:, 3:4]

    if pattern == "RGGB":
        top_left, top_right, bottom_left, bottom_right = r, g1, g2, b_ch
    elif pattern == "BGGR":
        top_left, top_right, bottom_left, bottom_right = b_ch, g1, g2, r
    elif pattern == "GRBG":
        top_left, top_right, bottom_left, bottom_right = g1, r, b_ch, g2
    elif pattern == "GBRG":
        top_left, top_right, bottom_left, bottom_right = g1, b_ch, r, g2
    else:
        raise ValueError(f"Unknown Bayer pattern: {pattern}")

    out[:, :, 0::2, 0::2] = top_left
    out[:, :, 0::2, 1::2] = top_right
    out[:, :, 1::2, 0::2] = bottom_left
    out[:, :, 1::2, 1::2] = bottom_right
    return out


def normalize_raw_frame(
    raw_img: torch.Tensor,
    black_level: float = 512.0,
    white_level: float = 16383.0
) -> torch.Tensor:
    """
    Normalizes raw integer pixel values to linear [0.0, 1.0] floating point representation.
    """
    normalized = (raw_img.float() - black_level) / (white_level - black_level)
    return torch.clamp(normalized, min=0.0, max=1.0)


def mosaic_rggb(rgb_3ch: torch.Tensor) -> torch.Tensor:
    """
    Samples a 3-channel RGB tensor [B, 3, H, W] into a 2D single-channel Bayer RGGB mosaic [B, 1, H, W].
    """
    b, _, h, w = rgb_3ch.shape
    bayer = torch.zeros((b, 1, h, w), dtype=rgb_3ch.dtype, device=rgb_3ch.device)

    # RGGB Layout:
    # (0, 0) = R (channel 0)
    # (0, 1) = G (channel 1)
    # (1, 0) = G (channel 1)
    # (1, 1) = B (channel 2)
    bayer[:, 0, 0::2, 0::2] = rgb_3ch[:, 0, 0::2, 0::2]
    bayer[:, 0, 0::2, 1::2] = rgb_3ch[:, 1, 0::2, 1::2]
    bayer[:, 0, 1::2, 0::2] = rgb_3ch[:, 1, 1::2, 0::2]
    bayer[:, 0, 1::2, 1::2] = rgb_3ch[:, 2, 1::2, 1::2]
    return bayer


def demosaic_rggb(bayer_2d: torch.Tensor) -> torch.Tensor:
    """
    Differentiable GPU bilinear demosaicing of 2D single-channel RGGB mosaic [B, 1, H, W]
    into 3-channel linear RGB [B, 3, H, W].
    """
    b, _, h, w = bayer_2d.shape
    device = bayer_2d.device
    dtype = bayer_2d.dtype

    # Create channel masks
    r_mask = torch.zeros((1, 1, 2, 2), dtype=dtype, device=device)
    r_mask[0, 0, 0, 0] = 1.0
    r_mask = r_mask.repeat(1, 1, h // 2, w // 2)

    b_mask = torch.zeros((1, 1, 2, 2), dtype=dtype, device=device)
    b_mask[0, 0, 1, 1] = 1.0
    b_mask = b_mask.repeat(1, 1, h // 2, w // 2)

    g_mask = 1.0 - r_mask - b_mask

    r_raw = bayer_2d * r_mask
    g_raw = bayer_2d * g_mask
    b_raw = bayer_2d * b_mask

    # Interpolation kernels
    k_g = torch.tensor([[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]], dtype=dtype, device=device).view(1, 1, 3, 3)
    k_rb = torch.tensor([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]], dtype=dtype, device=device).view(1, 1, 3, 3)

    r_interp = F.conv2d(r_raw, k_rb, padding=1)
    g_interp = F.conv2d(g_raw, k_g, padding=1)
    b_interp = F.conv2d(b_raw, k_rb, padding=1)

    rgb = torch.cat([r_interp, g_interp, b_interp], dim=1)
    return torch.clamp(rgb, min=0.0, max=1.0)


def apply_ccm(rgb: torch.Tensor, ccm: torch.Tensor) -> torch.Tensor:
    """
    Applies 3x3 Color Correction Matrix to RGB image [B, 3, H, W].
    CCM transforms sensor RGB -> output sRGB (or inverse).
    """
    b, c, h, w = rgb.shape
    ccm = ccm.to(device=rgb.device, dtype=rgb.dtype)
    # Reshape rgb to [B, 3, H*W]
    flat = rgb.view(b, 3, -1)
    # Matrix multiply: [3, 3] x [B, 3, H*W] -> [B, 3, H*W]
    transformed = torch.matmul(ccm, flat)
    out = transformed.view(b, 3, h, w)
    return torch.clamp(out, min=0.0, max=1.0)


def forward_isp(
    bayer_2d: torch.Tensor,
    ccm: Optional[torch.Tensor] = None,
    gamma: float = 2.2
) -> torch.Tensor:
    """
    Forward camera ISP:
    Linear Bayer mosaic [B, 1, H, W] -> Demosaic -> CCM -> Gamma Tone Curve -> sRGB [B, 3, H, W].
    """
    if ccm is None:
        ccm = DEFAULT_CAM2SRGB

    # 1. Demosaic
    rgb_linear = demosaic_rggb(bayer_2d)

    # 2. Color Correction Matrix
    rgb_ccm = apply_ccm(rgb_linear, ccm)

    # 3. Tone Mapping / Gamma
    srgb = torch.clamp(rgb_ccm, min=0.0, max=1.0) ** (1.0 / gamma)
    return torch.clamp(srgb, min=0.0, max=1.0)


def inverse_isp_unprocess(
    srgb: torch.Tensor,
    ccm: Optional[torch.Tensor] = None,
    shot_noise_sigma: float = 0.01,
    read_noise_sigma: float = 0.005,
    gamma: float = 2.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverts camera ISP to generate paired clean and noisy linear Bayer raw data from sRGB:
    sRGB [B, 3, H, W] -> Inverse Gamma -> Inverse CCM -> Bayer Mosaic -> Poisson-Gaussian Noise.

    Returns:
        noisy_bayer: [B, 1, H, W] linear raw mosaic with heteroscedastic noise.
        clean_bayer: [B, 1, H, W] pristine linear raw mosaic ground truth.
    """
    if ccm is None:
        ccm = DEFAULT_CAM2SRGB

    # 1. Inverse Gamma
    linear_srgb = torch.clamp(srgb, min=0.0, max=1.0) ** gamma

    # 2. Inverse CCM
    inv_ccm = torch.inverse(ccm.to(device=srgb.device, dtype=srgb.dtype))
    linear_cam = apply_ccm(linear_srgb, inv_ccm)

    # 3. Bayer Mosaicing
    clean_bayer = mosaic_rggb(linear_cam)

    # 4. Calibrated Heteroscedastic Poisson-Gaussian Sensor Noise
    # Var(y|x) = shot_noise^2 * x + read_noise^2
    var_map = (shot_noise_sigma ** 2) * clean_bayer + (read_noise_sigma ** 2)
    std_map = torch.sqrt(torch.clamp(var_map, min=1e-8))
    noise = torch.randn_like(clean_bayer) * std_map

    noisy_bayer = torch.clamp(clean_bayer + noise, min=0.0, max=1.0)
    return noisy_bayer, clean_bayer
