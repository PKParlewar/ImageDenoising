"""
Unit tests for differentiable raw Bayer and ISP pipeline utilities.
"""

import pytest
import torch
from src.utils.raw_utils import (
    unpack_raw_bayer,
    pack_raw_bayer,
    normalize_raw_frame,
    mosaic_rggb,
    demosaic_rggb,
    apply_ccm,
    forward_isp,
    inverse_isp_unprocess,
    DEFAULT_CAM2SRGB,
    GOOGLE_PIXEL_CAM2SRGB,
    SAMSUNG_S6_CAM2SRGB,
)


def test_packing_unpacking_consistency():
    bayer = torch.rand(2, 1, 64, 64)
    for pattern in ["RGGB", "BGGR", "GRBG", "GBRG"]:
        unpacked = unpack_raw_bayer(bayer, pattern=pattern)
        assert unpacked.shape == (2, 4, 32, 32)
        packed = pack_raw_bayer(unpacked, pattern=pattern)
        assert packed.shape == bayer.shape
        assert torch.allclose(packed, bayer, atol=1e-6)


def test_demosaic_shape_and_range():
    bayer = torch.rand(2, 1, 64, 64)
    rgb = demosaic_rggb(bayer)
    assert rgb.shape == (2, 3, 64, 64)
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0


def test_mosaic_rggb():
    rgb = torch.rand(2, 3, 64, 64)
    bayer = mosaic_rggb(rgb)
    assert bayer.shape == (2, 1, 64, 64)
    # Check RGGB alignment
    assert torch.allclose(bayer[:, 0, 0::2, 0::2], rgb[:, 0, 0::2, 0::2])
    assert torch.allclose(bayer[:, 0, 0::2, 1::2], rgb[:, 1, 0::2, 1::2])
    assert torch.allclose(bayer[:, 0, 1::2, 0::2], rgb[:, 1, 1::2, 0::2])
    assert torch.allclose(bayer[:, 0, 1::2, 1::2], rgb[:, 2, 1::2, 1::2])


def test_forward_inverse_isp_pipeline():
    srgb = torch.rand(2, 3, 64, 64)
    noisy_bayer, clean_bayer = inverse_isp_unprocess(
        srgb,
        ccm=DEFAULT_CAM2SRGB,
        shot_noise_sigma=0.005,
        read_noise_sigma=0.002
    )
    assert noisy_bayer.shape == (2, 1, 64, 64)
    assert clean_bayer.shape == (2, 1, 64, 64)
    assert noisy_bayer.min() >= 0.0 and noisy_bayer.max() <= 1.0
    assert clean_bayer.min() >= 0.0 and clean_bayer.max() <= 1.0

    # Test forward ISP on clean bayer
    recon_srgb = forward_isp(clean_bayer, ccm=DEFAULT_CAM2SRGB)
    assert recon_srgb.shape == srgb.shape
    assert recon_srgb.min() >= 0.0 and recon_srgb.max() <= 1.0


def test_demosaic_gradient_flow():
    bayer = torch.rand(2, 1, 32, 32, requires_grad=True)
    rgb = demosaic_rggb(bayer)
    loss = rgb.sum()
    loss.backward()
    assert bayer.grad is not None
    assert not torch.isnan(bayer.grad).any()
