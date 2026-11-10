"""
Unit tests for Dual-Tree Complex Wavelet Transform (DTCWT).
"""

import pytest
import torch
from src.wavelets.dtcwt import (
    DTCWTForward,
    DTCWTInverse,
    complex_to_mag_phase,
    mag_phase_to_complex,
)


def test_dtcwt_forward_inverse_reconstruction():
    """Tests that forward -> inverse DTCWT achieves exact reconstruction."""
    torch.manual_seed(42)
    b, c, h, w = 2, 3, 64, 64
    x = torch.randn(b, c, h, w, dtype=torch.float32)

    fwd = DTCWTForward(n_scales=2)
    inv = DTCWTInverse()

    c_low, highs = fwd(x)
    assert len(highs) == 2
    # Check shape of highs: [B, C, 6, H_s, W_s, 2]
    assert highs[0].shape == (b, c, 6, 32, 32, 2)
    assert highs[1].shape == (b, c, 6, 16, 16, 2)
    assert c_low.shape == (b, c, 32, 32)

    x_rec = inv(c_low, highs)
    assert x_rec.shape == x.shape

    # Exact max reconstruction error across entire image
    diff = torch.abs(x - x_rec)
    max_err = diff.max().item()
    assert max_err < 1e-5, f"Reconstruction error {max_err} exceeded threshold"


def test_complex_mag_phase_conversion():
    """Tests round-trip conversion between complex (real, imag) and (mag, phase)."""
    torch.manual_seed(42)
    high_complex = torch.randn(2, 3, 6, 32, 32, 2)

    mag, phase = complex_to_mag_phase(high_complex)
    assert mag.min() >= 0.0

    complex_rec = mag_phase_to_complex(mag, phase)
    assert torch.allclose(high_complex, complex_rec, atol=1e-5)
