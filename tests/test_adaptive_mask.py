"""
Unit tests for noise correlation estimation and scale-adaptive spectral masking.
"""

import pytest
import torch
from src.masking.noise_estimator import (
    detect_flat_regions,
    estimate_noise_autocorrelation,
    estimate_correlation_length,
    NoiseEstimator,
)
from src.masking.adaptive_mask import (
    generate_correlation_adaptive_mask,
    ScaleAdaptiveMasker,
    MagnitudeReconstructionNet,
)


def test_flat_region_and_autocorr():
    """Tests empirical autocorrelation on white noise vs correlated noise."""
    torch.manual_seed(42)
    b, c, h, w = 1, 1, 64, 64
    
    # Generate white noise
    white_noise = torch.randn(b, c, h, w)
    flat_mask = detect_flat_regions(white_noise)
    assert flat_mask.shape == (b, 1, h, w)

    kappa_white = estimate_noise_autocorrelation(white_noise, flat_mask=flat_mask, max_lag=3)
    assert kappa_white.shape == (b, c, 7, 7)
    # Peak at lag (0, 0) should be 1.0
    assert torch.allclose(kappa_white[:, :, 3, 3], torch.tensor([1.0]), atol=1e-3)
    # Off-diagonal lags for white noise should be near zero
    assert kappa_white[:, :, 0, 0].abs().item() < 0.3

    # Generate correlated noise (convolved with 5x5 blur kernel)
    blur_kernel = torch.ones(1, 1, 5, 5) / 25.0
    corr_noise = torch.nn.functional.conv2d(torch.randn(1, 1, 70, 70), blur_kernel, padding=0)[:, :, :64, :64]
    
    kappa_corr = estimate_noise_autocorrelation(corr_noise, max_lag=3)
    corr_len = estimate_correlation_length(kappa_corr)
    assert corr_len.item() >= 1.5, f"Expected correlation length >= 1.5, got {corr_len.item()}"


def test_adaptive_masker_and_reconstruction():
    """Tests scale-adaptive masking and magnitude reconstruction network."""
    torch.manual_seed(42)
    b, c = 2, 1
    mags = [
        torch.rand(b, c, 6, 32, 32),
        torch.rand(b, c, 6, 16, 16),
        torch.rand(b, c, 6, 8, 8)
    ]
    corr_len = torch.tensor([[3.0], [3.0]])  # Correlation length of 3 pixels

    masker = ScaleAdaptiveMasker(mask_ratio=0.2)
    masked_mags, mask_list, radii = masker(mags, corr_len)

    assert len(masked_mags) == 3
    # S=0 should have larger radius than S=2
    assert radii[0] >= radii[2]

    # Test reconstruction network
    pc = torch.rand(b, c, 64, 64)
    rec_net = MagnitudeReconstructionNet(in_channels=1, hidden_dim=32, n_scales=3)
    rec_mags = rec_net(masked_mags, pc)

    assert len(rec_mags) == 3
    for s in range(3):
        assert rec_mags[s].shape == mags[s].shape
        # Check non-negative magnitudes
        assert (rec_mags[s] >= 0.0).all()

    # Test loss computation & backpropagation
    loss = sum(torch.nn.functional.mse_loss(rec_mags[s], mags[s]) for s in range(3))
    loss.backward()
    assert rec_net.scale_blocks[0][0].weight.grad is not None
