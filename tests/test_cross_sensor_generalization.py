"""
Task 2.5: Cross-sensor and multi-noise kernel generalization test.

Verifies that the empirical autocorrelation estimator and scale-adaptive masking engine
automatically adjust to varying noise correlation kernels (k=1, 3, 5, 7) without hand-tuned strides.
"""

import pytest
import torch
import torch.nn.functional as F
from src.masking.noise_estimator import NoiseEstimator
from src.masking.adaptive_mask import ScaleAdaptiveMasker


def test_cross_sensor_adaptation():
    estimator = NoiseEstimator(max_lag=7, decay_threshold=0.2)
    masker = ScaleAdaptiveMasker(mask_ratio=0.2)

    kernel_sizes = [1, 3, 5, 7]
    measured_corr_lengths = []
    fine_scale_radii = []

    for k in kernel_sizes:
        # Generate correlated noise by convolving white noise with box/Gaussian filter of size k
        if k == 1:
            noise = torch.randn(1, 1, 64, 64)
        else:
            kernel = torch.ones(1, 1, k, k) / (k * k)
            noise = F.conv2d(torch.randn(1, 1, 64 + k, 64 + k), kernel, padding=0)[:, :, :64, :64]

        kappa_hat, corr_len = estimator(noise)
        measured_corr_lengths.append(corr_len.item())

        mags = [torch.rand(1, 1, 6, 32, 32), torch.rand(1, 1, 6, 16, 16)]
        _, _, radii = masker(mags, corr_len)
        fine_scale_radii.append(radii[0])

    # Verify monotonic relationship: larger noise correlation kernel -> larger measured correlation length
    for i in range(len(kernel_sizes) - 1):
        assert measured_corr_lengths[i] <= measured_corr_lengths[i + 1], (
            f"Expected correlation length to increase with kernel size: "
            f"{measured_corr_lengths[i]} > {measured_corr_lengths[i+1]}"
        )
        assert fine_scale_radii[i] <= fine_scale_radii[i + 1]
