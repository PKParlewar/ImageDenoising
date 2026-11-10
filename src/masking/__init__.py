"""
Noise correlation estimation and adaptive spectral masking modules.
"""

from .noise_estimator import (
    detect_flat_regions,
    estimate_noise_autocorrelation,
    estimate_correlation_length,
    NoiseEstimator,
)
from .adaptive_mask import (
    ScaleAdaptiveMasker,
    generate_correlation_adaptive_mask,
)

__all__ = [
    "detect_flat_regions",
    "estimate_noise_autocorrelation",
    "estimate_correlation_length",
    "NoiseEstimator",
    "ScaleAdaptiveMasker",
    "generate_correlation_adaptive_mask",
]
