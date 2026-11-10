"""
Unit tests for Phase-Domain Rectified Flow and Physics Calibration.
"""

import pytest
import torch
from src.flow.physics_schedule import (
    PhysicsScheduler,
    estimate_raw_sensor_variance,
    estimate_srgb_residual_variance,
)
from src.flow.proximal import (
    PhysicsProximalOperator,
    poisson_gaussian_proximal_projection,
    homoscedastic_proximal_projection,
)
from src.flow.rectified_flow import (
    DTCWTRectifiedFlow,
    AsymmetricVelocityNet,
)


def test_physics_schedulers():
    """Tests Mode A and Mode B physics schedulers."""
    x = torch.rand(2, 1, 32, 32)
    
    # Mode A (Raw sensor)
    sched_raw = PhysicsScheduler(mode="raw", shot_noise_sigma=0.01, read_noise_sigma=0.005)
    scale_raw = sched_raw(x)
    assert scale_raw.shape == (2, 1, 32, 32)
    assert scale_raw.min() >= 0.0

    # Mode B (sRGB)
    sched_srgb = PhysicsScheduler(mode="srgb")
    scale_srgb = sched_srgb(x)
    assert scale_srgb.shape == (2, 1, 32, 32)
    assert scale_srgb.min() >= 0.0


def test_proximal_safeguards():
    """Tests Poisson-Gaussian and Gaussian closed-form proximal projection operators."""
    y = torch.rand(2, 1, 32, 32)
    x_flow = torch.rand(2, 1, 32, 32)

    # Raw Mode
    x_raw_proj = poisson_gaussian_proximal_projection(y, x_flow, lambda_fidelity=0.05)
    assert x_raw_proj.shape == y.shape
    assert (x_raw_proj >= 0.0).all()

    # sRGB Mode
    x_srgb_proj = homoscedastic_proximal_projection(y, x_flow, lambda_fidelity=0.05)
    assert x_srgb_proj.shape == y.shape
    assert (x_srgb_proj >= 0.0).all() and (x_srgb_proj <= 1.0).all()


def test_rectified_flow_training_and_sampling():
    """Tests flow loss gradient and few-step sampling."""
    torch.manual_seed(42)
    b, c, h, w = 2, 1, 32, 32
    y_noisy = torch.rand(b, c, h, w)
    x_clean = torch.rand(b, c, h, w)

    flow = DTCWTRectifiedFlow(in_channels=1, hidden_dim=32, n_scales=2, physics_mode="srgb")

    # Forward loss
    loss_dict = flow.compute_flow_loss(y_noisy, x_clean)
    assert "loss" in loss_dict
    loss_dict["loss"].backward()

    # Check gradients
    assert flow.velocity_net.scale_blocks[0][0].weight.grad is not None

    # Sampling test (2 steps with Euler and Heun)
    denoised_euler = flow.denoise_sample(y_noisy, num_steps=2, solver="euler")
    denoised_heun = flow.denoise_sample(y_noisy, num_steps=2, solver="heun")
    assert denoised_euler.shape == y_noisy.shape
    assert denoised_heun.shape == y_noisy.shape
