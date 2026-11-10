"""
Physics-Informed Closed-Form Proximal Projection Safeguard.

Mode A: Closed-form Poisson-Gaussian proximal operator enforcing measurement consistency:
    x_hat = argmin_x  1/(2*sigma_raw^2) * (y - x)^2 + 1/(2*lambda) * (x - x_flow)^2
    Closed-form solution:
        x_hat = (lambda * y + sigma_raw^2 * x_flow) / (lambda + sigma_raw^2)

Mode B: Homoscedastic Gaussian proximal operator for sRGB/grayscale images.
"""

from typing import Optional
import torch
import torch.nn as nn
from .physics_schedule import estimate_raw_sensor_variance


def poisson_gaussian_proximal_projection(
    y_noisy: torch.Tensor,
    x_flow: torch.Tensor,
    shot_noise_sigma: float = 0.01,
    read_noise_sigma: float = 0.005,
    lambda_fidelity: float = 0.0
) -> torch.Tensor:
    """
    Mode A: Closed-form Poisson-Gaussian proximal safeguard.
    When lambda_fidelity == 0, returns x_flow directly.
    """
    if lambda_fidelity <= 0.0:
        return torch.clamp(x_flow, min=0.0)

    var_map = estimate_raw_sensor_variance(
        x_flow,
        shot_noise_sigma=shot_noise_sigma,
        read_noise_sigma=read_noise_sigma
    )
    weight_noisy = torch.clamp(lambda_fidelity / (lambda_fidelity + var_map + 1e-8), 0.0, 0.2)
    x_proj = weight_noisy * y_noisy + (1.0 - weight_noisy) * x_flow
    return torch.clamp(x_proj, min=0.0)


def homoscedastic_proximal_projection(
    y_noisy: torch.Tensor,
    x_flow: torch.Tensor,
    noise_sigma: float = 0.02,
    lambda_fidelity: float = 0.0
) -> torch.Tensor:
    """
    Mode B: Homoscedastic Gaussian proximal safeguard for sRGB / standard images.
    When lambda_fidelity == 0, returns x_flow directly.
    """
    if lambda_fidelity <= 0.0:
        return torch.clamp(x_flow, min=0.0, max=1.0)

    var = noise_sigma ** 2
    weight_noisy = min(0.2, lambda_fidelity / (lambda_fidelity + var + 1e-8))
    x_proj = weight_noisy * y_noisy + (1.0 - weight_noisy) * x_flow
    return torch.clamp(x_proj, min=0.0, max=1.0)


class PhysicsProximalOperator(nn.Module):
    """
    Applies mode-specific physics proximal safeguard to output of the flow integrator.
    """
    def __init__(
        self,
        mode: str = "raw",
        shot_noise_sigma: float = 0.01,
        read_noise_sigma: float = 0.005,
        noise_sigma: float = 0.02,
        lambda_fidelity: float = 0.0
    ):
        super().__init__()
        self.mode = mode
        self.shot_noise_sigma = shot_noise_sigma
        self.read_noise_sigma = read_noise_sigma
        self.noise_sigma = noise_sigma
        self.lambda_fidelity = lambda_fidelity

    def forward(self, y_noisy: torch.Tensor, x_flow: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return torch.clamp(x_flow, min=0.0, max=1.0)
        elif self.mode == "raw":
            return poisson_gaussian_proximal_projection(
                y_noisy=y_noisy,
                x_flow=x_flow,
                shot_noise_sigma=self.shot_noise_sigma,
                read_noise_sigma=self.read_noise_sigma,
                lambda_fidelity=self.lambda_fidelity
            )
        elif self.mode == "srgb":
            return homoscedastic_proximal_projection(
                y_noisy=y_noisy,
                x_flow=x_flow,
                noise_sigma=self.noise_sigma,
                lambda_fidelity=self.lambda_fidelity
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
