"""
2D Dual-Tree Complex Wavelet Transform (DTCWT) and Inversion in PyTorch.

Wraps Kingsbury's Dual-Tree Complex Wavelet Transform with:
- Directionally-selective subbands (6 directional orientations: ±15°, ±45°, ±75°)
- Exact perfect reconstruction (L_infinity error < 1e-6)
- Conversions between complex coefficients and (magnitude A, phase phi)
- Support for arbitrary batch, channel, and spatial dimensions
"""

from typing import List, Tuple, Union, Optional
import torch
import torch.nn as nn
from pytorch_wavelets import DTCWTForward as _PyWDTCWTForward, DTCWTInverse as _PyWDTCWTInverse


class DTCWTForward(nn.Module):
    """
    DTCWT Forward Transform.
    Input:
        x: [B, C, H, W]
    Returns:
        c_low: [B, C, H/(2^J), W/(2^J)]
        highs: List of J complex tensors of shape [B, C, 6, H/(2^s), W/(2^s), 2]
               where last dim is [real, imag].
    """
    def __init__(self, n_scales: int = 3, biort: str = "near_sym_a", qshift: str = "qshift_a"):
        super().__init__()
        self.n_scales = n_scales
        self.fwd = _PyWDTCWTForward(J=n_scales, biort=biort, qshift=qshift, o_dim=2, ri_dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Handles potential padding for odd dimensions if needed
        b, c, h, w = x.shape
        pad_h = (2 ** self.n_scales - (h % (2 ** self.n_scales))) % (2 ** self.n_scales)
        pad_w = (2 ** self.n_scales - (w % (2 ** self.n_scales))) % (2 ** self.n_scales)
        
        if pad_h > 0 or pad_w > 0:
            x_padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_padded = x

        c_low, highs = self.fwd(x_padded)
        return c_low, highs


class DTCWTInverse(nn.Module):
    """
    DTCWT Inverse Transform.
    Input:
        c_low: [B, C, H_low, W_low]
        highs: List of J tensors [B, C, 6, H_s, W_s, 2]
    Returns:
        x_rec: Reconstructed tensor [B, C, H, W]
    """
    def __init__(self, biort: str = "near_sym_a", qshift: str = "qshift_a"):
        super().__init__()
        self.inv = _PyWDTCWTInverse(biort=biort, qshift=qshift, o_dim=2, ri_dim=-1)

    def forward(self, c_low: torch.Tensor, highs: List[torch.Tensor], orig_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        x_rec = self.inv((c_low, highs))
        if orig_size is not None:
            x_rec = x_rec[:, :, :orig_size[0], :orig_size[1]]
        return x_rec


def complex_to_mag_phase(
    high_complex: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts complex subband tensor [B, C, 6, H_s, W_s, 2]
    into magnitude A [B, C, 6, H_s, W_s] and phase phi [B, C, 6, H_s, W_s].
    """
    real = high_complex[..., 0]
    imag = high_complex[..., 1]
    mag = torch.sqrt(real ** 2 + imag ** 2 + eps)
    phase = torch.atan2(imag, real)
    return mag, phase


def mag_phase_to_complex(
    mag: torch.Tensor,
    phase: torch.Tensor
) -> torch.Tensor:
    """
    Converts magnitude A [B, C, 6, H_s, W_s] and phase phi [B, C, 6, H_s, W_s]
    into complex subband tensor [B, C, 6, H_s, W_s, 2].
    """
    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    return torch.stack([real, imag], dim=-1)
