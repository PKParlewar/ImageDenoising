"""
Scale-Adaptive Spectral Masking and Phase-Conditioned Self-Supervision Module.

Step 2 & Step 3 of Section 2 in info.md:
Sizes the spectral mask radius adaptively per scale:
    r^{s,d} = g(kappa_hat, s)
and reconstructs masked magnitude conditioned on the invariant phase channel:
    L_mask = E[ || A^{s,d} - f_theta(A_hat^{s,d}, PC) ||^2 ]
"""

from typing import List, Tuple, Dict, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.wavelets.dtcwt import DTCWTForward, DTCWTInverse, complex_to_mag_phase, mag_phase_to_complex
from src.wavelets.phase_congruency import PhaseCongruencyCalculator


def generate_correlation_adaptive_mask(
    shape: Tuple[int, int],
    mask_radius: int,
    mask_ratio: float = 0.2,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Generates a 2D binary spatial blind-spot mask with circular exclusion zones of radius r.
    
    Args:
        shape: (H, W) spatial dimensions of the subband.
        mask_radius: Radius of the blind spot around each masked center.
        mask_ratio: Fraction of pixels to mask.
        device: PyTorch device.
        
    Returns:
        mask: [1, 1, H, W] binary mask where 1 indicates masked (blind-spot) pixels.
    """
    h, w = shape
    # Grid stride based on mask radius to prevent overlapping blind spots
    stride = max(1, 2 * mask_radius + 1)
    
    # Bernoulli sampling on subsampled grid
    sub_h = max(1, h // stride)
    sub_w = max(1, w // stride)
    
    rand_grid = torch.rand((1, 1, sub_h, sub_w), device=device)
    centers = (rand_grid < mask_ratio).float()
    
    # Upsample centers with dilated kernel / max pooling to create exclusion radius
    kernel_size = 2 * mask_radius + 1
    # Create circular structuring element
    y_coords, x_coords = torch.meshgrid(
        torch.arange(kernel_size, device=device) - mask_radius,
        torch.arange(kernel_size, device=device) - mask_radius,
        indexing="ij"
    )
    circle_kernel = (x_coords ** 2 + y_coords ** 2 <= mask_radius ** 2).float().unsqueeze(0).unsqueeze(0)
    
    # Upsample centers to full resolution
    centers_full = F.interpolate(centers, size=(h, w), mode="nearest")
    
    # Dilate centers by circle kernel
    pad = mask_radius
    dilated = F.conv2d(
        F.pad(centers_full, (pad, pad, pad, pad), mode="replicate"),
        circle_kernel,
        padding=0
    )
    mask = (dilated > 0).float()
    return mask


class ScaleAdaptiveMasker(nn.Module):
    """
    Applies scale-adaptive spectral masking to DTCWT magnitude subbands.
    """
    def __init__(self, mask_ratio: float = 0.2):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(
        self,
        mags: List[torch.Tensor],
        corr_lengths: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
        """
        Args:
            mags: List of S magnitude tensors, each of shape [B, C, 6, H_s, W_s].
            corr_lengths: [B, C] estimated raw correlation length in pixels.
            
        Returns:
            masked_mags: List of S masked magnitude tensors.
            mask_list: List of S binary masks [B, 1, 1, H_s, W_s].
            radii_per_scale: List of computed radii per scale s.
        """
        masked_mags = []
        mask_list = []
        radii_per_scale = []
        
        # Mean correlation length across batch and channels
        mean_corr = corr_lengths.mean().item()
        
        for s_idx, mag_s in enumerate(mags):
            # Scale s: subsampling factor is 2^s
            scale_factor = 2 ** s_idx
            # r^{s,d} = max(1, ceil(L_corr / 2^s))
            radius_s = max(1, int(math.ceil(mean_corr / scale_factor)))
            radii_per_scale.append(radius_s)
            
            b, c, n_dirs, h_s, w_s = mag_s.shape
            mask_s = generate_correlation_adaptive_mask(
                shape=(h_s, w_s),
                mask_radius=radius_s,
                mask_ratio=self.mask_ratio,
                device=mag_s.device
            ) # [1, 1, H_s, W_s]
            
            mask_s_expanded = mask_s.unsqueeze(2).repeat(b, c, n_dirs, 1, 1)  # [B, C, 6, H_s, W_s]
            
            # Mask out magnitude (replace with local average of unmasked points or zero)
            masked_mag_s = mag_s * (1.0 - mask_s_expanded)
            
            masked_mags.append(masked_mag_s)
            mask_list.append(mask_s_expanded)
            
        return masked_mags, mask_list, radii_per_scale


class MagnitudeReconstructionNet(nn.Module):
    """
    Phase-Conditioned Magnitude Denoiser f_theta(A_hat, PC).
    Reconstructs masked magnitude coefficients conditioned on neighboring subbands
    and the noise-invariant Phase Congruency channel.
    """
    def __init__(self, in_channels: int = 1, hidden_dim: int = 48, n_scales: int = 3):
        super().__init__()
        self.n_scales = n_scales
        self.in_channels = in_channels
        
        # Condition encoder for Phase Congruency map (native resolution)
        self.pc_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, hidden_dim // 2), hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, hidden_dim // 2), hidden_dim // 2),
        )
        
        # Subband reconstruction branches per scale
        self.scale_blocks = nn.ModuleList()
        for s in range(n_scales):
            # Input: 6 directional subband magnitudes + downsampled PC features
            in_dim = in_channels * 6 + hidden_dim // 2
            last_conv = nn.Conv2d(hidden_dim, in_channels * 6, kernel_size=3, padding=1)
            nn.init.kaiming_normal_(last_conv.weight, a=0.2, nonlinearity="leaky_relu")
            last_conv.weight.data.mul_(0.1)
            nn.init.zeros_(last_conv.bias)
            block = nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                last_conv,
            )
            self.scale_blocks.append(block)

    def forward(
        self,
        masked_mags: List[torch.Tensor],
        pc: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Args:
            masked_mags: List of S masked magnitude tensors [B, C, 6, H_s, W_s].
            pc: Phase Congruency map [B, C, H, W].
            
        Returns:
            reconstructed_mags: List of S reconstructed magnitude tensors [B, C, 6, H_s, W_s].
        """
        pc_feat = self.pc_encoder(pc)
        reconstructed_mags = []
        
        for s_idx, mag_s in enumerate(masked_mags):
            b, c, n_dirs, h_s, w_s = mag_s.shape
            # Flatten [B, C, 6, H_s, W_s] to [B, C*6, H_s, W_s]
            mag_s_flat = mag_s.view(b, c * n_dirs, h_s, w_s)
            
            # Downsample PC feature map to scale s resolution
            pc_s = F.interpolate(pc_feat, size=(h_s, w_s), mode="bilinear", align_corners=False)
            
            # Concatenate along channel dimension
            inp_s = torch.cat([mag_s_flat, pc_s], dim=1)
            
            # Residual reconstruction
            delta_s_flat = self.scale_blocks[s_idx](inp_s)
            rec_s_flat = F.softplus(mag_s_flat + 0.1 * delta_s_flat, beta=4.0)
            rec_s = rec_s_flat.view(b, c, n_dirs, h_s, w_s)
            reconstructed_mags.append(rec_s)
            
        return reconstructed_mags
