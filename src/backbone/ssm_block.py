"""
Bidirectional Selective State Space Model (SSM / Mamba-style) Backbone.

Executes selective scan along the structure-guided (PC-warped) Hilbert sequence.
Optimized pure-PyTorch implementation with patch embedding for high speed and scale invariance.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pc_warping import warp_hilbert_trajectory, warped_flatten_2d, warped_unflatten_2d


class SelectiveSSM1D(nn.Module):
    """
    1D Selective State Space Layer (S6 core):
        h_t = exp(Delta_t * A) * h_{t-1} + (Delta_t * B_t) * x_t
        y_t = C_t * h_t + D * x_t
    """
    def __init__(self, d_model: int = 64, d_state: int = 16, dt_rank: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        x_dbl = self.x_proj(x)
        dt_raw, B_t, C_t = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        delta = F.softplus(self.dt_proj(dt_raw))
        A = -torch.exp(self.A_log.float())
        
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)).permute(1, 0, 2, 3).contiguous()
        dB_x = (delta.unsqueeze(-1) * B_t.unsqueeze(2) * x.unsqueeze(-1)).permute(1, 0, 2, 3).contiguous()
        C_perm = C_t.permute(1, 0, 2).unsqueeze(2).contiguous()
        x_perm = x.permute(1, 0, 2).contiguous()
        
        # Fast chunked scan
        chunk_size = 16
        if l % chunk_size == 0:
            n_chunks = l // chunk_size
            dA_c = dA.view(n_chunks, chunk_size, b, d, self.d_state)
            dB_c = dB_x.view(n_chunks, chunk_size, b, d, self.d_state)
            C_c = C_perm.view(n_chunks, chunk_size, b, 1, self.d_state)
            x_c = x_perm.view(n_chunks, chunk_size, b, d)
            
            h = torch.zeros((b, d, self.d_state), dtype=x.dtype, device=x.device)
            ys = []
            for k in range(n_chunks):
                dAk = dA_c[k]
                dBk = dB_c[k]
                Ck = C_c[k]
                xk = x_c[k]
                for t in range(chunk_size):
                    h = dAk[t] * h + dBk[t]
                    ys.append((h * Ck[t]).sum(dim=-1) + self.D * xk[t])
            return torch.stack(ys, dim=1)
        else:
            h = torch.zeros((b, d, self.d_state), dtype=x.dtype, device=x.device)
            ys = []
            for t in range(l):
                h = dA[t] * h + dB_x[t]
                y_t = (h * C_perm[t]).sum(dim=-1) + self.D * x_perm[t]
                ys.append(y_t)
            return torch.stack(ys, dim=1)


class BidirectionalSSMBlock(nn.Module):
    """
    Bidirectional SSM Block processing forward and reverse sequences.
    """
    def __init__(self, d_model: int = 64, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.fwd_ssm = SelectiveSSM1D(d_model=d_model, d_state=d_state)
        self.bwd_ssm = SelectiveSSM1D(d_model=d_model, d_state=d_state)
        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm(x)
        y_fwd = self.fwd_ssm(norm_x)
        x_rev = torch.flip(norm_x, dims=[1])
        y_bwd = torch.flip(self.bwd_ssm(x_rev), dims=[1])
        y_fused = self.out_proj(torch.cat([y_fwd, y_bwd], dim=-1))
        return x + y_fused


class StructureGuidedSSMBackbone(nn.Module):
    """
    Full 2D Structure-Guided SSM Backbone with patch embedding:
    1. Downsamples spatial map by patch_size (default 2) to accelerate scan.
    2. Flattens along PC-warped Hilbert scan path.
    3. Bidirectional selective state space modeling.
    4. Unflattens and upsamples back to original resolution with residual skip connection.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 128,
        d_state: int = 16,
        num_layers: int = 2,
        default_window: int = 8,
        patch_size: int = 2
    ):
        super().__init__()
        self.default_window = default_window
        self.patch_size = patch_size
        
        # Patch embedding
        self.in_proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        
        self.layers = nn.ModuleList([
            BidirectionalSSMBlock(d_model=d_model, d_state=d_state)
            for _ in range(num_layers)
        ])
        
        # Patch unembedding
        self.out_proj = nn.ConvTranspose2d(d_model, out_channels, kernel_size=patch_size, stride=patch_size)
        nn.init.kaiming_normal_(self.out_proj.weight, a=0.2, nonlinearity="leaky_relu")
        self.out_proj.weight.data.mul_(0.1)
        nn.init.zeros_(self.out_proj.bias)
        
        # Skip projection
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        pc: torch.Tensor,
        window_size: Optional[int] = None
    ) -> torch.Tensor:
        b, _, h, w = x.shape
        ws = self.default_window if window_size is None else window_size
        
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            pc_padded = F.pad(pc, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_padded = x
            pc_padded = pc

        h_pad, w_pad = x_padded.shape[-2:]
        h_p, w_p = h_pad // self.patch_size, w_pad // self.patch_size
        
        # Downscale PC prior to match patch grid
        pc_patch = F.interpolate(pc_padded, size=(h_p, w_p), mode="bilinear", align_corners=False)
        warped_indices = warp_hilbert_trajectory(pc_patch, window_size=ws)
        
        feat = self.in_proj(x_padded)  # [B, D, H_p, W_p]
        
        seq = warped_flatten_2d(feat, warped_indices).transpose(1, 2)
        for layer in self.layers:
            seq = layer(seq)
            
        feat_out = warped_unflatten_2d(seq.transpose(1, 2), h_p, w_p, warped_indices)
        out_padded = self.out_proj(feat_out)
        
        res_padded = self.skip(x_padded) + out_padded
        if pad_h > 0 or pad_w > 0:
            return res_padded[:, :, :h, :w]
        return res_padded
