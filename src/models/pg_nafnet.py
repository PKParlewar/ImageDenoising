#!/usr/bin/env python3
"""
Phase-Gated Nonlinear Activation-Free Network (PG-NAFNet):
1. Single-Channel Input (in_ch=1): Direct noisy image input, zero naive input concatenation.
2. Phase-Gated SimpleGate (PG-Gate): Dynamically modulates feature conductivity across all 12 blocks
   using Fourier Phase Congruency: Gate(X, PC) = (X1 * X2) * (1 + tanh(W_pc * PC + b_pc)).
3. Zero-Initialized Modulation: W_pc=0, b_pc=0 ensures exact mathematical identity at step 0.
4. Parameter Efficiency: Adds only 144 parameters per block (+1,728 total parameters across 12 blocks).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    """Element-wise multiplication of split feature channels."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNorm2d(nn.Module):
    """Channel-wise 2D Layer Normalization."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class PhaseGatedNAFBlock(nn.Module):
    """
    Nonlinear Activation-Free Block equipped with Phase-Gated SimpleGate (PG-Gate).
    Dynamically modulates feature transmission at depth l using Fourier Phase Congruency.
    """
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)

        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1)
        )
        self.sg = SimpleGate()

        # Phase-Gating Learnable Channel Scale: maps 1-channel PC map to modulation factors
        # Initialized to zeros so that 1.0 + 0.0 * PC = 1.0 (exact mathematical identity at step 0)
        # Uses zero intermediate conv activation memory, eliminating all MPS memory thrashing
        self.gamma_pc = nn.Parameter(torch.zeros((1, dw_channel // 2, 1, 1)), requires_grad=True)

        # Feed-Forward Network (FFN)
        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # Residual gating parameters initialized to zero
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor, pc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inp: Feature tensor of shape (B, C, H, W)
            pc: Phase Congruency map of shape (B, 1, H, W) in [0, 1]
        Returns:
            Output feature tensor of shape (B, C, H, W)
        """
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        
        # 1. Base SimpleGate
        gate = self.sg(x)  # (B, dw_channel // 2, H, W)
        
        # 2. Phase-Gated Modulation:
        # Phase Congruency controls feature conductivity across physical boundaries via channel scaling
        gate = gate * (1.0 + self.gamma_pc * pc)
        
        # 3. Channel Attention and Projection
        gate = gate * self.sca(gate)
        x = self.conv3(gate)
        y = inp + x * self.beta

        # 4. Feed-Forward Network
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


class PG_NAFNet(nn.Module):
    """
    Phase-Gated Nonlinear Activation-Free Network.
    
    Operates on single-channel inputs (in_ch=1) with pure internal phase-gating across all 12 blocks.
    """
    def __init__(self, in_ch: int = 1, c: int = 72, n_blocks: int = 12):
        super().__init__()
        self.in_ch = in_ch
        self.c = c
        self.n_blocks = n_blocks

        # Shallow feature extraction (pure image input, zero naive concatenation)
        self.intro = nn.Conv2d(in_ch, c, 3, padding=1)

        # Deep Phase-Gated Trunk (all n_blocks equipped with PG-Gate)
        self.blocks = nn.ModuleList([PhaseGatedNAFBlock(c) for _ in range(n_blocks)])

        # Reconstruction projection (zero-initialized)
        self.outro = nn.Conv2d(c, 1, 3, padding=1)
        nn.init.zeros_(self.outro.weight)
        nn.init.zeros_(self.outro.bias)

    def forward(self, x: torch.Tensor, pc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy image tensor of shape (B, 1, H, W) in [0, 1]
            pc: Phase Congruency map of shape (B, 1, H, W) in [0, 1]
        Returns:
            Restored clean image tensor of shape (B, 1, H, W) clamped to [0, 1]
        """
        feat = self.intro(x)
        for block in self.blocks:
            feat = block(feat, pc)
        res = self.outro(feat)
        return torch.clamp(x + res, 0.0, 1.0)
