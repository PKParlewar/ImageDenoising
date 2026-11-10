"""
Unit tests for Hilbert scanning, PC-guided warping, and SSM backbone.
"""

import pytest
import torch
from src.backbone.hilbert_scan import (
    generate_hilbert_curve_2d,
    hilbert_flatten_2d,
    hilbert_unflatten_2d,
)
from src.backbone.pc_warping import (
    compute_traversal_cost,
    warp_hilbert_trajectory,
    warped_flatten_2d,
    warped_unflatten_2d,
    analyze_warping_tradeoff,
)
from src.backbone.ssm_block import (
    BidirectionalSSMBlock,
    StructureGuidedSSMBackbone,
)


def test_hilbert_bijection_and_reconstruction():
    """Tests that Hilbert curve covers every pixel exactly once and unflattening is lossless."""
    h, w = 32, 32
    y_idx, x_idx = generate_hilbert_curve_2d(h, w)
    
    assert len(y_idx) == h * w
    assert len(x_idx) == h * w
    
    # Check all coordinate pairs (y, x) are unique
    coords = set((y_idx[i].item(), x_idx[i].item()) for i in range(h * w))
    assert len(coords) == h * w

    # Test round-trip flatten -> unflatten
    x = torch.randn(2, 3, h, w)
    seq, _ = hilbert_flatten_2d(x)
    assert seq.shape == (2, 3, h * w)

    x_rec = hilbert_unflatten_2d(seq, h, w, y_idx, x_idx)
    assert torch.allclose(x, x_rec)


def test_pc_warped_bijection_across_windows():
    """Tests that PC-warping produces a valid bijection for all window sizes w."""
    h, w = 16, 16
    pc = torch.rand(1, 1, h, w)
    
    for ws in [0, 2, 4, 8, 16]:
        warped_indices = warp_hilbert_trajectory(pc, window_size=ws)
        assert len(warped_indices) == h * w
        # All indices in 0..h*w-1 must be present exactly once
        unique_indices = torch.unique(warped_indices)
        assert len(unique_indices) == h * w

        # Test flatten / unflatten roundtrip
        x = torch.randn(1, 2, h, w)
        seq = warped_flatten_2d(x, warped_indices)
        x_rec = warped_unflatten_2d(seq, h, w, warped_indices)
        assert torch.allclose(x, x_rec, atol=1e-5)


def test_ssm_backbone_gradient():
    """Tests forward pass and gradient flow through StructureGuidedSSMBackbone."""
    torch.manual_seed(42)
    b, c, h, w = 1, 1, 16, 16
    x = torch.randn(b, c, h, w, requires_grad=True)
    pc = torch.rand(b, 1, h, w)

    backbone = StructureGuidedSSMBackbone(
        in_channels=1,
        out_channels=1,
        d_model=32,
        d_state=8,
        num_layers=1,
        default_window=4
    )
    
    out = backbone(x, pc)
    assert out.shape == (b, c, h, w)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_warping_tradeoff_sweep():
    """Tests Task 3.4 tradeoff analysis functionality."""
    pc = torch.rand(1, 1, 16, 16)
    results = analyze_warping_tradeoff(pc, window_sizes=[0, 2, 4])
    assert len(results["window_sizes"]) == 3
    assert len(results["mean_costs"]) == 3
    assert len(results["mean_step_distances"]) == 3
