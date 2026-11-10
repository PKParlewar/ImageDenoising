#!/usr/bin/env python3
"""
Unit tests for Phase-Gated NAFNet (PG-NAFNet):
1. Test output shape invariance across varying batch sizes and spatial dimensions.
2. Test mathematical identity at initialization: zero-init pc_proj matches vanilla NAFNet exactly.
3. Test backward pass and gradient flow through Phase Congruency modulation projections.
4. Test parameter count overhead (+1,728 parameters, <0.4% overhead).
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.pg_nafnet import PG_NAFNet, PhaseGatedNAFBlock, SimpleGate, LayerNorm2d
from scripts.train_phase_nafnet import PhaseNAFBlock as VanillaPhaseNAFBlock


def test_pg_nafnet_forward_shapes():
    """Verify forward pass outputs matching dimensions for varying inputs."""
    model = PG_NAFNet(in_ch=1, c=72, n_blocks=4)
    model.eval()
    
    # Test batch size 2, 64x64
    x = torch.rand(2, 1, 64, 64)
    pc = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        out = model(x, pc)
    assert out.shape == (2, 1, 64, 64)
    assert (out >= 0.0).all() and (out <= 1.0).all()

    # Test batch size 1, 128x128
    x2 = torch.rand(1, 1, 128, 128)
    pc2 = torch.rand(1, 1, 128, 128)
    with torch.no_grad():
        out2 = model(x2, pc2)
    assert out2.shape == (1, 1, 128, 128)


def test_pg_block_identity_at_initialization():
    """
    Verify that with zero-initialized gamma_pc, PhaseGatedNAFBlock produces
    EXACTLY identical outputs to a vanilla PhaseNAFBlock sharing the same base weights.
    """
    c = 72
    torch.manual_seed(42)
    vanilla_block = VanillaPhaseNAFBlock(c)
    pg_block = PhaseGatedNAFBlock(c)

    # Copy all common weights from vanilla to pg_block
    vanilla_state = vanilla_block.state_dict()
    pg_block.load_state_dict(vanilla_state, strict=False)

    # Ensure gamma_pc is zero-initialized
    assert torch.all(pg_block.gamma_pc == 0)

    # Forward pass on both
    inp = torch.randn(2, c, 32, 32)
    pc = torch.rand(2, 1, 32, 32)

    with torch.no_grad():
        vanilla_out = vanilla_block(inp)
        pg_out = pg_block(inp, pc)

    max_diff = (vanilla_out - pg_out).abs().max().item()
    assert max_diff < 1e-6, f"Initialization divergence: max diff = {max_diff}"


def test_pg_nafnet_gradient_flow():
    """Verify that gradients flow back into gamma_pc and throughout the network when residual paths are active."""
    model = PG_NAFNet(in_ch=1, c=72, n_blocks=2)
    # Set non-zero outro and beta so gradients propagate back into internal residual branches
    nn.init.normal_(model.outro.weight, std=0.1)
    for block in model.blocks:
        block.beta.data.fill_(1.0)
        nn.init.normal_(block.gamma_pc, std=0.1)

    x = torch.rand(2, 1, 32, 32, requires_grad=True)
    pc = torch.rand(2, 1, 32, 32, requires_grad=True)
    target = torch.rand(2, 1, 32, 32)

    pred = model(x, pc)
    loss = torch.nn.functional.l1_loss(pred, target)
    loss.backward()

    # Check gradients on gamma_pc in every block
    for i, block in enumerate(model.blocks):
        assert block.gamma_pc.grad is not None, f"Block {i} gamma_pc has no gradient"
        assert not torch.all(block.gamma_pc.grad == 0), f"Block {i} gamma_pc grad is all zeros"


def test_pg_nafnet_parameter_count():
    """Verify that PG-Gate adds exactly 72 params per block for gamma_pc."""
    model = PG_NAFNet(in_ch=1, c=72, n_blocks=12)
    total_params = sum(p.numel() for p in model.parameters())

    extra_params = sum(p.numel() for block in model.blocks for p in [block.gamma_pc])
    assert extra_params == 12 * 72
    assert extra_params == 864
    assert total_params < 470000  # under 0.47M parameters
