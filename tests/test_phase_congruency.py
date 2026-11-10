"""
Unit tests for Phase Congruency calculator.
"""

import pytest
import torch
from src.wavelets.phase_congruency import PhaseCongruencyCalculator, measure_pc_snr_breakdown


def test_phase_congruency_range_and_stability():
    """Tests that Phase Congruency values are normalized in [0, 1] and gradient flows cleanly."""
    torch.manual_seed(42)
    b, c, h, w = 1, 1, 64, 64
    x = torch.rand(b, c, h, w, requires_grad=True)

    pc_calc = PhaseCongruencyCalculator(n_scales=2)
    pc = pc_calc(x)

    assert pc.shape == (b, c, h, w)
    assert pc.min().item() >= 0.0
    assert pc.max().item() <= 1.0

    # Test backprop
    loss = pc.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_pc_snr_sweep():
    """Tests the SNR breakdown measurement harness."""
    torch.manual_seed(42)
    clean_img = torch.rand(1, 1, 64, 64)
    snr_range = [20.0, 10.0, 0.0]

    sweep_res = measure_pc_snr_breakdown(
        clean_img,
        snr_db_range=snr_range,
        n_scales=2
    )

    assert len(sweep_res["pc_mse_list"]) == len(snr_range)
    # Higher noise (lower SNR) should generally produce higher PC error
    assert sweep_res["pc_mse_list"][-1] >= sweep_res["pc_mse_list"][0]
