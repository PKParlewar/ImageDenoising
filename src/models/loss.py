"""
Composite Loss Functions for PC-SSFlow v2.

Implements:
- L_mask: Magnitude reconstruction loss on masked spectral coefficients.
- L_PC: Phase-congruency invariance self-supervision loss.
- L_Flow: Asymmetric rectified flow matching loss.
- Composite combined training loss.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PCSSFlowLoss(nn.Module):
    """
    Composite training loss for self-supervised bootstrap and flow stages.
    """
    def __init__(
        self,
        lambda_mask: float = 1.0,
        lambda_pc: float = 0.05,
        lambda_reg: float = 0.001,
        weight_mag: float = 1.0,
        weight_phase: float = 0.05
    ):
        super().__init__()
        self.lambda_mask = lambda_mask
        self.lambda_pc = lambda_pc
        self.lambda_reg = lambda_reg
        self.weight_mag = weight_mag
        self.weight_phase = weight_phase

    def self_supervised_loss(
        self,
        rec_mags: List[torch.Tensor],
        orig_mags: List[torch.Tensor],
        masks: List[torch.Tensor],
        pc_pred: torch.Tensor,
        pc_orig: torch.Tensor,
        denoised_img: torch.Tensor,
        y_noisy: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes L_self = L_mask + lambda_1 * L_PC + lambda_fid * L_fid + lambda_2 * L_reg
        """
        device = pc_pred.device
        loss_mask = torch.tensor(0.0, device=device)

        # L_mask only computed on masked positions
        for rec_s, orig_s, mask_s in zip(rec_mags, orig_mags, masks):
            diff_sq = (rec_s - orig_s) ** 2
            mask_sum = mask_s.sum().clamp(min=1.0)
            loss_mask = loss_mask + (diff_sq * mask_s).sum() / mask_sum

        # L_PC: Phase congruency invariance L1 loss
        loss_pc = F.l1_loss(pc_pred, pc_orig)

        # Spatial fidelity anchor
        loss_fid = torch.tensor(0.0, device=device)
        if y_noisy is not None:
            loss_fid = F.l1_loss(denoised_img, y_noisy)

        # L_reg: Total variation / smoothness regularization
        tv_h = torch.mean(torch.abs(denoised_img[:, :, 1:, :] - denoised_img[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(denoised_img[:, :, :, 1:] - denoised_img[:, :, :, :-1]))
        loss_reg = tv_h + tv_w

        total_self = (
            self.lambda_mask * loss_mask +
            self.lambda_pc * loss_pc +
            0.1 * loss_fid +
            self.lambda_reg * loss_reg
        )

        return {
            "loss_total": total_self,
            "loss_mask": loss_mask,
            "loss_pc": loss_pc,
            "loss_fid": loss_fid,
            "loss_reg": loss_reg,
        }

    def flow_loss(
        self,
        v_mags: List[torch.Tensor],
        v_phases: List[torch.Tensor],
        u_mags: List[torch.Tensor],
        u_phases: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Computes asymmetric flow matching loss.
        """
        device = v_mags[0].device
        loss_mag = torch.tensor(0.0, device=device)
        loss_phase = torch.tensor(0.0, device=device)

        for vm, um in zip(v_mags, u_mags):
            loss_mag = loss_mag + F.mse_loss(vm, um)

        for vp, up in zip(v_phases, u_phases):
            loss_phase = loss_phase + F.mse_loss(vp, up)

        total_flow = self.weight_mag * loss_mag + self.weight_phase * loss_phase
        return {
            "loss_flow": total_flow,
            "loss_mag": loss_mag,
            "loss_phase": loss_phase,
        }
