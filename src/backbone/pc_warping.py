"""
Structure-Guided (PC-Warped) Hilbert Traversal and Tradeoff Analysis.

Section 4 of info.md:
Computes traversal cost field rho(u, v) = 1 - PC_norm(u, v)
and performs bounded lookahead warping (window w) to bend the scan path
along continuous structural edges while preserving fractal locality bounds.
"""

from typing import Tuple, List, Dict, Any, Optional
import math
import torch
import torch.nn.functional as F

from .hilbert_scan import generate_hilbert_curve_2d


def compute_traversal_cost(pc: torch.Tensor) -> torch.Tensor:
    """
    Computes traversal cost field rho(u, v) = 1 - PC_norm(u, v).
    Strong structures have low cost (0.0), flat/noisy areas have high cost (1.0).
    
    Args:
        pc: [B, 1, H, W] or [1, 1, H, W] normalized Phase Congruency map.
    Returns:
        rho: Traversal cost field of same shape.
    """
    pc_min = pc.min()
    pc_max = pc.max()
    pc_norm = (pc - pc_min) / (pc_max - pc_min + 1e-8)
    return 1.0 - pc_norm


def warp_hilbert_trajectory(
    pc: torch.Tensor,
    window_size: int = 8
) -> torch.Tensor:
    """
    Generates structure-warped traversal permutation sigma: [L]
    mapping sequence index t -> warped Hilbert coordinate index.
    GPU-native vectorized block permutation with zero CPU copies.
    
    Args:
        pc: [1, 1, H, W] or [H, W] Phase Congruency map.
        window_size: Bounded lookahead window w (w=0 yields vanilla Hilbert scan).
        
    Returns:
        warped_indices: [H*W] LongTensor containing bijective permutation of [0, ..., H*W-1].
    """
    if pc.ndim == 4:
        pc_2d = pc[0, 0]
    elif pc.ndim == 3:
        pc_2d = pc[0]
    else:
        pc_2d = pc

    h, w = pc_2d.shape
    total_len = h * w
    y_h, x_h = generate_hilbert_curve_2d(h, w)
    y_h = y_h.to(pc.device)
    x_h = x_h.to(pc.device)
    
    # 1D flattened coordinates of canonical Hilbert curve
    canonical_flat = (y_h * w + x_h)  # [L]
    
    if window_size <= 0:
        return canonical_flat
        
    # Extract costs along canonical Hilbert path directly on GPU
    cost_field = compute_traversal_cost(pc_2d.unsqueeze(0).unsqueeze(0))[0, 0]
    costs_1d = cost_field[y_h, x_h]  # [L]
    
    # Vectorized bounded-window permutation on GPU
    chunk_size = max(2, window_size * 2)
    n_chunks = total_len // chunk_size
    
    if n_chunks > 0:
        costs_chunked = costs_1d[:n_chunks * chunk_size].view(n_chunks, chunk_size)
        perm_local = torch.argsort(costs_chunked, dim=-1)
        offsets = torch.arange(n_chunks, device=pc.device).unsqueeze(1) * chunk_size
        perm_global = (perm_local + offsets).view(-1)
        if n_chunks * chunk_size < total_len:
            rem = torch.arange(n_chunks * chunk_size, total_len, device=pc.device)
            perm_global = torch.cat([perm_global, rem])
    else:
        perm_global = torch.argsort(costs_1d)
        
    warped_flat = canonical_flat[perm_global]
    return warped_flat


def warped_flatten_2d(
    x: torch.Tensor,
    warped_flat_indices: torch.Tensor
) -> torch.Tensor:
    """
    Flattens 2D tensor [B, C, H, W] into 1D sequence [B, C, L] along warped trajectory.
    """
    b, c, h, w = x.shape
    x_reshaped = x.view(b, c, h * w)
    seq = torch.index_select(x_reshaped, dim=-1, index=warped_flat_indices)
    return seq


def warped_unflatten_2d(
    seq: torch.Tensor,
    h: int,
    w: int,
    warped_flat_indices: torch.Tensor
) -> torch.Tensor:
    """
    Unflattens 1D sequence [B, C, L] into 2D tensor [B, C, H, W] using inverse permutation.
    """
    b, c, l = seq.shape
    inv_indices = torch.argsort(warped_flat_indices)
    out = torch.index_select(seq, dim=-1, index=inv_indices)
    return out.view(b, c, h, w)


def analyze_warping_tradeoff(
    pc_map: torch.Tensor,
    window_sizes: List[int] = [0, 2, 4, 8, 16, 32]
) -> Dict[str, Any]:
    """
    Task 3.4: Analyzes the trade-off between lookahead window w and:
    1. Structural Continuity Metric: 1D Traversal Total Variation along path (lower is better, smoother along edges).
    2. Spatial Locality Distortion: mean Euclidean step distance ||p_{t+1} - p_t||_2.
    """
    h, w = pc_map.shape[-2:]
    costs = compute_traversal_cost(pc_map)[0, 0].cpu()
    
    results = {
        "window_sizes": window_sizes,
        "mean_costs": [],
        "path_total_variations": [],
        "structural_continuity_scores": [],
        "mean_step_distances": [],
        "max_step_distances": [],
    }
    
    for ws in window_sizes:
        warped_flat = warp_hilbert_trajectory(pc_map, window_size=ws).cpu()
        
        # 1. Structural continuity: evaluate costs along the warped path
        y_coords = warped_flat // w
        x_coords = warped_flat % w
        path_costs = costs[y_coords, x_coords]
        mean_cost = path_costs.mean().item()
        
        # 1D Traversal Total Variation: sum of cost jumps between consecutive sequence tokens
        cost_jumps = torch.abs(path_costs[1:] - path_costs[:-1])
        tv_path = cost_jumps.mean().item()
        # Continuity score (1.0 - normalized TV, higher is better)
        continuity_score = max(0.0, 1.0 - tv_path)
        
        # 2. Spatial locality: consecutive Euclidean step distances
        dy = (y_coords[1:] - y_coords[:-1]).float()
        dx = (x_coords[1:] - x_coords[:-1]).float()
        step_dists = torch.sqrt(dy ** 2 + dx ** 2)
        
        results["mean_costs"].append(mean_cost)
        results["path_total_variations"].append(tv_path)
        results["structural_continuity_scores"].append(continuity_score)
        results["mean_step_distances"].append(step_dists.mean().item())
        results["max_step_distances"].append(step_dists.max().item())
        
    return results
