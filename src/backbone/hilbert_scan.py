"""
Canonical 2D Hilbert Curve generation and tensor flattening/unflattening.

Provides exact 2D Hilbert space-filling curve index mappings for arbitrary resolutions.
"""

from typing import Tuple, List, Optional
import math
import functools
import torch


def d2xy(n: int, d: int) -> Tuple[int, int]:
    """
    Maps 1D index d on an n x n grid (n is a power of 2) to 2D coordinates (y, x).
    """
    x = y = 0
    s = 1
    t = d
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return y, x


@functools.lru_cache(maxsize=32)
def generate_hilbert_curve_2d(h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates 2D Hilbert trajectory indices (y_indices, x_indices) of length H * W.
    Guarantees every coordinate (y, x) in [0, H-1] x [0, W-1] is visited exactly once.
    
    Returns:
        y_indices: [H*W] LongTensor
        x_indices: [H*W] LongTensor
    """
    max_dim = max(h, w)
    power = math.ceil(math.log2(max(max_dim, 2)))
    n = 2 ** power
    
    y_coords = []
    x_coords = []
    
    for d in range(n * n):
        y, x = d2xy(n, d)
        if y < h and x < w:
            y_coords.append(y)
            x_coords.append(x)
            
    assert len(y_coords) == h * w, f"Hilbert traversal count {len(y_coords)} != {h * w}"
    
    y_t = torch.tensor(y_coords, dtype=torch.long)
    x_t = torch.tensor(x_coords, dtype=torch.long)
    return y_t, x_t


def hilbert_flatten_2d(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flattens spatial 2D tensor [B, C, H, W] into 1D sequence [B, C, L] along Hilbert curve.
    
    Args:
        x: [B, C, H, W]
    Returns:
        seq: [B, C, H*W]
        (y_indices, x_indices): Trajectory coordinates
    """
    b, c, h, w = x.shape
    y_idx, x_idx = generate_hilbert_curve_2d(h, w)
    y_idx = y_idx.to(x.device)
    x_idx = x_idx.to(x.device)
    
    # 1D flattened coordinates
    flat_indices = y_idx * w + x_idx
    x_reshaped = x.view(b, c, h * w)
    seq = torch.index_select(x_reshaped, dim=-1, index=flat_indices)
    return seq, (y_idx, x_idx)


def hilbert_unflatten_2d(
    seq: torch.Tensor,
    h: int,
    w: int,
    y_idx: Optional[torch.Tensor] = None,
    x_idx: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Reconstructs 2D tensor [B, C, H, W] from 1D Hilbert sequence [B, C, H*W].
    """
    b, c, l = seq.shape
    if y_idx is None or x_idx is None:
        y_idx, x_idx = generate_hilbert_curve_2d(h, w)
        y_idx = y_idx.to(seq.device)
        x_idx = x_idx.to(seq.device)
        
    flat_indices = y_idx * w + x_idx
    out = torch.empty((b, c, h * w), dtype=seq.dtype, device=seq.device)
    
    # Scatter back to spatial coordinates
    # We create inverse mapping
    inv_indices = torch.argsort(flat_indices)
    out = torch.index_select(seq, dim=-1, index=inv_indices)
    return out.view(b, c, h, w)
