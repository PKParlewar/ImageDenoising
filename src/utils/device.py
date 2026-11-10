"""
Hardware device selection utilities supporting CUDA, Apple Silicon MPS, and CPU.
"""

from typing import Optional
import torch


def get_default_device(device_name: Optional[str] = None) -> torch.device:
    """
    Selects the optimal acceleration device.
    Hierarchy: User requested -> CUDA -> Apple Silicon MPS -> CPU.
    """
    if device_name is not None and device_name.lower() != "auto":
        return torch.device(device_name)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
