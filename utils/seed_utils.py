"""
Reproducibility utilities for CM-MTD framework.
Sets random seeds across NumPy, Python, TensorFlow, and PyTorch.
"""
import os
import random
import numpy as np
from typing import Optional


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility across all frameworks.
    
    Args:
        seed: Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_device(prefer_gpu: bool = True) -> str:
    """
    Get the best available compute device.
    
    Args:
        prefer_gpu: If True, use GPU when available.
    
    Returns:
        Device string ('cuda', 'cuda:0', or 'cpu').
    """
    try:
        import torch
        if prefer_gpu and torch.cuda.is_available():
            device = "cuda"
        elif prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        return device
    except ImportError:
        return "cpu"
