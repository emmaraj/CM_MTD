"""
Compatibility shims for optional heavy dependencies.
Provides no-op or pure-numpy fallbacks so the project imports cleanly
even when PyTorch / TensorFlow / gymnasium are not installed.
Import this module BEFORE any conditional torch/tf usage.
"""
import sys
import types
import logging

logger = logging.getLogger("cm_mtd.compat")


# ── tqdm ─────────────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # minimal no-op shim
        def __init__(self, iterable=None, *args, **kwargs):
            self._it = iterable
            desc = kwargs.get("desc", "")
            if desc:
                print(f"{desc}...")
        def __iter__(self):
            return iter(self._it) if self._it is not None else iter([])
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): pass
        def set_postfix(self, **kw): pass
        def close(self): pass
    sys.modules["tqdm"] = types.ModuleType("tqdm")
    sys.modules["tqdm"].tqdm = tqdm
    logger.info("tqdm not found — using no-op shim")


# ── torch ─────────────────────────────────────────────────────────────────────
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not found — RL agents will be unavailable. "
                   "Install with: pip install torch")


# ── TensorFlow / Keras ────────────────────────────────────────────────────────
TF_AVAILABLE = False
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    try:
        import keras
        TF_AVAILABLE = True
    except ImportError:
        logger.warning("TensorFlow/Keras not found — LSTM model unavailable. "
                       "Install with: pip install tensorflow>=2.15")


# ── gymnasium ─────────────────────────────────────────────────────────────────
GYM_AVAILABLE = False
try:
    import gymnasium
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    # Provide a minimal stub so environment/dtmn_env.py imports cleanly
    import numpy as np

    class _Box:
        def __init__(self, low, high, shape, dtype=float):
            self.low = low; self.high = high
            self.shape = shape; self.dtype = dtype

    class _Discrete:
        def __init__(self, n):
            self.n = n
        def sample(self):
            import random
            return random.randint(0, self.n - 1)

    class _Spaces(types.ModuleType):
        Box = _Box
        Discrete = _Discrete

    class _Env:
        """Minimal gymnasium.Env shim."""
        metadata = {}
        observation_space = None
        action_space = None
        def reset(self, **kw): raise NotImplementedError
        def step(self, action): raise NotImplementedError
        def render(self): pass
        def close(self): pass

    gym_module = types.ModuleType("gymnasium")
    gym_module.Env = _Env
    gym_module.spaces = _Spaces("gymnasium.spaces")
    sys.modules["gymnasium"] = gym_module
    sys.modules["gymnasium.spaces"] = gym_module.spaces
    logger.info("gymnasium not found — using minimal shim")
