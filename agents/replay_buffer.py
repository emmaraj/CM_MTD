"""
Experience Replay Buffers — B₁ (DQN) and B₂ (PPO).

Both buffers work in pure NumPy; tensors are only created at sample time
if PyTorch is available, making the module importable without torch.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.compat import TORCH_AVAILABLE
if TORCH_AVAILABLE:
    import torch

logger = logging.getLogger("cm_mtd.buffer")


# ─── DQN Replay Buffer (B₁) ──────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular experience replay buffer for DQN (Algorithm 1, B₁).
    Stores D' = {S_t, O_t, R^u_t, S_{t+1}}.

    Args:
        capacity:  Maximum transitions stored.
        state_dim: Observation vector length.
        device:    PyTorch device string ('cpu' / 'cuda').
    """

    def __init__(self, capacity: int = 100_000,
                 state_dim: int = 96,
                 device: str = "cpu") -> None:
        self.capacity  = capacity
        self.state_dim = state_dim
        self.device    = device
        self._ptr  = 0
        self._size = 0

        self._states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions     = np.zeros(capacity, dtype=np.int64)
        self._rewards     = np.zeros(capacity, dtype=np.float32)
        self._dones       = np.zeros(capacity, dtype=np.float32)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Add one transition (Algorithm 1 line 32: B₁ = B₁ ∪ D'_t)."""
        self._states[self._ptr]      = state
        self._next_states[self._ptr] = next_state
        self._actions[self._ptr]     = action
        self._rewards[self._ptr]     = reward
        self._dones[self._ptr]       = float(done)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict:
        """Random mini-batch.  Returns torch tensors if available, else numpy."""
        if self._size < batch_size:
            raise RuntimeError(f"Buffer has {self._size} < {batch_size} samples")
        idx = np.random.choice(self._size, size=batch_size, replace=False)
        batch = {
            "states":      self._states[idx],
            "actions":     self._actions[idx],
            "rewards":     self._rewards[idx],
            "next_states": self._next_states[idx],
            "dones":       self._dones[idx],
        }
        if TORCH_AVAILABLE:
            dev = torch.device(self.device)
            return {
                "states":      torch.FloatTensor(batch["states"]).to(dev),
                "actions":     torch.LongTensor(batch["actions"]).to(dev),
                "rewards":     torch.FloatTensor(batch["rewards"]).to(dev),
                "next_states": torch.FloatTensor(batch["next_states"]).to(dev),
                "dones":       torch.FloatTensor(batch["dones"]).to(dev),
            }
        return batch          # plain numpy fallback

    def is_ready(self, min_size: int) -> bool:
        return self._size >= min_size

    def __len__(self) -> int:
        return self._size


# ─── PPO Rollout Buffer (B₂) ─────────────────────────────────────────────────

class RolloutBuffer:
    """
    Fixed-size rollout buffer for PPO (Algorithm 1, B₂).
    Computes GAE advantages (lines 25-26) and returns (line 27).

    Args:
        n_steps:    Rollout length before a PPO update.
        state_dim:  Observation dimension.
        gamma:      Discount γ.
        gae_lambda: GAE λ (ξ in paper).
        device:     PyTorch device string.
    """

    def __init__(self, n_steps: int = 128, state_dim: int = 96,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 device: str = "cpu") -> None:
        self.n_steps    = n_steps
        self.state_dim  = state_dim
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.device     = device
        self.reset()

    def reset(self) -> None:
        self._states    = np.zeros((self.n_steps, self.state_dim), dtype=np.float32)
        self._actions   = np.zeros(self.n_steps, dtype=np.int64)
        self._rewards   = np.zeros(self.n_steps, dtype=np.float32)
        self._values    = np.zeros(self.n_steps + 1, dtype=np.float32)
        self._log_probs = np.zeros(self.n_steps, dtype=np.float32)
        self._dones     = np.zeros(self.n_steps, dtype=np.float32)
        self._ptr = 0

    def push(self, state: np.ndarray, action: int, reward: float,
             value: float, log_prob: float, done: bool) -> None:
        """Store one step (Algorithm 1 line 24: B₂ = B₂ ∪ D_k)."""
        if self._ptr >= self.n_steps:
            raise OverflowError("RolloutBuffer full — call reset() first")
        self._states[self._ptr]    = state
        self._actions[self._ptr]   = action
        self._rewards[self._ptr]   = reward
        self._values[self._ptr]    = value
        self._log_probs[self._ptr] = log_prob
        self._dones[self._ptr]     = float(done)
        self._ptr += 1

    def compute_returns_and_advantages(
            self, last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        GAE — Algorithm 1 lines 25-27:
            δ_t = r_t + γ V(s_{t+1}) - V(s_t)
            Â_k = Σ_{q=k}^K (γλ)^{q-k} δ_q
            V̂_k = Â_k + V(s_k)
        """
        self._values[self._ptr] = last_value
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(self._ptr)):
            delta = (self._rewards[t]
                     + self.gamma * self._values[t + 1] * (1 - self._dones[t])
                     - self._values[t])
            gae = delta + self.gamma * self.gae_lambda * (1 - self._dones[t]) * gae
            advantages[t] = gae
        returns = advantages[:self._ptr] + self._values[:self._ptr]
        return advantages[:self._ptr], returns

    def get_batches(self, mini_batch_size: int) -> List[Dict]:
        """Shuffled mini-batches for PPO update (Algorithm 1 line 37)."""
        advantages, returns = self.compute_returns_and_advantages()
        n = self._ptr

        adv_std  = advantages.std() + 1e-8
        adv_norm = (advantages - advantages.mean()) / adv_std

        indices = np.random.permutation(n)
        batches = []
        for start in range(0, n, mini_batch_size):
            idx = indices[start: start + mini_batch_size]
            b = {
                "states":     self._states[idx],
                "actions":    self._actions[idx],
                "advantages": adv_norm[idx],
                "returns":    returns[idx],
                "log_probs":  self._log_probs[idx],
            }
            if TORCH_AVAILABLE:
                dev = torch.device(self.device)
                b = {k: (torch.FloatTensor(v).to(dev)
                         if k != "actions"
                         else torch.LongTensor(v).to(dev))
                     for k, v in b.items()}
            batches.append(b)
        return batches

    def is_full(self) -> bool:
        return self._ptr >= self.n_steps

    def __len__(self) -> int:
        return self._ptr
