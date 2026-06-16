"""
Neural Network Architectures for DQN (upper layer) and PPO (lower layer).

DQN  uses ReLU activations, hidden layers [256]         (Table I).
PPO  uses tanh activations, hidden layers [256, 256]    (Table I).

All nn.Module subclasses are defined inside the TORCH_AVAILABLE guard so
this module imports cleanly even when PyTorch is not installed.
"""
import logging
from typing import List, Optional, Tuple
import numpy as np
from utils.compat import TORCH_AVAILABLE

logger = logging.getLogger("cm_mtd.networks")


def _require_torch(name: str) -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            f"{name} requires PyTorch — install with:  pip install torch"
        )


if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ── Dueling DQN ──────────────────────────────────────────────────────────
    class DuelingQNetwork(nn.Module):
        """
        Dueling DQN for upper-layer macro-action selection.
        Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)

        Args:
            state_dim:     Observation dimension.
            n_actions:     Number of macro-actions (default 4).
            hidden_layers: Per Table I: [256].
            activation:    'relu' (Table I).
        """
        def __init__(self, state_dim: int, n_actions: int = 4,
                     hidden_layers: Optional[List[int]] = None,
                     activation: str = "relu"):
            super().__init__()
            hidden_layers = hidden_layers or [256]
            act = nn.ReLU if activation.lower() == "relu" else nn.Tanh

            trunk, in_dim = [], state_dim
            for h in hidden_layers:
                trunk += [nn.Linear(in_dim, h), act()]
                in_dim = h
            self.trunk = nn.Sequential(*trunk)

            self.value_stream = nn.Sequential(
                nn.Linear(in_dim, 128), act(), nn.Linear(128, 1))
            self.advantage_stream = nn.Sequential(
                nn.Linear(in_dim, 128), act(), nn.Linear(128, n_actions))
            self._init_weights()

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            s = self.trunk(state)
            V = self.value_stream(s)
            A = self.advantage_stream(s)
            return V + A - A.mean(dim=1, keepdim=True)

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)

    # ── PPO Actor-Critic ──────────────────────────────────────────────────────
    class PPOActorCritic(nn.Module):
        """
        Shared-trunk Actor-Critic for PPO (lower layer).
        Hidden [256, 256] + tanh activation  (Table I).

        Args:
            state_dim:     Observation dimension.
            n_actions:     Number of mutation actions.
            hidden_layers: [256, 256] per Table I.
            activation:    'tanh' (Table I).
        """
        def __init__(self, state_dim: int, n_actions: int,
                     hidden_layers: Optional[List[int]] = None,
                     activation: str = "tanh"):
            super().__init__()
            hidden_layers = hidden_layers or [256, 256]
            act = nn.Tanh if activation.lower() == "tanh" else nn.ReLU

            shared, in_dim = [], state_dim
            for h in hidden_layers:
                shared += [nn.Linear(in_dim, h), act()]
                in_dim = h
            self.shared = nn.Sequential(*shared)
            self.actor  = nn.Linear(in_dim, n_actions)
            self.critic = nn.Linear(in_dim, 1)
            self._init_weights()

        def forward(self, state):
            f = self.shared(state)
            return self.actor(f), self.critic(f)

        def get_action(self, state, deterministic: bool = False):
            logits, values = self.forward(state)
            dist    = torch.distributions.Categorical(logits=logits)
            actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            return actions, dist.log_prob(actions), values

        def evaluate_actions(self, state, actions):
            logits, values = self.forward(state)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.log_prob(actions), dist.entropy().mean(), values

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)
            nn.init.orthogonal_(self.actor.weight,  gain=0.01)
            nn.init.orthogonal_(self.critic.weight, gain=1.0)

else:
    # Placeholder classes that raise a clear error when instantiated
    class DuelingQNetwork:   # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch("DuelingQNetwork")
    class PPOActorCritic:    # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch("PPOActorCritic")
