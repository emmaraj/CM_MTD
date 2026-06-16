"""
Upper-Layer DQN Agent — Section VI-B of the paper.

Implements the upper-layer DQN that selects macro-actions O_t from {o_s, o_a, o_r, o_c}.

Algorithm 1 (paper), upper-layer logic:
  Line  5: Initialize main Q-network with weight θ^u_t
  Line  6: Initialize target Q-network with weight θ^u_{t-n} = θ^u_t
  Lines 12–18: ε-greedy macro-action selection
  Line 29: Observe upper layer reward R^u_t
  Line 30: Obtain next state S_{t+1} by LSTMNet
  Lines 31–32: Store D'_t = {S_t, O_t, R^u_t, S_{t+1}} into B₁
  Lines 35–36: Minimize DQN loss (Eq. 14): L(θ) = E[(y_t - Q(S_t, O_t; θ_t))²]
  Line 42:    θ^u_{t-n} ← θ^u_t (target network update)

DQN Loss (Eq. 14):
  L(θ_t) = E[(y_t - Q_t(S_t, O_t, θ_t))²]
  y_t = R_t + max_{A'} Q(S', O', θ_{t-n})

Upper-layer value function (Eq. 13):
  V*_u(S_t, O_t) = max_{π_u} E[R_t + Σ_{k=1}^∞ γ^k R_{t+k}]
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.compat import TORCH_AVAILABLE
if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
else:
    torch = None  # type: ignore

from models.networks import DuelingQNetwork
from agents.replay_buffer import ReplayBuffer

logger = logging.getLogger("cm_mtd.dqn")


class UpperLayerDQN:
    """
    DQN agent for upper-layer macro-action selection.

    Uses Dueling DQN with experience replay and target network.
    Implements ε-greedy exploration with decay.

    Args:
        state_dim:      Observation dimension.
        n_macro_actions: Number of macro-actions (4).
        hidden_layers:  DNN hidden layer sizes (Table I: [256]).
        learning_rate:  Adam optimizer LR (Table I: 1e-3).
        gamma:          Discount factor γ (0.99).
        epsilon_start:  Initial exploration rate ε.
        epsilon_end:    Minimum exploration rate.
        epsilon_decay:  Multiplicative ε decay per episode.
        buffer_size:    Replay buffer capacity.
        batch_size:     Mini-batch size for Q-network updates.
        target_update:  Target network update frequency (episodes).
        device:         Compute device.
        dueling:        Use Dueling DQN architecture.
    """

    def __init__(
        self,
        state_dim: int,
        n_macro_actions: int = 4,
        hidden_layers: Optional[List[int]] = None,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        min_buffer_size: int = 1_000,
        target_update: int = 10,
        device: str = "cpu",
        dueling: bool = True,
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_macro_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update = target_update
        self.device = torch.device(device)

        hidden = hidden_layers or [256]

        # ── Main Q-network (θ^u_t) — Algorithm 1 line 5 ─────────────────
        self.q_network = DuelingQNetwork(
            state_dim=state_dim,
            n_actions=n_macro_actions,
            hidden_layers=hidden,
        ).to(self.device)

        # ── Target Q-network (θ^u_{t-n}) — Algorithm 1 line 6 ───────────
        self.target_network = DuelingQNetwork(
            state_dim=state_dim,
            n_actions=n_macro_actions,
            hidden_layers=hidden,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer (Table I: Adam)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=2000, gamma=0.5
        )

        # Replay buffer B₁ — Algorithm 1 line 4
        self.buffer = ReplayBuffer(
            capacity=buffer_size,
            state_dim=state_dim,
            device=str(self.device),
        )

        # Training counters
        self._episode_count = 0
        self._update_count = 0

        # Metrics tracking
        self.losses: List[float] = []
        self.epsilon_history: List[float] = []
        self.q_value_history: List[float] = []

        logger.info(
            f"UpperLayerDQN initialized | "
            f"state_dim={state_dim} | n_actions={n_macro_actions} | "
            f"device={self.device}"
        )

    # ─── Action Selection ────────────────────────────────────────────────────

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """
        Select macro-action using ε-greedy policy.

        Algorithm 1, lines 12–17:
            Generate random p
            if p ≤ ε: select O_t randomly
            else:      O_t = arg max_{O'} Q_t(S_t, O_t; θ^u_t)

        Args:
            state:        State observation [state_dim].
            deterministic: If True, always exploit (used for evaluation).

        Returns:
            Macro-action integer in {0, 1, 2, 3}.
        """
        if not deterministic and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(q_values.argmax(dim=1).item())

    # ─── Buffer Management ───────────────────────────────────────────────────

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store transition in replay buffer B₁.

        Algorithm 1, line 32: B₁ = B₁ ∪ D'_t
        D'_t = {S_t, O_t, R^u_t, S_{t+1}}
        """
        self.buffer.push(state, action, reward, next_state, done)

    # ─── Network Update ──────────────────────────────────────────────────────

    def update(self) -> Optional[float]:
        """
        Perform one DQN gradient update.

        Algorithm 1, lines 35–36:
          Compute target: y_t = R_t + max_{A'} Q(S', O', θ^u_{t-n})
          Minimize: L(θ_t) = E[(y_t - Q_t(S_t, O_t; θ_t))²]  (Eq. 14)

        Returns:
            Loss value, or None if buffer not ready.
        """
        if not self.buffer.is_ready(self.min_buffer_size):
            return None

        batch = self.buffer.sample(self.batch_size)
        states      = batch["states"]
        actions     = batch["actions"]
        rewards     = batch["rewards"]
        next_states = batch["next_states"]
        dones       = batch["dones"]

        # Current Q-values: Q(S_t, O_t; θ_t)
        q_current = self.q_network(states)
        q_taken   = q_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Eq. 14): y_t = R_t + γ · max_a Q(S', a; θ_{t-n})
        with torch.no_grad():
            q_next   = self.target_network(next_states)
            q_target = rewards + self.gamma * q_next.max(dim=1)[0] * (1 - dones)

        # Huber loss (more stable than MSE for Q-learning)
        loss = F.smooth_l1_loss(q_taken, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.losses.append(loss_val)
        self.q_value_history.append(float(q_taken.mean().item()))
        self._update_count += 1

        return loss_val

    def update_target_network(self) -> None:
        """
        Hard-copy main network weights to target.

        Algorithm 1, line 42: θ^u_{t-n} ← θ^u_t
        Called every `target_update` episodes.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> None:
        """
        Decay exploration rate ε after each episode.
        ε = max(ε_end, ε × decay_rate)
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
        self._episode_count += 1

        # Periodic target network update
        if self._episode_count % self.target_update == 0:
            self.update_target_network()

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_network":      self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer":      self.optimizer.state_dict(),
            "epsilon":        self.epsilon,
            "episode_count":  self._episode_count,
            "losses":         self.losses,
        }, path)
        logger.info(f"DQN checkpoint saved to {path}")

    def load(self, path: str) -> None:
        """Load a saved checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.target_network.load_state_dict(ckpt["target_network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self._episode_count = ckpt.get("episode_count", 0)
        self.losses = ckpt.get("losses", [])
        logger.info(f"DQN checkpoint loaded from {path}")

    @classmethod
    def from_config(cls, config: Dict, state_dim: int) -> "UpperLayerDQN":
        """Instantiate from config dict."""
        dqn_cfg = config.get("dqn", {})
        device = config.get("experiment", {}).get("device", "cpu")
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
        except ImportError:
            device = "cpu"

        return cls(
            state_dim=state_dim,
            n_macro_actions=config.get("smdp", {}).get("n_macro_actions", 4),
            hidden_layers=dqn_cfg.get("hidden_layers", [256]),
            learning_rate=dqn_cfg.get("learning_rate", 1e-3),
            gamma=dqn_cfg.get("gamma", 0.99),
            epsilon_start=dqn_cfg.get("epsilon_start", 1.0),
            epsilon_end=dqn_cfg.get("epsilon_end", 0.01),
            epsilon_decay=dqn_cfg.get("epsilon_decay", 0.995),
            buffer_size=dqn_cfg.get("buffer_size", 100_000),
            batch_size=dqn_cfg.get("batch_size", 64),
            min_buffer_size=dqn_cfg.get("min_buffer_size", 1_000),
            target_update=dqn_cfg.get("target_update_freq", 10),
            device=device,
            dueling=dqn_cfg.get("dueling", True),
        )
