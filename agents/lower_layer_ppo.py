"""
Lower-Layer PPO Agent — Section VI-B of the paper.

Implements the lower-layer PPO that selects mutation actions A_k
(mutated routes and/or IP address spaces) within the current macro-action.

Algorithm 1 (paper), lower-layer logic:
  Line  7: Initialize critic network V(S_k, φ)
  Line  8: Initialize actor network with weight θ^l_k
  Lines 19–28: For k=1..K, run PPO policy π_{θ^l_{k-n}} to select A_k
  Lines 34–40: For each epoch, compute J^{actor} and J^{critic}, update networks
  Line 43:    θ^l_{k-n} ← θ^l_k

PPO Actor Loss (Eq. 16 — clipped surrogate objective):
  L^{clip}(θ_t) = E[min(r_t(θ_t), clip(r_t(θ_t), 1-ε, 1+ε)) Â_t]

  where r_t(θ_t) = π(A_t|S_t, O_t; θ_t) / π(A_t|S_t, O_t; θ^old_t)  (Eq. 17)

Lower-layer value function (Eq. 15):
  V*_l(S_t, A_t; O_t) = max_{π_l} E[R_t + Σ_{k=1}^N γ^k R_{t+k}]

Algorithm 1 — J^{actor} (line 37):
  J^{actor}(θ) = (1/K) Σ min(r_k(θ), clip(r_k(θ), 1-ε, 1+ε)) Â_k

Algorithm 1 — J^{critic} (line 39):
  J^{critic}(θ) = -(1/K) Σ (V̂_k - V(S_k, φ))²
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional heavy deps — shims loaded first
from utils.compat import TORCH_AVAILABLE
if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F   # ← moved from bottom of file
    import torch.optim as optim
else:
    torch = None  # type: ignore

from models.networks import PPOActorCritic
from agents.replay_buffer import RolloutBuffer

logger = logging.getLogger("cm_mtd.ppo")


class LowerLayerPPO:
    """
    PPO agent for lower-layer mutation action selection.

    Given the current macro-action (context from upper DQN),
    selects specific mutation parameters (route/IP choices)
    using the PPO clipped surrogate objective.

    Args:
        state_dim:      Observation dimension (same as upper layer).
        n_actions:      Number of concrete mutation actions.
        hidden_layers:  DNN hidden layers (Table I: [256, 256]).
        learning_rate:  Adam LR (Table I: 1e-4).
        gamma:          Discount factor γ (0.99).
        gae_lambda:     GAE λ = ξ in paper (0.95).
        clip_range:     PPO clip ε (Table I: 0.2).
        value_coef:     Value loss coefficient.
        entropy_coef:   Entropy bonus coefficient.
        n_steps:        Rollout steps per update.
        mini_batch_size: K mini-batch size (Table I: 5).
        n_epochs:       PPO update epochs per rollout.
        max_grad_norm:  Gradient clipping.
        device:         Compute device.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 4,
        hidden_layers: Optional[List[int]] = None,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_steps: int = 256,
        mini_batch_size: int = 5,
        n_epochs: int = 10,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.mini_batch_size = mini_batch_size
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        hidden = hidden_layers or [256, 256]

        # ── Actor-Critic network (θ^l_k, φ) — Algorithm 1 lines 7–8 ─────
        self.actor_critic = PPOActorCritic(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_layers=hidden,
            activation="tanh",   # Table I: tanh activation
        ).to(self.device)

        # Old policy (for importance sampling ratio r_t) — Eq. 17
        self.actor_critic_old = PPOActorCritic(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_layers=hidden,
            activation="tanh",
        ).to(self.device)
        self._sync_old_policy()

        # Optimizer (Table I: Adam, lr=1e-4)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.lr_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=10000
        )

        # Rollout buffer B₂ — Algorithm 1 line 4
        self.buffer = RolloutBuffer(
            n_steps=n_steps,
            state_dim=state_dim,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=str(self.device),
        )

        # Metrics tracking
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        self.entropy_history: List[float] = []
        self._update_count = 0

        logger.info(
            f"LowerLayerPPO initialized | "
            f"state_dim={state_dim} | n_actions={n_actions} | "
            f"clip_ε={clip_range} | device={self.device}"
        )

    # ─── Action Selection ─────────────────────────────────────────────────────

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Sample mutation action from current policy π_{θ^l_{k-n}}.

        Algorithm 1, line 20: Run policy π_{θ^l_{k-n}} to select action A_k.

        Args:
            state:        State observation [state_dim].
            deterministic: If True, take argmax (evaluation mode).

        Returns:
            action:   Mutation action integer.
            log_prob: Log probability of selected action.
            value:    Critic value estimate V(s).
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            actions, log_probs, values = self.actor_critic.get_action(
                state_t, deterministic=deterministic
            )
        return (
            int(actions.item()),
            float(log_probs.item()),
            float(values.item()),
        )

    # ─── Buffer Management ────────────────────────────────────────────────────

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """
        Store rollout step in buffer B₂.

        Algorithm 1, line 24: B₂ = B₂ ∪ D_k
        D_k = (S_k, A_k, R^l_k, S_{k+1})
        """
        if not self.buffer.is_full():
            self.buffer.push(state, action, reward, value, log_prob, done)

    # ─── Network Update ───────────────────────────────────────────────────────

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout data.

        Algorithm 1, lines 34–40:
          For each epoch:
            Compute J^{actor} (Eq. 16 / line 37)
            Update θ^l_k by ∇_θ J^{actor} (line 38)
            Compute J^{critic} (line 39)
            Update φ by ∇_φ J^{critic} (line 40)

        Args:
            last_value: Bootstrap value V(s_T) for GAE computation.

        Returns:
            Dict with actor_loss, critic_loss, entropy.
        """
        if len(self.buffer) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            mini_batches = self.buffer.get_batches(self.mini_batch_size)

            for batch in mini_batches:
                states     = batch["states"]
                actions    = batch["actions"]
                advantages = batch["advantages"]
                returns    = batch["returns"]
                old_log_probs = batch["log_probs"]

                # Current policy evaluation
                log_probs, entropy, values = self.actor_critic.evaluate_actions(
                    states, actions
                )

                # ── PPO clipped ratio r_t(θ) (Eq. 17) ─────────────────────
                ratio = torch.exp(log_probs - old_log_probs)

                # ── J^{actor} clipped objective (Eq. 16 / Algorithm 1 line 37) ──
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # ── J^{critic} value loss (Algorithm 1 line 39) ───────────
                critic_loss = F.mse_loss(values.squeeze(-1), returns)

                # Total loss (with entropy bonus for exploration)
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    max_norm=self.max_grad_norm,
                )
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # Sync old policy — Algorithm 1 line 43: θ^l_{k-n} ← θ^l_k
        self._sync_old_policy()
        self.buffer.reset()
        self._update_count += 1

        metrics = {
            "actor_loss": total_actor_loss / max(n_updates, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }
        self.actor_losses.append(metrics["actor_loss"])
        self.critic_losses.append(metrics["critic_loss"])
        self.entropy_history.append(metrics["entropy"])

        return metrics

    def _sync_old_policy(self) -> None:
        """Hard copy current actor-critic weights to old policy network."""
        self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())
        self.actor_critic_old.eval()

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save PPO checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor_critic":     self.actor_critic.state_dict(),
            "actor_critic_old": self.actor_critic_old.state_dict(),
            "optimizer":        self.optimizer.state_dict(),
            "actor_losses":     self.actor_losses,
            "critic_losses":    self.critic_losses,
            "update_count":     self._update_count,
        }, path)
        logger.info(f"PPO checkpoint saved to {path}")

    def load(self, path: str) -> None:
        """Load a PPO checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(ckpt["actor_critic"])
        self.actor_critic_old.load_state_dict(ckpt["actor_critic_old"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.actor_losses  = ckpt.get("actor_losses", [])
        self.critic_losses = ckpt.get("critic_losses", [])
        self._update_count = ckpt.get("update_count", 0)
        logger.info(f"PPO checkpoint loaded from {path}")

    @classmethod
    def from_config(cls, config: Dict, state_dim: int) -> "LowerLayerPPO":
        """Instantiate from config dict."""
        ppo_cfg = config.get("ppo", {})
        device  = config.get("experiment", {}).get("device", "cpu")
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
        except ImportError:
            device = "cpu"

        return cls(
            state_dim=state_dim,
            n_actions=config.get("smdp", {}).get("n_macro_actions", 4),
            hidden_layers=ppo_cfg.get("hidden_layers", [256, 256]),
            learning_rate=ppo_cfg.get("learning_rate", 1e-4),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            value_coef=ppo_cfg.get("value_coef", 0.5),
            entropy_coef=ppo_cfg.get("entropy_coef", 0.01),
            n_steps=ppo_cfg.get("n_steps", 256),
            mini_batch_size=ppo_cfg.get("mini_batch_size", 5),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            device=device,
        )


