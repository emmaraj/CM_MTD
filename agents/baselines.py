"""
Baseline Implementations for Comparison with CM-MTD.

Paper baselines (Section VII-B):
  1. STATIC:        No mutation — static IP addresses and routes.
  2. HAM_ONLY:      Always deploy HAM (FRVM-style random virtual IP rotation).
  3. RM_ONLY:       Always deploy RM.
  4. RRT+FRVM:      Adaptive-period RM [31] + random virtual IP multiplexing [28,29].
  5. DQN_RM+FRVM:   DQN-based RM [32] + FRVM [28,29].

All baselines share the same evaluation interface:
  baseline.select_action(state) → macro_action int

This allows plugging any baseline into the same evaluation loop.
"""
import logging
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

from agents.replay_buffer import ReplayBuffer
from models.networks import DuelingQNetwork

logger = logging.getLogger("cm_mtd.baselines")

# Macro-action constants (matching dtmn_env and reward_functions)
MACRO_STATIC   = 0
MACRO_HAM_ONLY = 1
MACRO_RM_ONLY  = 2
MACRO_HAM_RM   = 3


# ─── Base Class ──────────────────────────────────────────────────────────────

class BaselineAgent:
    """Abstract base class for all MTD baselines."""

    name: str = "Baseline"

    def select_action(self, state: np.ndarray) -> int:
        raise NotImplementedError

    def update(self, *args, **kwargs) -> None:
        pass  # Most baselines are policy-free (no learning)

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


# ─── 1. Static Baseline ───────────────────────────────────────────────────────

class StaticBaseline(BaselineAgent):
    """
    Static network — no mutation, fixed IP addresses and routes.
    Macro-action always = STATIC (o_s).
    Represents the pure reactive defense scenario.
    """
    name = "STATIC"

    def select_action(self, state: np.ndarray) -> int:
        return MACRO_STATIC


# ─── 2. HAM-Only Baseline (FRVM) ─────────────────────────────────────────────

class HAMOnlyBaseline(BaselineAgent):
    """
    HAM-only: always deploy host address mutation (FRVM-style).
    Macro-action always = HAM_ONLY (o_a).

    Simulates FRVM [28,29]: flexible random virtual IP multiplexing.
    Mutations happen at a fixed random period.
    """
    name = "HAM_ONLY"

    def select_action(self, state: np.ndarray) -> int:
        return MACRO_HAM_ONLY


# ─── 3. RM-Only Baseline ─────────────────────────────────────────────────────

class RMOnlyBaseline(BaselineAgent):
    """
    RM-only: always deploy route mutation.
    Macro-action always = RM_ONLY (o_r).
    """
    name = "RM_ONLY"

    def select_action(self, state: np.ndarray) -> int:
        return MACRO_RM_ONLY


# ─── 4. RRT+FRVM Baseline ────────────────────────────────────────────────────

class RRTFRVMBaseline(BaselineAgent):
    """
    RRT+FRVM: random adaptive-period route mutation + random virtual IP rotation.

    RRT  [31]: Cost-effective RM with adaptive mutation period.
               Mutation happens with probability p(state) ∝ attack severity.
    FRVM [28,29]: Flexible random virtual IP multiplexing (pure HAM).

    Since both MTD schemes are active, this maps to MACRO_HAM_RM.
    The adaptation is in *when* to trigger — here we use a simple
    state-dependent probability threshold.
    """
    name = "RRT_FRVM"

    def __init__(self, mutation_prob: float = 0.5, seed: int = 42) -> None:
        """
        Args:
            mutation_prob: Base probability of applying both HAM and RM.
            seed: Random seed.
        """
        self.base_prob = mutation_prob
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: np.ndarray) -> int:
        """
        Adaptive decision: deploy HAM+RM with probability proportional
        to estimated attack severity (dominant attack probability).
        """
        # Estimate attack severity from state (probability of non-benign events)
        if state.size > 0:
            state_mat = state.reshape(-1, state.size // max(1, len(state) // 8))
            # Mean probability of non-benign events
            benign_prob = float(state_mat[:, 0].mean()) if state_mat.shape[1] > 0 else 0.5
            attack_severity = 1.0 - benign_prob
        else:
            attack_severity = self.base_prob

        # Adaptive mutation probability
        trigger_prob = min(0.9, self.base_prob + attack_severity * 0.4)

        if self.rng.random() < trigger_prob:
            return MACRO_HAM_RM   # Both HAM and RM (o_c)
        else:
            return MACRO_STATIC   # Stay static this slot


# ─── 5. DQN-RM+FRVM Baseline ─────────────────────────────────────────────────

class DQNRMFRVMBaseline(BaselineAgent):
    """
    DQN-RM+FRVM: DRL-based route mutation [32] + FRVM [28,29].

    DQ-RM [32]: Learns when to mutate routes using DQN.
                Here simplified to DQN choosing between STATIC and RM_ONLY,
                then FRVM (HAM) always runs in parallel → effective action is
                either HAM_ONLY or HAM_RM.
    FRVM always active → HAM is always on.
    DQN decides whether to also apply RM.

    Learning uses the same DQN architecture as the upper layer
    but with only 2 effective choices:
        DQN action 0 → HAM_ONLY  (FRVM active, no RM)
        DQN action 1 → HAM_RM    (FRVM + route mutation)
    """
    name = "DQN_RM_FRVM"

    def __init__(
        self,
        state_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update: int = 10,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device(device)
        self._episode = 0
        self.rng = np.random.default_rng(seed)

        # DQN with 2 actions: [HAM_ONLY, HAM_RM]
        self.q_net = DuelingQNetwork(
            state_dim=state_dim, n_actions=2, hidden_layers=[256]
        ).to(self.device)
        self.target_net = DuelingQNetwork(
            state_dim=state_dim, n_actions=2, hidden_layers=[256]
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(
            capacity=buffer_size, state_dim=state_dim, device=str(self.device)
        )
        self.losses: List[float] = []

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """DQN selects whether RM is active (FRVM/HAM always on)."""
        if not deterministic and self.rng.random() < self.epsilon:
            dqn_action = int(self.rng.integers(0, 2))
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_vals = self.q_net(state_t)
            dqn_action = int(q_vals.argmax(dim=1).item())

        # Map DQN action → macro-action
        # 0 → HAM_ONLY (FRVM only), 1 → HAM_RM (FRVM + route mutation)
        return MACRO_HAM_ONLY if dqn_action == 0 else MACRO_HAM_RM

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        # Map macro-action back to DQN action index
        dqn_action = 0 if action == MACRO_HAM_ONLY else 1
        self.buffer.push(state, dqn_action, reward, next_state, done)

    def update(self) -> Optional[float]:
        if not self.buffer.is_ready(self.batch_size):
            return None
        batch = self.buffer.sample(self.batch_size)
        states      = batch["states"]
        actions     = batch["actions"]
        rewards     = batch["rewards"]
        next_states = batch["next_states"]
        dones       = batch["dones"]

        q_cur  = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next   = self.target_net(next_states)
            q_target = rewards + self.gamma * q_next.max(1)[0] * (1 - dones)

        loss = F.smooth_l1_loss(q_cur, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self.losses.append(loss.item())
        return loss.item()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self._episode += 1
        if self._episode % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


# ─── Baseline Factory ─────────────────────────────────────────────────────────

def build_all_baselines(
    state_dim: int,
    config: Optional[Dict] = None,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, BaselineAgent]:
    """
    Build all comparison baselines.

    Args:
        state_dim: Observation dimension (must match environment).
        config:    Configuration dict (optional).
        device:    Compute device for DQN-based baselines.
        seed:      Random seed.

    Returns:
        Dict mapping baseline name → BaselineAgent instance.
    """
    baselines: Dict[str, BaselineAgent] = {
        "STATIC":     StaticBaseline(),
        "HAM_ONLY":   HAMOnlyBaseline(),
        "RM_ONLY":    RMOnlyBaseline(),
        "RRT_FRVM":   RRTFRVMBaseline(mutation_prob=0.5, seed=seed),
        "DQN_RM_FRVM": DQNRMFRVMBaseline(
            state_dim=state_dim,
            device=device,
            seed=seed,
        ),
    }
    logger.info(f"Built {len(baselines)} baselines: {list(baselines.keys())}")
    return baselines


def run_baseline_episode(
    baseline: BaselineAgent,
    env: "DTMNEnvironment",
    seed: int = 42,
    train_dqn_rm: bool = False,
) -> Dict[str, float]:
    """
    Run a single episode with a baseline policy.

    Args:
        baseline:      Baseline agent to evaluate.
        env:           DTMN environment.
        seed:          Episode random seed.
        train_dqn_rm:  If True and baseline is DQNRMFRVMBaseline, call update().

    Returns:
        Episode metrics dict.
    """
    state, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    prev_state = state.copy()

    while not done:
        action = baseline.select_action(state)
        next_state, reward, done, _, info = env.step(action)

        # Train DQN-RM during evaluation episodes (it learns online)
        if train_dqn_rm and isinstance(baseline, DQNRMFRVMBaseline):
            baseline.store_transition(prev_state, action, reward, next_state, done)
            baseline.update()

        total_reward += reward
        prev_state = state
        state = next_state

    if train_dqn_rm and isinstance(baseline, DQNRMFRVMBaseline):
        baseline.decay_epsilon()

    dsr = env.get_episode_dsr()
    rtt = float(np.mean(env._ep_rtt)) if env._ep_rtt else 0.0
    plr = float(np.mean(env._ep_plr)) if env._ep_plr else 0.0

    return {
        "dsr":    dsr,
        "reward": total_reward,
        "rtt":    rtt,
        "plr":    plr,
    }
