"""
DTMN Gymnasium Environment — Section III-C (SMDP Model).

Implements the Semi-Markov Decision Process (SMDP) for collaborative MTD scheduling.

State Space:
    S_t = {e^{t+1}_1, ..., e^{t+1}_n}
    Predicted security events for all n nodes (LSTM output), flattened.
    State vector: [n_nodes × n_classes]

Action Space (Macro-Actions O):
    0 = STATIC    (o_s): static IP + routes
    1 = HAM_ONLY  (o_a): host address mutation only
    2 = RM_ONLY   (o_r): route mutation only
    3 = HAM_RM    (o_c): both HAM and RM

Reward:
    R_total = R_d + R_c  (Eq. 3)
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Gymnasium with graceful fallback
from utils.compat import GYM_AVAILABLE
if GYM_AVAILABLE:
    import gymnasium as gym
    from gymnasium import spaces
    _BaseEnv = gym.Env
else:
    import sys
    gym = sys.modules.get("gymnasium")
    spaces = sys.modules.get("gymnasium.spaces")
    _BaseEnv = gym.Env if gym else object

from environment.network_model import NetworkModel
from environment.reward_functions import RewardFunction, RewardComponents
from datasets.cicids2017_loader import N_CLASSES

logger = logging.getLogger("cm_mtd.env")

MACRO_NAMES = {0: "STATIC", 1: "HAM_ONLY", 2: "RM_ONLY", 3: "HAM_RM"}


class DTMNEnvironment(_BaseEnv):
    """
    Digital Twin Mobile Network (DTMN) Gymnasium-compatible environment.

    Simulates the SMDP for collaborative MTD scheduling. Attack events
    are driven by a pre-loaded sequence (CICIDS2017 or synthetic), and
    the LSTM predictor provides the network state vector.

    Args:
        network_config:      Network model config dict.
        reward_config:       Reward function config dict.
        lstm_predictor:      Trained LSTMAttackPredictor (or None).
        attack_sequence:     Integer array of attack labels [T].
        sequence_length:     LSTM look-back L.
        n_steps_per_episode: T — macro-action steps per episode.
        K_steps:             K — inner time slots per macro-action.
        seed:                Random seed.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        network_config: Dict,
        reward_config: Dict,
        lstm_predictor=None,
        attack_sequence: Optional[np.ndarray] = None,
        sequence_length: int = 10,
        n_steps_per_episode: int = 25,
        K_steps: int = 5,
        seed: int = 42,
    ) -> None:
        if GYM_AVAILABLE:
            super().__init__()

        self.net_cfg = network_config
        self.L = sequence_length
        self.T = n_steps_per_episode
        self.K = K_steps
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)

        # Network model
        self.network = NetworkModel(
            n_nodes=network_config.get("n_nodes", 12),
            n_switches=network_config.get("n_switches", 12),
            n_flows=network_config.get("n_flows", 10),
            n_ip_spaces=network_config.get("n_ip_spaces", 30),
            waxman_alpha=network_config.get("waxman_alpha", 0.2),
            waxman_beta=network_config.get("waxman_beta", 0.15),
            seed=seed,
        )
        self.n_nodes   = self.network.n
        self.n_classes = N_CLASSES

        # Reward function
        self.reward_fn = RewardFunction.from_config({"reward": reward_config})

        # LSTM predictor (optional)
        self.lstm = lstm_predictor

        # Attack sequence
        if attack_sequence is not None:
            self.attack_seq = np.asarray(attack_sequence, dtype=np.int64)
        else:
            self.attack_seq = self._generate_default_attack_sequence()
        self.seq_len = len(self.attack_seq)

        # ── Observation / action spaces ───────────────────────────────────
        obs_dim = self.n_nodes * self.n_classes
        if GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(4)

        # Episode state
        self._t: int = 0
        self._episode: int = 0
        self._seq_ptr: int = 0
        self._event_history = np.zeros((self.n_nodes, self.L), dtype=np.int64)
        self._state = np.zeros(obs_dim, dtype=np.float32)

        # Accumulators
        self._ep_n_scanned: float = 0.0
        self._ep_n_compromised: float = 0.0
        self._ep_total_scanned: float = 0.0
        self._ep_total_switches: float = 0.0
        self._ep_rewards: List[float] = []
        self._ep_rtt: List[float] = []
        self._ep_plr: List[float] = []

        logger.info(
            "DTMNEnvironment | obs_dim=%d | n_actions=4 | T=%d | K=%d",
            obs_dim, self.T, self.K,
        )

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed_val = seed

        self._t = 0
        self._episode += 1

        safe_range = max(1, self.seq_len - self.T * self.K - self.L - 1)
        self._seq_ptr = int(self.rng.integers(0, safe_range))

        self.network.reset(seed=self.seed_val)

        # Warm-start history with per-node jitter (fixes identical-start bug)
        for node in range(self.n_nodes):
            node_offset = int(self.rng.integers(0, max(1, self.L)))
            for lt in range(self.L):
                ptr = (self._seq_ptr + node_offset + lt) % self.seq_len
                self._event_history[node, lt] = self.attack_seq[ptr]
        self._seq_ptr += self.L

        # Reset accumulators
        self._ep_n_scanned = 0.0
        self._ep_n_compromised = 0.0
        self._ep_total_scanned = 0.0
        self._ep_total_switches = 0.0
        self._ep_rewards = []
        self._ep_rtt = []
        self._ep_plr = []

        self._state = self._compute_state()
        return self._state.copy(), {}

    def step(
        self, macro_action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one macro-action for exactly K inner time slots.

        Returns K reward values in info["rewards"] so the HDRL agent
        can feed individual (s, a, r) tuples to the PPO rollout buffer.
        """
        total_reward = 0.0
        rewards_per_k: List[float] = []
        n_scanned_per_k: List[float] = []
        n_comp_per_k: List[float] = []

        # ── K inner steps ─────────────────────────────────────────────────
        for k_idx in range(self.K):
            attack_type = int(self.attack_seq[self._seq_ptr % self.seq_len])
            self._seq_ptr += 1

            net_result = self.network.step_attack(
                attack_type=attack_type,
                macro_action=macro_action,
                predicted_events=self._state.reshape(self.n_nodes, self.n_classes),
            )

            rc: RewardComponents = self.reward_fn.compute(
                macro_action=macro_action,
                n_nodes_scanned=net_result["n_scanned"],
                n_switches_compromised=net_result["n_switches_compromised"],
                ham_resource_costs=net_result["ham_costs"],
                rm_resource_costs=net_result["rm_costs"],
                attack_success=net_result["attack_success"],
            )

            step_reward = rc.R_total
            total_reward += step_reward
            rewards_per_k.append(float(step_reward))
            n_scanned_per_k.append(float(net_result["n_scanned"]))
            n_comp_per_k.append(float(net_result["n_switches_compromised"]))

            # Accumulate episode stats
            self._ep_n_scanned     += net_result["n_scanned"]
            self._ep_n_compromised += net_result["n_switches_compromised"]
            self._ep_total_scanned  += net_result["total_scanned"]
            self._ep_total_switches += net_result["total_switches_route"]

            # Update event history for each node
            for node in range(self.n_nodes):
                self._event_history[node] = np.roll(self._event_history[node], -1)
                self._event_history[node, -1] = attack_type

            # Network performance
            self._ep_rtt.append(self.network.compute_rtt(macro_action))
            self._ep_plr.append(self.network.compute_plr(macro_action))

        # Advance outer time step
        self._t += 1
        self._ep_rewards.append(total_reward)
        terminated = self._t >= self.T
        truncated  = False

        # Next state
        self._state = self._compute_state()

        info: Dict[str, Any] = {
            "macro_action":      macro_action,
            "macro_name":        MACRO_NAMES[macro_action],
            "rewards":           rewards_per_k,           # always len == K
            "n_scanned_steps":   n_scanned_per_k,
            "n_compromised_steps": n_comp_per_k,
        }
        if terminated:
            info.update(self._compute_episode_summary())

        return self._state.copy(), float(total_reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        state_mat = self._state.reshape(self.n_nodes, self.n_classes)
        dominant  = np.argmax(state_mat, axis=1)
        logger.info("Ep %d | Step %d/%d | dominant attacks: %s",
                    self._episode, self._t, self.T, dominant)

    def close(self) -> None:
        pass

    # ── State computation ─────────────────────────────────────────────────────

    def _compute_state(self) -> np.ndarray:
        """LSTM prediction → SMDP state S_t."""
        if self.lstm is not None:
            try:
                probs = self.lstm.predict_proba(self._event_history)  # [n_nodes, n_classes]
                return probs.flatten().astype(np.float32)
            except Exception:
                pass

        # Fallback: empirical frequency in recent history
        state = np.zeros((self.n_nodes, self.n_classes), dtype=np.float32)
        for node in range(self.n_nodes):
            counts = np.bincount(self._event_history[node], minlength=self.n_classes).astype(np.float32)
            total  = counts.sum()
            state[node] = counts / max(total, 1.0)
        return state.flatten()

    # ── Episode summary ───────────────────────────────────────────────────────

    def _compute_episode_summary(self) -> Dict[str, float]:
        from utils.metrics import compute_dsr
        dsr = compute_dsr(
            np.array([self._ep_n_scanned]),
            np.array([self._ep_n_compromised]),
            np.array([self._ep_total_scanned]),
            np.array([self._ep_total_switches]),
        )
        return {
            "episode_dsr":          float(dsr[0]),
            "episode_total_reward": float(sum(self._ep_rewards)),
            "episode_mean_reward":  float(np.mean(self._ep_rewards)) if self._ep_rewards else 0.0,
            "episode_mean_rtt":     float(np.mean(self._ep_rtt)) if self._ep_rtt else 0.0,
            "episode_mean_plr":     float(np.mean(self._ep_plr)) if self._ep_plr else 0.0,
        }

    def get_episode_dsr(self) -> float:
        from utils.metrics import compute_dsr
        return float(compute_dsr(
            np.array([self._ep_n_scanned]),
            np.array([self._ep_n_compromised]),
            np.array([self._ep_total_scanned]),
            np.array([self._ep_total_switches]),
        )[0])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _generate_default_attack_sequence(self, n_steps: int = 50_000) -> np.ndarray:
        from datasets.sequence_builder import SyntheticDataGenerator
        gen = SyntheticDataGenerator(n_classes=N_CLASSES, seed=self.seed_val)
        _, y = gen.generate_event_sequence(
            n_nodes=self.n_nodes, n_steps=n_steps, sequence_length=self.L
        )
        return y.astype(np.int64)

    def set_attack_sequence(self, sequence: np.ndarray) -> None:
        self.attack_seq = sequence.astype(np.int64)
        self.seq_len    = len(self.attack_seq)

    def get_obs_dim(self) -> int:
        return self.n_nodes * self.n_classes

    def get_n_actions(self) -> int:
        return 4
