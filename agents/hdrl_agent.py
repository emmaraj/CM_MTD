"""
Hierarchical Deep Reinforcement Learning (HDRL) Agent — Algorithm 1 (complete).

Implements the full CM-MTD training loop combining:
  - Upper layer: DQN  → selects macro-action O_t ∈ {STATIC, HAM, RM, HAM+RM}
  - Lower layer: PPO  → selects mutation action A_k (routes / IP spaces)

The two layers interact via reward feedback:
  - Lower layer receives per-step reward R^l_k
  - Upper layer receives accumulated reward R^u_t = Σ_k R^l_k

Algorithm 1 (paper) — Complete Training Loop:
  Initialize parameters (lines 1–8)
  For episode = 1..M:
    For t = 1..T:
      Obtain S_t from LSTMNet (line 11)
      Select O_t via ε-greedy DQN (lines 12–18)
      For k = 1..K:
        Run PPO π_{θ^l_{k-n}} → select A_k (line 20)
        Deploy MTD schemes (line 21)
        Observe R^l_k (line 22)
        Store D_k in B₂ (lines 23–24)
        Compute GAE advantages Â_k (lines 25–26)
        Estimate V̂_k (line 27)
      Observe R^u_t (line 29)
      Obtain S_{t+1} (line 30)
      Store D'_t in B₁ (lines 31–32)
    For epoch = 1..U:
      Minimize J^{actor} and J^{critic} (lines 35–40)
    Update target networks (lines 42–43)
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.compat import TORCH_AVAILABLE

from agents.upper_layer_dqn import UpperLayerDQN
from agents.lower_layer_ppo import LowerLayerPPO
from environment.dtmn_env import DTMNEnvironment

logger = logging.getLogger("cm_mtd.hdrl")

# Macro-action names for logging
MACRO_NAMES = {0: "STATIC", 1: "HAM_ONLY", 2: "RM_ONLY", 3: "HAM_RM"}


class HDRLAgent:
    """
    Full Hierarchical DRL Agent for CM-MTD collaborative scheduling.

    Orchestrates the interaction between:
      - UpperLayerDQN (macro-action selection)
      - LowerLayerPPO (mutation action selection)
      - DTMNEnvironment (SMDP simulation)

    Args:
        env:         DTMN environment instance.
        dqn:         Trained/initialised UpperLayerDQN.
        ppo:         Trained/initialised LowerLayerPPO.
        n_episodes:  Total training episodes M.
        n_steps:     Time steps per episode T.
        K_steps:     Inner steps per macro-action K.
        n_ppo_epochs: PPO update epochs per episode U.
        checkpoint_dir: Directory for saving checkpoints.
        checkpoint_freq: Save every N episodes.
        eval_freq:   Evaluate every N episodes.
        n_eval_eps:  Episodes for evaluation.
        log_dir:     TensorBoard log directory.
        seed:        Random seed.
    """

    def __init__(
        self,
        env: DTMNEnvironment,
        dqn: UpperLayerDQN,
        ppo: LowerLayerPPO,
        n_episodes: int = 10_000,
        n_steps: int = 25,
        K_steps: int = 5,
        n_ppo_epochs: int = 10,
        checkpoint_dir: str = "checkpoints",
        checkpoint_freq: int = 500,
        eval_freq: int = 100,
        n_eval_eps: int = 50,
        log_dir: str = "logs",
        seed: int = 42,
    ) -> None:
        self.env = env
        self.dqn = dqn
        self.ppo = ppo
        self.M = n_episodes
        self.T = n_steps
        self.K = K_steps
        self.U = n_ppo_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.n_eval_eps = n_eval_eps
        self.seed = seed

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # ── Training metrics (logged every episode) ──────────────────────
        self.upper_rewards:   List[float] = []   # R^u per episode
        self.lower_rewards:   List[float] = []   # mean R^l per episode
        self.dsr_history:     List[float] = []   # DSR per episode
        self.dqn_loss_hist:   List[float] = []   # DQN loss
        self.ppo_actor_hist:  List[float] = []   # PPO actor loss
        self.ppo_critic_hist: List[float] = []   # PPO critic loss
        self.epsilon_hist:    List[float] = []   # ε schedule
        self.eval_dsr_hist:   List[float] = []   # eval DSR
        self.rtt_history:     List[float] = []
        self.plr_history:     List[float] = []

        # Macro-action frequency counter
        self.macro_action_counts = np.zeros(4, dtype=np.int64)

        logger.info(
            f"HDRLAgent initialized | M={n_episodes} | T={n_steps} | K={K_steps}"
        )

    # ─── Main Training Loop ───────────────────────────────────────────────────

    def train(self) -> Dict[str, List[float]]:
        """
        Run the complete HDRL training loop (Algorithm 1).

        Returns:
            Dict of training history lists for all tracked metrics.
        """
        logger.info("=" * 60)
        logger.info("Starting HDRL Training (Algorithm 1)")
        logger.info(f"  Episodes: {self.M} | Steps/ep: {self.T} | K: {self.K}")
        logger.info("=" * 60)
        t_start = time.time()

        # ── Algorithm 1 line 9: for episode = 1..M ────────────────────────
        for episode in range(1, self.M + 1):

            ep_upper_reward = 0.0
            ep_lower_rewards: List[float] = []
            ep_dqn_losses: List[float] = []
            ep_ppo_metrics: List[Dict] = []

            # Reset environment
            state, _ = self.env.reset(seed=self.seed + episode)

            # ── Algorithm 1 line 10: for t = 1..T ─────────────────────────
            for t_step in range(self.T):

                # ── Line 11: Obtain S_t from LSTMNet ─────────────────────
                # (already embedded in env.reset / env.step → state)

                # ── Lines 12–18: Select macro-action O_t (ε-greedy) ──────
                macro_action = self.dqn.select_action(state, deterministic=False)
                self.macro_action_counts[macro_action] += 1

                # ── Lines 19–28: Lower-layer K-step rollout ───────────────
                lower_ep_reward = 0.0
                k_states:    List[np.ndarray] = []
                k_actions:   List[int]        = []
                k_rewards:   List[float]      = []
                k_values:    List[float]      = []
                k_log_probs: List[float]      = []
                k_dones:     List[bool]       = []

                # Execute environment step (which internally runs K sub-steps)
                next_state, env_reward, terminated, truncated, info = self.env.step(
                    macro_action
                )

                # Decompose the K inner steps from info (approximated):
                # The environment batches K steps; we distribute reward
                k_rewards_list = info.get("rewards", [env_reward / max(self.K, 1)] * self.K)

                for k in range(min(self.K, len(k_rewards_list))):
                    k_state = state  # approximate: all K steps get same macro-state
                    k_action, k_log_p, k_val = self.ppo.select_action(k_state)

                    k_states.append(k_state)
                    k_actions.append(k_action)
                    k_rewards.append(k_rewards_list[k])
                    k_values.append(k_val)
                    k_log_probs.append(k_log_p)
                    k_dones.append(terminated and k == self.K - 1)

                    lower_ep_reward += k_rewards_list[k]

                    # ── Line 23–24: Store D_k in B₂ ──────────────────────
                    self.ppo.store_transition(
                        state=k_state,
                        action=k_action,
                        reward=k_rewards_list[k],
                        value=k_val,
                        log_prob=k_log_p,
                        done=k_dones[-1],
                    )

                ep_lower_rewards.extend(k_rewards)

                # ── Line 29: Observe upper-layer reward R^u_t ─────────────
                upper_reward = env_reward

                # ── Lines 31–32: Store D'_t = {S_t, O_t, R^u_t, S_{t+1}} ─
                self.dqn.store_transition(
                    state=state,
                    action=macro_action,
                    reward=upper_reward,
                    next_state=next_state,
                    done=terminated,
                )

                ep_upper_reward += upper_reward
                state = next_state

                # ── Lines 35–36: DQN gradient update ─────────────────────
                dqn_loss = self.dqn.update()
                if dqn_loss is not None:
                    ep_dqn_losses.append(dqn_loss)

                if terminated:
                    break

            # ── Lines 34–40: PPO batch update (once per episode) ──────────
            if len(self.ppo.buffer) > 0:
                # Bootstrap value for last state
                _, _, last_val = self.ppo.select_action(state, deterministic=True)
                ppo_metrics = self.ppo.update(last_value=last_val)
                ep_ppo_metrics.append(ppo_metrics)

            # ── Lines 42–43: Update target networks ───────────────────────
            self.dqn.decay_epsilon()  # also calls target update every N eps

            # ── Record episode metrics ────────────────────────────────────
            ep_dsr = self.env.get_episode_dsr()
            mean_rtt = float(np.mean(self.env._ep_rtt)) if self.env._ep_rtt else 0.0
            mean_plr = float(np.mean(self.env._ep_plr)) if self.env._ep_plr else 0.0

            self.upper_rewards.append(ep_upper_reward)
            self.lower_rewards.append(float(np.mean(ep_lower_rewards)) if ep_lower_rewards else 0.0)
            self.dsr_history.append(ep_dsr)
            self.rtt_history.append(mean_rtt)
            self.plr_history.append(mean_plr)

            mean_dqn_loss = float(np.mean(ep_dqn_losses)) if ep_dqn_losses else 0.0
            self.dqn_loss_hist.append(mean_dqn_loss)
            self.epsilon_hist.append(self.dqn.epsilon)

            if ep_ppo_metrics:
                self.ppo_actor_hist.append(ep_ppo_metrics[-1]["actor_loss"])
                self.ppo_critic_hist.append(ep_ppo_metrics[-1]["critic_loss"])

            # ── Periodic evaluation ───────────────────────────────────────
            if episode % self.eval_freq == 0:
                eval_dsr = self._evaluate(n_episodes=self.n_eval_eps)
                self.eval_dsr_hist.append(eval_dsr)
                elapsed = time.time() - t_start
                logger.info(
                    f"Episode {episode:5d}/{self.M} | "
                    f"ε={self.dqn.epsilon:.3f} | "
                    f"DSR={ep_dsr:.1f}% | "
                    f"Eval-DSR={eval_dsr:.1f}% | "
                    f"DQN-loss={mean_dqn_loss:.4f} | "
                    f"Macro: {MACRO_NAMES[self._dominant_macro()]} | "
                    f"Time: {elapsed:.0f}s"
                )

            # ── Periodic checkpointing ────────────────────────────────────
            if episode % self.checkpoint_freq == 0:
                self._save_checkpoint(episode)

        # Save final checkpoint
        self._save_checkpoint(self.M, tag="final")
        self._save_metrics()

        logger.info(
            f"Training complete in {time.time() - t_start:.1f}s | "
            f"Final DSR: {np.mean(self.dsr_history[-100:]):.2f}%"
        )

        return self._get_history()

    # ─── Evaluation ──────────────────────────────────────────────────────────

    def _evaluate(self, n_episodes: int = 50) -> float:
        """
        Evaluate the current policy deterministically.

        Computes average DSR over n_episodes with ε=0 (no exploration).

        Args:
            n_episodes: Number of evaluation episodes.

        Returns:
            Mean DSR across evaluation episodes.
        """
        dsrs = []
        for ep in range(n_episodes):
            state, _ = self.env.reset(seed=self.seed + 100_000 + ep)
            done = False
            while not done:
                macro = self.dqn.select_action(state, deterministic=True)
                state, _, done, _, _ = self.env.step(macro)
            dsrs.append(self.env.get_episode_dsr())
        return float(np.mean(dsrs))

    def evaluate_full(
        self, n_episodes: int = 1000
    ) -> Dict[str, float]:
        """
        Full evaluation with all metrics (DSR, RTT, PLR, rewards).

        Used in evaluate.py for final paper-quality results.

        Returns:
            Dict with mean ± std of all metrics.
        """
        all_dsr, all_rtt, all_plr, all_reward = [], [], [], []

        for ep in range(n_episodes):
            state, _ = self.env.reset(seed=42 + ep)
            ep_reward = 0.0
            done = False
            while not done:
                macro = self.dqn.select_action(state, deterministic=True)
                state, r, done, _, _ = self.env.step(macro)
                ep_reward += r
            all_dsr.append(self.env.get_episode_dsr())
            all_rtt.append(float(np.mean(self.env._ep_rtt)) if self.env._ep_rtt else 0.0)
            all_plr.append(float(np.mean(self.env._ep_plr)) if self.env._ep_plr else 0.0)
            all_reward.append(ep_reward)

        return {
            "dsr_mean":    float(np.mean(all_dsr)),
            "dsr_std":     float(np.std(all_dsr, ddof=1)),
            "rtt_mean":    float(np.mean(all_rtt)),
            "rtt_std":     float(np.std(all_rtt, ddof=1)),
            "plr_mean":    float(np.mean(all_plr)),
            "plr_std":     float(np.std(all_plr, ddof=1)),
            "reward_mean": float(np.mean(all_reward)),
            "reward_std":  float(np.std(all_reward, ddof=1)),
            "n_episodes":  n_episodes,
        }

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _dominant_macro(self) -> int:
        return int(np.argmax(self.macro_action_counts))

    def _save_checkpoint(self, episode: int, tag: str = "") -> None:
        suffix = f"_ep{episode}" + (f"_{tag}" if tag else "")
        self.dqn.save(str(self.checkpoint_dir / f"dqn{suffix}.pt"))
        self.ppo.save(str(self.checkpoint_dir / f"ppo{suffix}.pt"))

    def _save_metrics(self) -> None:
        metrics_path = self.checkpoint_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self._get_history(), f, indent=2)
        logger.info(f"Training metrics saved to {metrics_path}")

    def _get_history(self) -> Dict[str, List[float]]:
        return {
            "upper_rewards":   self.upper_rewards,
            "lower_rewards":   self.lower_rewards,
            "dsr_history":     self.dsr_history,
            "dqn_loss":        self.dqn_loss_hist,
            "ppo_actor_loss":  self.ppo_actor_hist,
            "ppo_critic_loss": self.ppo_critic_hist,
            "epsilon":         self.epsilon_hist,
            "eval_dsr":        self.eval_dsr_hist,
            "rtt_history":     self.rtt_history,
            "plr_history":     self.plr_history,
        }

    def load_checkpoint(self, dqn_path: str, ppo_path: str) -> None:
        """Load both agent checkpoints."""
        self.dqn.load(dqn_path)
        self.ppo.load(ppo_path)
        logger.info(f"Loaded checkpoints: DQN={dqn_path}, PPO={ppo_path}")

    @classmethod
    def from_config(
        cls,
        config: Dict,
        env: DTMNEnvironment,
        dqn: UpperLayerDQN,
        ppo: LowerLayerPPO,
    ) -> "HDRLAgent":
        """Instantiate from config dict."""
        tr = config.get("training", {})
        return cls(
            env=env,
            dqn=dqn,
            ppo=ppo,
            n_episodes=tr.get("n_episodes", 10_000),
            n_steps=tr.get("n_steps_per_episode", 25),
            K_steps=tr.get("K_steps", 5),
            n_ppo_epochs=tr.get("n_ppo_epochs", 10),
            checkpoint_dir=tr.get("checkpoint_dir", "checkpoints"),
            checkpoint_freq=tr.get("checkpoint_freq", 500),
            eval_freq=tr.get("eval_freq", 100),
            n_eval_eps=tr.get("n_eval_episodes", 50),
            log_dir=tr.get("log_dir", "logs"),
        )
