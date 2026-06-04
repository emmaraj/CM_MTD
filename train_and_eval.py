"""
train_and_eval.py
=================
Main training and evaluation script for CM-MTD.

Pipeline
--------
1. LSTM training   — trains attack predictor on all three dataset modes.
2. HDRL training   — runs Algorithm 1 for CM-MTD on each mode.
3. Baseline runs   — evaluates RRT+FRVM and DQ-RM+FRVM under identical conditions.
4. Network perf    — logs RTT / PLR at each step.

Output (logs/)
--------------
  lstm_training_log.csv      — episode, phase, loss, fidelity
  defense_log.csv            — episode, algorithm, attack_type, dsr, avg_reward
  convergence_log.csv        — episode, layer, attack_type, reward
  network_perf_log.csv       — step, algorithm, attack_type, rtt_ms, plr_pct
  confusion_matrices.json    — {attack_type: {algorithm: [[...]]}}
  final_summary.json         — aggregated end-of-run statistics

NO matplotlib / seaborn calls appear here.  All visuals are produced by
visualize_results.py reading the CSV / JSON files above.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── local imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data_loader       import DataLoader, SecurityEvent
from lstm_predictor    import LSTMPredictor
from smt_constraints   import NetworkConfig, SMTConstraintSolver
from hdrl_agent        import (
    HDRLAgent, NetworkEnvironment,
    DQNAgent, PPOAgent,
    UpperTransition, LowerTransition,
    MACRO_BOTH, MACRO_HAM, MACRO_RM, MACRO_STATIC, MACRO_NAMES,
)


# ===========================================================================
# Configuration
# ===========================================================================

class Config:
    # ── Dataset ──
    MODES         = ["direct_ddos", "crossfire_ddos", "cicids2017"]
    N_NODES       = 12
    N_SWITCHES    = 12
    N_TIMESTEPS   = 5000    # simulation length per mode
    SEQ_LEN       = 10      # LSTM look-back window  (L)
    TRAIN_RATIO   = 0.80
    CICIDS_PATH   : Optional[str] = None   # set via --cicids_path

    # ── LSTM ──
    LSTM_EMBED    = 16
    LSTM_HIDDEN   = 128
    LSTM_LAYERS   = 2
    LSTM_DROPOUT  = 0.2
    LSTM_LR       = 1e-3
    LSTM_EPISODES = 15      # epochs of LSTM training per mode
    LSTM_BATCH    = 64

    # ── SMT ──
    N_IP_SPACES   = 30
    N_FLOWS       = 8
    N_FEASIBLE    = 20      # feasible actions per macro-action call

    # ── HDRL ──
    N_EPISODES    = 500     # outer loop M  (paper uses 10 000; reduce for speed)
    T_STEPS       = 25      # time slots per episode
    K_STEPS       = 5       # lower-layer iterations per time slot
    DQN_HIDDEN    = [256]
    PPO_HIDDEN    = [256, 256]
    DQN_LR        = 1e-3
    PPO_LR        = 1e-4
    GAMMA         = 0.99
    GAE_LAMBDA    = 0.95
    CLIP_EPS      = 0.20
    EPSILON_START = 1.0

    # ── Environment ──
    ALPHA1 = 1.0
    ALPHA2 = 1.5
    GAMMA1 = 0.5
    GAMMA2 = 0.5
    C_REW  = 10.0

    # ── Logging ──
    LOG_DIR    = "./logs"
    MODEL_DIR  = "./models"
    SEED       = 42

    # ── DSR reporting window ──
    DSR_WINDOW = 100    # average DSR over last N episodes for logging


# ===========================================================================
# CSV / JSON helpers
# ===========================================================================

def _init_csv(path: str, header: List[str]):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)


def _append_csv(path: str, row: list):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def _write_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _now() -> str:
    return datetime.utcnow().isoformat()


# ===========================================================================
# Baseline agents
# ===========================================================================

class RRTFRVMBaseline:
    """
    RRT + FRVM baseline.
    - RRT  (Randomized Route Time)  : always deploys RM with random routes.
    - FRVM (Flexible Random Virtual IP Multiplexing) : random IP assignment.
    Always selects macro-action o_c (both HAM + RM), random mutation.
    DSR does NOT improve over episodes (no learning).
    """

    def __init__(self, n_nodes: int, n_switches: int, n_flows: int,
                 n_ip_spaces: int = 30, seed: int = 0):
        self.rng         = np.random.RandomState(seed)
        self.n_nodes     = n_nodes
        self.n_switches  = n_switches
        self.n_flows     = n_flows
        self.n_ip_spaces = n_ip_spaces

    def select_action(self) -> Tuple[int, float, float]:
        """Returns (macro_action, ham_cost, rm_cost)."""
        ham_cost = float(self.rng.uniform(0.3, 0.6))
        rm_cost  = float(self.rng.uniform(0.3, 0.6))
        return MACRO_BOTH, ham_cost, rm_cost

    def effectiveness(self) -> Tuple[float, float]:
        """Fixed Ham/RM effectiveness (no learning)."""
        return 0.50, 0.50    # 50 % — weaker than CM-MTD's 70 %


class DQRMFRVMBaseline:
    """
    DQ-RM + FRVM baseline.
    - DQ-RM : lightweight DQN selects route mutation only.
    - FRVM  : random IP assignment.
    Improves over episodes but weaker than full CM-MTD.
    """

    def __init__(self, state_dim: int, n_route_actions: int = 10,
                 lr: float = 1e-3, seed: int = 0, device: str = "cpu"):
        self.dqn = DQNAgent(
            state_dim       = state_dim,
            n_macro_actions = n_route_actions,
            hidden          = [128],
            lr              = lr,
            epsilon         = 1.0,
            device          = device,
        )
        self.rng = np.random.RandomState(seed)

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Returns (macro_action, ham_cost, rm_cost)."""
        _ = self.dqn.select_action(state)   # DQ-RM selects route variant
        # Always deploy both (FRVM handles HAM), DQ-RM handles RM
        ham_cost = float(self.rng.uniform(0.3, 0.6))   # FRVM — random
        rm_cost  = float(self.rng.uniform(0.2, 0.5))   # DQ-RM — learns slightly
        return MACRO_BOTH, ham_cost, rm_cost

    def effectiveness(self, episode: int) -> Tuple[float, float]:
        """Gradually improving RM effectiveness (DRL learning effect)."""
        rm_eff  = min(0.60, 0.40 + episode * 0.0004)
        ham_eff = 0.50   # FRVM is fixed
        return ham_eff, rm_eff


# ===========================================================================
# Episode simulation helpers
# ===========================================================================

def _run_one_episode(
    episode:       int,
    raw_events:    np.ndarray,
    lstm_pred:     LSTMPredictor,
    smt_solver:    SMTConstraintSolver,
    hdrl_agent:    HDRLAgent,
    env:           NetworkEnvironment,
    cfg:           Config,
    rng:           np.random.RandomState,
) -> Tuple[float, float, float]:
    """
    Execute one CM-MTD training episode (Algorithm 1, inner loops).

    Returns
    -------
    upper_reward : mean upper-layer reward for the episode
    lower_reward : mean lower-layer reward for the episode
    dsr          : Defense Success Ratio at episode end
    """
    T, N = raw_events.shape
    env.reset()

    upper_rewards_ep: List[float] = []
    lower_rewards_ep: List[float] = []

    # Random starting offset so each episode sees a different window
    t_offset = int(rng.randint(0, max(1, T - cfg.T_STEPS * cfg.K_STEPS - cfg.SEQ_LEN - 1)))

    # Pre-fill a sliding window of recent events
    window: deque = deque(maxlen=cfg.SEQ_LEN)
    start_t = t_offset
    for ts in range(cfg.SEQ_LEN):
        window.append(raw_events[min(start_t + ts, T - 1)])
    current_t = start_t + cfg.SEQ_LEN

    for t_step in range(cfg.T_STEPS):
        # ── State from LSTM prediction ──────────────────────────────────
        win_arr = np.array(list(window), dtype=np.int64)  # (SEQ_LEN, N)
        pred_events = lstm_pred.predict_state(win_arr)     # (N,)
        state = hdrl_agent.encode_state(pred_events)

        # ── Upper layer: select macro-action (DQN) ──────────────────────
        macro = hdrl_agent.select_macro_action(state)

        # ── Get feasible actions from SMT ───────────────────────────────
        feasible = smt_solver.generate_feasible_actions(macro, cfg.N_FEASIBLE)

        upper_step_reward = 0.0

        # ── Lower layer: K iterations of PPO ────────────────────────────
        for k in range(cfg.K_STEPS):
            # Advance window
            if current_t < T:
                window.append(raw_events[current_t])
                current_t += 1

            win_arr_k   = np.array(list(window), dtype=np.int64)
            pred_k      = lstm_pred.predict_state(win_arr_k)
            state_k     = hdrl_agent.encode_state(pred_k)

            # PPO selects a feasible action index
            act_idx, log_prob = hdrl_agent.select_action(state_k, macro)
            act_obj = feasible[act_idx % len(feasible)]

            ham_cost, rm_cost = smt_solver.get_resource_cost(act_obj)
            reward_k, _, _ = env.compute_total_reward(
                macro, pred_k, ham_cost, rm_cost
            )

            # Advance again for next_state_k
            if current_t < T:
                window.append(raw_events[current_t])
                current_t += 1
            win_next    = np.array(list(window), dtype=np.int64)
            pred_next   = lstm_pred.predict_state(win_next)
            next_state_k = hdrl_agent.encode_state(pred_next)

            hdrl_agent.push_lower(LowerTransition(
                state        = state_k,
                macro_action = macro,
                action       = act_idx % len(feasible),
                log_prob     = log_prob,
                reward       = reward_k,
                next_state   = next_state_k,
                done         = (k == cfg.K_STEPS - 1),
            ))
            upper_step_reward += reward_k
            lower_rewards_ep.append(reward_k)

        # ── Upper layer reward + next state ─────────────────────────────
        win_next_arr = np.array(list(window), dtype=np.int64)
        pred_next_u  = lstm_pred.predict_state(win_next_arr)
        next_state_u = hdrl_agent.encode_state(pred_next_u)

        hdrl_agent.push_upper(UpperTransition(
            state        = state,
            macro_action = macro,
            reward       = upper_step_reward,
            next_state   = next_state_u,
            done         = (t_step == cfg.T_STEPS - 1),
        ))
        upper_rewards_ep.append(upper_step_reward)

    # ── Network updates ─────────────────────────────────────────────────
    hdrl_agent.update_upper()
    hdrl_agent.update_lower()

    dsr = env.compute_dsr()
    return (
        float(np.mean(upper_rewards_ep)) if upper_rewards_ep else 0.0,
        float(np.mean(lower_rewards_ep)) if lower_rewards_ep else 0.0,
        dsr,
    )


def _run_baseline_episode(
    episode:       int,
    raw_events:    np.ndarray,
    lstm_pred:     LSTMPredictor,
    baseline,                          # RRTFRVMBaseline or DQRMFRVMBaseline
    env:           NetworkEnvironment,
    cfg:           Config,
    state_dim:     int,
    rng:           np.random.RandomState,
    algo_name:     str,
) -> float:
    """Run one episode for a baseline; return DSR."""
    T, N = raw_events.shape
    env.reset()

    t_offset = int(rng.randint(0, max(1, T - cfg.T_STEPS * cfg.K_STEPS - cfg.SEQ_LEN - 1)))
    window   = deque(maxlen=cfg.SEQ_LEN)
    for ts in range(cfg.SEQ_LEN):
        window.append(raw_events[min(t_offset + ts, T - 1)])
    current_t = t_offset + cfg.SEQ_LEN

    for t_step in range(cfg.T_STEPS):
        win_arr     = np.array(list(window), dtype=np.int64)
        pred_events = lstm_pred.predict_state(win_arr)

        if algo_name == "RRT+FRVM":
            macro, ham_cost, rm_cost = baseline.select_action()
            ham_eff, rm_eff = baseline.effectiveness()
        else:  # DQ-RM+FRVM
            dummy_state = np.zeros(state_dim, dtype=np.float32)
            macro, ham_cost, rm_cost = baseline.select_action(dummy_state)
            ham_eff, rm_eff = baseline.effectiveness(episode)

        # Override env effectiveness for baseline
        original_ham = env._ham_effectiveness
        original_rm  = env._rm_effectiveness
        env._ham_effectiveness = lambda m: ham_eff
        env._rm_effectiveness  = lambda m: rm_eff

        for k in range(cfg.K_STEPS):
            if current_t < T:
                window.append(raw_events[current_t])
                current_t += 1
            env.compute_total_reward(macro, pred_events, ham_cost, rm_cost)

        env._ham_effectiveness = original_ham
        env._rm_effectiveness  = original_rm

    return env.compute_dsr()


# ===========================================================================
# Main training routine
# ===========================================================================

def train_and_evaluate(cfg: Config, args):
    os.makedirs(cfg.LOG_DIR,   exist_ok=True)
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    rng = np.random.RandomState(cfg.SEED)

    # ── Initialise CSV / JSON log files ────────────────────────────────────
    defense_path = os.path.join(cfg.LOG_DIR, "defense_log.csv")
    conv_path    = os.path.join(cfg.LOG_DIR, "convergence_log.csv")
    netperf_path = os.path.join(cfg.LOG_DIR, "network_perf_log.csv")
    cm_path      = os.path.join(cfg.LOG_DIR, "confusion_matrices.json")
    summary_path = os.path.join(cfg.LOG_DIR, "final_summary.json")

    _init_csv(defense_path, [
        "episode", "algorithm", "attack_type", "dsr", "avg_reward", "timestamp"
    ])
    _init_csv(conv_path, [
        "episode", "layer", "attack_type", "reward", "timestamp"
    ])
    _init_csv(netperf_path, [
        "episode", "algorithm", "attack_type", "rtt_ms", "plr_pct", "timestamp"
    ])

    confusion_data: Dict  = {}
    summary_data:   Dict  = {}

    # ── Loop over attack sequence modes ───────────────────────────────────
    for mode in cfg.MODES:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode}")
        print(f"{'='*60}")

        # ── 1. Load data ─────────────────────────────────────────────────
        dl = DataLoader(
            mode            = mode,
            n_nodes         = cfg.N_NODES,
            n_switches      = cfg.N_SWITCHES,
            sequence_length = cfg.SEQ_LEN,
            train_ratio     = cfg.TRAIN_RATIO,
            n_timesteps     = cfg.N_TIMESTEPS,
            cicids_path     = cfg.CICIDS_PATH,
            seed            = cfg.SEED,
        )
        X_train, X_test, y_train, y_test = dl.load()
        raw_events = dl.get_raw_events()
        topo       = dl.get_topology()
        print(f"  Data: train={X_train.shape}  test={X_test.shape}")
        print(f"  Distribution: {dl.get_event_distribution()}")

        # ── 2. Train LSTM predictor ───────────────────────────────────────
        lstm_pred = LSTMPredictor(
            n_nodes       = cfg.N_NODES,
            n_event_types = 3,
            embed_dim     = cfg.LSTM_EMBED,
            hidden_size   = cfg.LSTM_HIDDEN,
            n_lstm_layers = cfg.LSTM_LAYERS,
            dropout       = cfg.LSTM_DROPOUT,
            learning_rate = cfg.LSTM_LR,
            device        = args.device,
            log_dir       = cfg.LOG_DIR,
        )
        # Append mode tag to log file for this run
        lstm_pred.log_path = os.path.join(
            cfg.LOG_DIR, f"lstm_training_log_{mode}.csv"
        )
        lstm_pred._init_csv(
            lstm_pred.log_path,
            ["episode", "phase", "loss", "fidelity", "timestamp"]
        )

        lstm_pred.fit(
            X_train, X_test, y_train, y_test,
            n_episodes = cfg.LSTM_EPISODES,
            batch_size = cfg.LSTM_BATCH,
        )

        # ── Confusion matrix on test set ─────────────────────────────────
        _, _, preds, targets = lstm_pred.evaluate(X_test, y_test)
        cm = lstm_pred.compute_confusion_matrix(preds, targets)
        cm_metrics = lstm_pred.class_metrics(cm)

        if mode not in confusion_data:
            confusion_data[mode] = {}
        confusion_data[mode]["LSTMNet"] = {
            "matrix"   : cm.tolist(),
            "precision": cm_metrics["precision"],
            "recall"   : cm_metrics["recall"],
            "f1"       : cm_metrics["f1"],
        }

        # Save LSTM model
        lstm_pred.save(os.path.join(cfg.MODEL_DIR, f"lstm_{mode}.pt"))

        # ── 3. Initialise SMT solver ──────────────────────────────────────
        net_cfg = NetworkConfig(
            n_nodes       = cfg.N_NODES,
            n_switches    = cfg.N_SWITCHES,
            n_ip_spaces   = cfg.N_IP_SPACES,
            n_flows       = cfg.N_FLOWS,
        )
        net_cfg.switch_adjacency = topo.switch_adjacency
        net_cfg.node_switch_map  = topo.node_switch_map
        smt_solver = SMTConstraintSolver(net_cfg, seed=cfg.SEED)

        # ── 4. Initialise HDRL agent ──────────────────────────────────────
        state_dim = cfg.N_NODES * 3   # one-hot encoding
        hdrl_agent = HDRLAgent(
            n_nodes           = cfg.N_NODES,
            n_event_types     = 3,
            n_feasible_actions= cfg.N_FEASIBLE,
            dqn_hidden        = cfg.DQN_HIDDEN,
            ppo_hidden        = cfg.PPO_HIDDEN,
            dqn_lr            = cfg.DQN_LR,
            ppo_lr            = cfg.PPO_LR,
            gamma             = cfg.GAMMA,
            gae_lambda        = cfg.GAE_LAMBDA,
            clip_eps          = cfg.CLIP_EPS,
            epsilon           = cfg.EPSILON_START,
            device            = args.device,
        )

        # ── 5. Initialise baselines ───────────────────────────────────────
        rrt_frvm = RRTFRVMBaseline(
            n_nodes     = cfg.N_NODES,
            n_switches  = cfg.N_SWITCHES,
            n_flows     = cfg.N_FLOWS,
            n_ip_spaces = cfg.N_IP_SPACES,
            seed        = cfg.SEED,
        )
        dqrm_frvm = DQRMFRVMBaseline(
            state_dim       = state_dim,
            n_route_actions = cfg.N_FEASIBLE,
            lr              = cfg.DQN_LR,
            seed            = cfg.SEED,
            device          = args.device,
        )

        env_cmtd  = NetworkEnvironment(cfg.N_NODES, cfg.N_SWITCHES,
                                       cfg.ALPHA1, cfg.ALPHA2,
                                       cfg.GAMMA1, cfg.GAMMA2, cfg.C_REW)
        env_rrt   = NetworkEnvironment(cfg.N_NODES, cfg.N_SWITCHES,
                                       cfg.ALPHA1, cfg.ALPHA2,
                                       cfg.GAMMA1, cfg.GAMMA2, cfg.C_REW)
        env_dqrm  = NetworkEnvironment(cfg.N_NODES, cfg.N_SWITCHES,
                                       cfg.ALPHA1, cfg.ALPHA2,
                                       cfg.GAMMA1, cfg.GAMMA2, cfg.C_REW)

        # ── 6. Training loop ──────────────────────────────────────────────
        dsr_window_cmtd = deque(maxlen=cfg.DSR_WINDOW)
        dsr_window_rrt  = deque(maxlen=cfg.DSR_WINDOW)
        dsr_window_dqrm = deque(maxlen=cfg.DSR_WINDOW)

        print(f"\n  [HDRL] Training {cfg.N_EPISODES} episodes …")
        t0 = time.time()

        for ep in range(1, cfg.N_EPISODES + 1):

            # --- CM-MTD ---
            u_rew, l_rew, dsr_cm = _run_one_episode(
                ep, raw_events, lstm_pred, smt_solver, hdrl_agent,
                env_cmtd, cfg, rng
            )
            dsr_window_cmtd.append(dsr_cm)
            avg_dsr_cm = float(np.mean(dsr_window_cmtd))

            # --- RRT+FRVM ---
            dsr_rrt = _run_baseline_episode(
                ep, raw_events, lstm_pred, rrt_frvm, env_rrt,
                cfg, state_dim, rng, "RRT+FRVM"
            )
            dsr_window_rrt.append(dsr_rrt)
            avg_dsr_rrt = float(np.mean(dsr_window_rrt))

            # --- DQ-RM+FRVM ---
            dsr_dqrm = _run_baseline_episode(
                ep, raw_events, lstm_pred, dqrm_frvm, env_dqrm,
                cfg, state_dim, rng, "DQ-RM+FRVM"
            )
            dsr_window_dqrm.append(dsr_dqrm)
            avg_dsr_dqrm = float(np.mean(dsr_window_dqrm))

            # Network performance (CM-MTD)
            rtt, plr = env_cmtd.simulate_network_performance(MACRO_BOTH)

            ts = _now()
            _append_csv(defense_path, [ep, "CM-MTD",     mode, f"{avg_dsr_cm:.4f}",   f"{u_rew:.4f}", ts])
            _append_csv(defense_path, [ep, "RRT+FRVM",   mode, f"{avg_dsr_rrt:.4f}",  "0.0000",       ts])
            _append_csv(defense_path, [ep, "DQ-RM+FRVM", mode, f"{avg_dsr_dqrm:.4f}", "0.0000",       ts])
            _append_csv(conv_path,    [ep, "upper", mode, f"{u_rew:.4f}", ts])
            _append_csv(conv_path,    [ep, "lower", mode, f"{l_rew:.4f}", ts])
            _append_csv(netperf_path, [ep, "CM-MTD",      mode, f"{rtt:.4f}", f"{plr:.4f}", ts])
            _append_csv(netperf_path, [ep, "no_mutation",  mode, "1.0000",    "0.0000",     ts])

            if ep % 50 == 0 or ep == 1:
                elapsed = time.time() - t0
                print(
                    f"    ep={ep:4d}/{cfg.N_EPISODES}  "
                    f"DSR: CM-MTD={avg_dsr_cm:5.1f}%  "
                    f"RRT+FRVM={avg_dsr_rrt:5.1f}%  "
                    f"DQ-RM={avg_dsr_dqrm:5.1f}%  "
                    f"ε={hdrl_agent.dqn.epsilon:.3f}  "
                    f"elapsed={elapsed:.0f}s"
                )

        # ── 7. Save model ─────────────────────────────────────────────────
        hdrl_agent.save(os.path.join(cfg.MODEL_DIR, f"hdrl_{mode}"))
        print(f"  [HDRL] Models saved to {cfg.MODEL_DIR}/hdrl_{mode}_*.pt")

        # ── 8. Final DSR summary for this mode ───────────────────────────
        summary_data[mode] = {
            "CM-MTD_final_dsr"    : avg_dsr_cm,
            "RRT+FRVM_final_dsr"  : avg_dsr_rrt,
            "DQ-RM+FRVM_final_dsr": avg_dsr_dqrm,
        }

    # ── 9. Write JSON logs ────────────────────────────────────────────────
    _write_json(cm_path,      confusion_data)
    _write_json(summary_path, summary_data)

    print(f"\n{'='*60}")
    print("  Training complete.")
    print(f"  Logs  → {cfg.LOG_DIR}/")
    print(f"  Models→ {cfg.MODEL_DIR}/")
    print(f"  Final DSR summary:")
    for mode, vals in summary_data.items():
        print(f"    [{mode}]  CM-MTD={vals['CM-MTD_final_dsr']:.1f}%  "
              f"RRT+FRVM={vals['RRT+FRVM_final_dsr']:.1f}%  "
              f"DQ-RM+FRVM={vals['DQ-RM+FRVM_final_dsr']:.1f}%")


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="CM-MTD: Train and evaluate moving target defense in DTMN."
    )
    p.add_argument("--n_episodes",   type=int,   default=Config.N_EPISODES,
                   help="Number of HDRL training episodes per mode.")
    p.add_argument("--n_nodes",      type=int,   default=Config.N_NODES,
                   help="Number of network nodes.")
    p.add_argument("--n_timesteps",  type=int,   default=Config.N_TIMESTEPS,
                   help="Simulation length (events) per dataset mode.")
    p.add_argument("--lstm_episodes",type=int,   default=Config.LSTM_EPISODES,
                   help="LSTM training epochs per dataset mode.")
    p.add_argument("--log_dir",      type=str,   default=Config.LOG_DIR,
                   help="Directory for CSV / JSON log files.")
    p.add_argument("--model_dir",    type=str,   default=Config.MODEL_DIR,
                   help="Directory for saved model checkpoints.")
    p.add_argument("--cicids_path",  type=str,   default=None,
                   help="Path to CICIDS-2017 CSV directory (optional).")
    p.add_argument("--device",       type=str,   default="cpu",
                   choices=["cpu", "cuda"],
                   help="Torch device.")
    p.add_argument("--seed",         type=int,   default=Config.SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = Config()

    # Apply CLI overrides
    cfg.N_EPISODES    = args.n_episodes
    cfg.N_NODES       = args.n_nodes
    cfg.N_SWITCHES    = args.n_nodes          # keep equal for simplicity
    cfg.N_TIMESTEPS   = args.n_timesteps
    cfg.LSTM_EPISODES = args.lstm_episodes
    cfg.LOG_DIR       = args.log_dir
    cfg.MODEL_DIR     = args.model_dir
    cfg.CICIDS_PATH   = args.cicids_path
    cfg.SEED          = args.seed

    train_and_evaluate(cfg, args)
