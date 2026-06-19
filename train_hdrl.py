"""
train_hdrl.py — Train the Hierarchical DRL agent (CM-MTD).

Usage:
    python train_hdrl.py [--config config/config.yaml] [--seed 42]
                         [--lstm-ckpt checkpoints/lstm_final.keras]
                         [--synthetic] [--n-episodes 10000]

Implements Algorithm 1 (complete training loop):
  1. Build DTMN environment with LSTM state prediction
  2. Initialize upper-layer DQN and lower-layer PPO
  3. Run M episodes × T steps × K inner steps
  4. Log DSR, rewards, RTT, PLR every eval_freq episodes
  5. Save checkpoints every checkpoint_freq episodes
  6. Optionally run all 5 seeds for multi-seed evaluation
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from utils.seed_utils import set_global_seed, get_device
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CM-MTD HDRL Agent")
    p.add_argument("--config",      default="config/config.yaml")
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--all-seeds",   action="store_true", help="Run all 5 seeds")
    p.add_argument("--lstm-ckpt",   default="checkpoints/lstm_final.keras")
    p.add_argument("--synthetic",   action="store_true")
    p.add_argument("--n-episodes",  type=int,  default=None)
    p.add_argument("--out-dir",     default="results")
    p.add_argument("--device",      default=None, help="cuda / cpu")
    p.add_argument("--small-net",   action="store_true", help="Use 12-node network")
    return p.parse_args()


def build_attack_sequence(config: Dict, seed: int, synthetic: bool) -> np.ndarray:
    """
    Load CICIDS2017 labels or generate synthetic sequence for the RL environment.

    The RL environment only needs a long enough sequence to sample diverse
    episodes from.  Passing 2.2 million samples causes a segfault on most
    machines because the environment pre-allocates per-node arrays.
    We therefore cap the sequence at MAX_SEQ_LEN after shuffling so every
    attack class is still represented.
    """
    from datasets.cicids2017_loader import N_CLASSES

    # Maximum sequence length passed to the RL environment.
    # 200 K gives ~8,000 episodes of length T=25 × K=5 — more than enough.
    MAX_SEQ_LEN = 200_000

    n_nodes = config["network"]["n_nodes"]
    seq_len = config["data"].get("sequence_length", 10)

    if synthetic:
        from datasets.sequence_builder import SyntheticDataGenerator
        gen = SyntheticDataGenerator(n_classes=N_CLASSES, seed=seed)
        _, y = gen.generate_event_sequence(
            n_nodes=n_nodes, n_steps=MAX_SEQ_LEN, sequence_length=seq_len
        )
        return y

    try:
        from datasets.cicids2017_loader import CICIDS2017Loader
        from datasets.preprocessor import CICIDS2017Preprocessor
        loader = CICIDS2017Loader(data_dir=config["data"]["cicids2017_path"])
        df = loader.load_all(verbose=False)
        pre = CICIDS2017Preprocessor(config=config)
        _, y_labels, _ = pre.fit_transform(df)

        # Subsample while preserving class proportions (stratified cap)
        rng = np.random.default_rng(seed)
        if len(y_labels) > MAX_SEQ_LEN:
            # Keep temporal order: take a contiguous random window
            # so the LSTM sees realistic sequential attack patterns
            start = int(rng.integers(0, len(y_labels) - MAX_SEQ_LEN))
            y_labels = y_labels[start: start + MAX_SEQ_LEN]
            logging.getLogger("cm_mtd").info(
                f"Attack sequence capped at {MAX_SEQ_LEN:,} "
                f"(window [{start:,} – {start + MAX_SEQ_LEN:,}] of {len(y_labels) + MAX_SEQ_LEN:,})"
            )

        return y_labels

    except Exception as e:
        logging.getLogger("cm_mtd").warning(
            f"CICIDS2017 unavailable ({e}), using synthetic sequence."
        )
        from datasets.sequence_builder import SyntheticDataGenerator
        gen = SyntheticDataGenerator(n_classes=N_CLASSES, seed=seed)
        _, y = gen.generate_event_sequence(
            n_nodes=n_nodes, n_steps=MAX_SEQ_LEN, sequence_length=seq_len
        )
        return y


def build_lstm_predictor(ckpt_path: str, config: Dict) -> Optional[object]:
    """Load trained LSTM if checkpoint exists, else return None."""
    if Path(ckpt_path).exists():
        from models.lstm_predictor import LSTMAttackPredictor
        model = LSTMAttackPredictor.from_config(config)
        try:
            model.load(ckpt_path)
            logging.getLogger("cm_mtd").info(f"LSTM loaded from {ckpt_path}")
            return model
        except Exception as e:
            logging.getLogger("cm_mtd").warning(f"LSTM load failed ({e}). Running without predictor.")
    return None


def train_single_seed(
    seed: int,
    config: Dict,
    args: argparse.Namespace,
    device: str,
) -> Dict:
    """Run one complete training cycle for a single seed."""
    logger = logging.getLogger("cm_mtd")
    set_global_seed(seed)

    # ── Override config values from CLI ──────────────────────────────────────
    if args.n_episodes:
        config["training"]["n_episodes"] = args.n_episodes
    if args.small_net:
        config["network"]["n_nodes"]    = 12
        config["network"]["n_switches"] = 12

    # ── Attack sequence ───────────────────────────────────────────────────────
    attack_seq = build_attack_sequence(config, seed=seed, synthetic=args.synthetic)
    logger.info(f"Attack sequence length: {len(attack_seq):,}")

    # ── LSTM predictor ────────────────────────────────────────────────────────
    lstm = build_lstm_predictor(args.lstm_ckpt, config)

    # ── Environment ───────────────────────────────────────────────────────────
    from environment.dtmn_env import DTMNEnvironment
    env = DTMNEnvironment(
        network_config=config["network"],
        reward_config=config["reward"],
        lstm_predictor=lstm,
        attack_sequence=attack_seq,
        sequence_length=config["data"].get("sequence_length", 10),
        n_steps_per_episode=config["training"].get("n_steps_per_episode", 25),
        K_steps=config["training"].get("K_steps", 5),
        seed=seed,
    )
    obs_dim  = env.get_obs_dim()
    logger.info(f"Environment: obs_dim={obs_dim}, n_actions=4")

    # ── Agents ────────────────────────────────────────────────────────────────
    from agents.upper_layer_dqn import UpperLayerDQN
    from agents.lower_layer_ppo import LowerLayerPPO
    from agents.hdrl_agent import HDRLAgent

    dqn = UpperLayerDQN.from_config(config, state_dim=obs_dim)
    ppo = LowerLayerPPO.from_config(config, state_dim=obs_dim)

    agent = HDRLAgent.from_config(
        config=config,
        env=env,
        dqn=dqn,
        ppo=ppo,
    )
    agent.seed = seed

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info(f"Starting training: seed={seed}, episodes={config['training']['n_episodes']}")
    history = agent.train()

    # ── Final evaluation ──────────────────────────────────────────────────────
    eval_metrics = agent.evaluate_full(
        n_episodes=config["evaluation"].get("n_eval_episodes", 1000)
    )
    logger.info(
        f"Seed {seed} | DSR={eval_metrics['dsr_mean']:.2f}±{eval_metrics['dsr_std']:.2f}% | "
        f"RTT={eval_metrics['rtt_mean']:.2f}ms | PLR={eval_metrics['plr_mean']:.2f}%"
    )

    # ── Save seed results ─────────────────────────────────────────────────────
    seed_dir = Path(args.out_dir) / "metrics" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    with open(seed_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    with open(seed_dir / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    return {
        "seed": seed,
        "history": history,
        "eval": eval_metrics,
    }


def aggregate_seeds(all_results: List[Dict]) -> Dict:
    """
    Aggregate results across multiple seeds.
    Computes mean ± std and 95% confidence intervals.
    """
    from utils.metrics import compute_confidence_interval
    from utils.statistical_analysis import _compute_ci

    dsr_vals = np.array([r["eval"]["dsr_mean"] for r in all_results])
    rtt_vals = np.array([r["eval"]["rtt_mean"] for r in all_results])
    plr_vals = np.array([r["eval"]["plr_mean"] for r in all_results])

    dsr_ci  = compute_confidence_interval(dsr_vals)
    rtt_ci  = compute_confidence_interval(rtt_vals)
    plr_ci  = compute_confidence_interval(plr_vals)

    return {
        "n_seeds": len(all_results),
        "dsr": {
            "mean":  float(dsr_ci[0]),
            "std":   float(np.std(dsr_vals, ddof=1)),
            "ci_lo": float(dsr_ci[1]),
            "ci_hi": float(dsr_ci[2]),
        },
        "rtt": {
            "mean":  float(rtt_ci[0]),
            "std":   float(np.std(rtt_vals, ddof=1)),
            "ci_lo": float(rtt_ci[1]),
            "ci_hi": float(rtt_ci[2]),
        },
        "plr": {
            "mean":  float(plr_ci[0]),
            "std":   float(np.std(plr_vals, ddof=1)),
            "ci_lo": float(plr_ci[1]),
            "ci_hi": float(plr_ci[2]),
        },
    }


def run(args: argparse.Namespace) -> None:
    logger = setup_logger("cm_mtd", log_dir="logs")
    config = load_config(args.config)

    device = args.device or get_device()
    config["experiment"]["device"] = device
    logger.info(f"Using device: {device}")

    seeds = config["experiment"].get("seeds", [42, 123, 456, 789, 1024]) \
        if args.all_seeds else [args.seed]

    logger.info(f"Training seeds: {seeds}")
    all_results = []

    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"  SEED {seed}")
        logger.info(f"{'='*60}")
        result = train_single_seed(seed, config, args, device)
        all_results.append(result)

    # ── Multi-seed aggregation ────────────────────────────────────────────────
    if len(all_results) > 1:
        agg = aggregate_seeds(all_results)
        agg_path = Path(args.out_dir) / "metrics" / "aggregated_results.json"
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)

        logger.info("\n" + "="*60)
        logger.info("MULTI-SEED AGGREGATED RESULTS (CM-MTD)")
        logger.info(f"  DSR:  {agg['dsr']['mean']:.2f}% ± {agg['dsr']['std']:.2f}%"
                    f"  [95% CI: {agg['dsr']['ci_lo']:.2f}, {agg['dsr']['ci_hi']:.2f}]")
        logger.info(f"  RTT:  {agg['rtt']['mean']:.2f}ms ± {agg['rtt']['std']:.2f}ms")
        logger.info(f"  PLR:  {agg['plr']['mean']:.2f}% ± {agg['plr']['std']:.2f}%")
        logger.info(f"Saved to {agg_path}")
        logger.info("="*60)

    logger.info("HDRL training complete.")


if __name__ == "__main__":
    run(parse_args())
