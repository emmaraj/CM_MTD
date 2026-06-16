"""
train_baselines.py — Pre-train DQN-RM+FRVM baseline before evaluation.

The DQ-RM+FRVM baseline [32] learns online with a DQN, so it must be
trained for a number of episodes before a fair comparison against CM-MTD.
RRT+FRVM, STATIC, HAM-only, and RM-only are policy-free and need no training.

Usage:
    python train_baselines.py [--config config/config.yaml]
                              [--n-episodes 5000] [--seed 42]
                              [--synthetic] [--all-seeds]
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from utils.logger import setup_logger
from utils.seed_utils import set_global_seed, get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-train DQN-RM+FRVM baseline")
    p.add_argument("--config",     default="config/config.yaml")
    p.add_argument("--n-episodes", type=int, default=5000)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--all-seeds",  action="store_true")
    p.add_argument("--synthetic",  action="store_true")
    p.add_argument("--out-dir",    default="results")
    p.add_argument("--device",     default=None)
    return p.parse_args()


def train_dqn_rm_frvm(
    seed: int,
    config: Dict,
    n_episodes: int,
    attack_seq: np.ndarray,
    device: str,
    out_dir: str,
) -> Dict:
    """Train the DQN-RM+FRVM baseline for n_episodes and save checkpoint."""
    from utils.compat import TORCH_AVAILABLE
    logger = logging.getLogger("cm_mtd.train_baselines")

    if not TORCH_AVAILABLE:
        logger.warning(
            "PyTorch not available — DQN-RM+FRVM cannot be trained.\n"
            "Install with: pip install torch\n"
            "Saving policy-free baselines only."
        )
        return {"status": "skipped", "reason": "torch_unavailable"}

    set_global_seed(seed)

    from environment.dtmn_env import DTMNEnvironment
    from agents.baselines import DQNRMFRVMBaseline

    env = DTMNEnvironment(
        network_config=config["network"],
        reward_config=config["reward"],
        attack_sequence=attack_seq,
        sequence_length=config["data"].get("sequence_length", 10),
        n_steps_per_episode=config["training"].get("n_steps_per_episode", 25),
        K_steps=config["training"].get("K_steps", 5),
        seed=seed,
    )
    obs_dim = env.get_obs_dim()

    baseline = DQNRMFRVMBaseline(
        state_dim=obs_dim,
        learning_rate=config["dqn"].get("learning_rate", 1e-3),
        gamma=config["dqn"].get("gamma", 0.99),
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=config["dqn"].get("buffer_size", 50_000),
        batch_size=config["dqn"].get("batch_size", 64),
        device=device,
        seed=seed,
    )

    dsr_history: List[float] = []
    reward_history: List[float] = []
    log_every = max(1, n_episodes // 20)

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        ep_reward = 0.0
        done = False
        prev = state.copy()

        while not done:
            action = baseline.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            baseline.store_transition(prev, action, reward, next_state, done)
            baseline.update()
            ep_reward += reward
            prev = state
            state = next_state

        baseline.decay_epsilon()
        dsr_history.append(env.get_episode_dsr())
        reward_history.append(ep_reward)

        if episode % log_every == 0:
            mean_dsr = float(np.mean(dsr_history[-100:]))
            logger.info(
                f"[DQN-RM+FRVM seed={seed}] Ep {episode:4d}/{n_episodes} | "
                f"DSR={mean_dsr:.1f}% | ε={baseline.epsilon:.3f}"
            )

    # Save checkpoint
    ckpt_dir = Path(out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if TORCH_AVAILABLE:
        import torch
        torch.save({
            "q_net":     baseline.q_net.state_dict(),
            "optimizer": baseline.optimizer.state_dict(),
            "epsilon":   baseline.epsilon,
            "n_episodes": n_episodes,
            "dsr_history": dsr_history,
        }, ckpt_dir / f"dqn_rm_frvm_seed{seed}.pt")

    metrics_dir = Path(out_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"dqn_rm_frvm_seed{seed}.json", "w") as f:
        json.dump({
            "seed": seed,
            "n_episodes": n_episodes,
            "final_dsr_mean": float(np.mean(dsr_history[-100:])),
            "final_dsr_std": float(np.std(dsr_history[-100:], ddof=1)),
            "dsr_history": dsr_history,
        }, f, indent=2)

    logger.info(
        f"DQN-RM+FRVM seed={seed} trained | "
        f"Final DSR: {np.mean(dsr_history[-100:]):.2f}%"
    )
    return {"dsr_history": dsr_history, "seed": seed}


def run(args: argparse.Namespace) -> None:
    logger = setup_logger("cm_mtd.train_baselines", log_dir="logs")
    config = load_config(args.config)
    device = args.device or get_device()
    config["experiment"]["device"] = device

    seeds = config["experiment"]["seeds"] if args.all_seeds else [args.seed]
    logger.info(f"Training DQN-RM+FRVM baseline | seeds={seeds} | n_episodes={args.n_episodes}")

    # Build attack sequence
    from datasets.cicids2017_loader import N_CLASSES
    if args.synthetic:
        from datasets.sequence_builder import SyntheticDataGenerator
        gen = SyntheticDataGenerator(n_classes=N_CLASSES, seed=42)
        _, attack_seq = gen.generate_event_sequence(
            n_nodes=config["network"]["n_nodes"],
            n_steps=100_000,
            sequence_length=config["data"].get("sequence_length", 10),
        )
    else:
        try:
            from datasets.cicids2017_loader import CICIDS2017Loader
            from datasets.preprocessor import CICIDS2017Preprocessor
            loader = CICIDS2017Loader(data_dir=config["data"]["cicids2017_path"])
            df = loader.load_all(verbose=False)
            pre = CICIDS2017Preprocessor(config=config)
            _, attack_seq, _ = pre.fit_transform(df)
        except Exception as e:
            logger.warning(f"CICIDS2017 unavailable ({e}) — using synthetic.")
            from datasets.sequence_builder import SyntheticDataGenerator
            gen = SyntheticDataGenerator(n_classes=N_CLASSES, seed=42)
            _, attack_seq = gen.generate_event_sequence(
                n_nodes=config["network"]["n_nodes"], n_steps=100_000,
                sequence_length=config["data"].get("sequence_length", 10),
            )

    all_results = []
    for seed in seeds:
        result = train_dqn_rm_frvm(
            seed=seed, config=config,
            n_episodes=args.n_episodes,
            attack_seq=attack_seq,
            device=device, out_dir=args.out_dir,
        )
        all_results.append(result)

    # Summary
    trained = [r for r in all_results if r.get("status") != "skipped"]
    if trained:
        dsrs = [float(np.mean(r["dsr_history"][-100:])) for r in trained]
        logger.info(
            f"\nDQN-RM+FRVM Training Complete | "
            f"Mean DSR across seeds: {np.mean(dsrs):.2f}% ± {np.std(dsrs, ddof=1):.2f}%"
        )
    logger.info("Policy-free baselines (STATIC, HAM, RM, RRT+FRVM) require no training.")


if __name__ == "__main__":
    run(parse_args())
