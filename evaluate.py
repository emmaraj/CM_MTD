"""
evaluate.py — Full evaluation pipeline for CM-MTD vs all baselines.

Usage:
    python evaluate.py [--config config/config.yaml] [--all-seeds]
                       [--dqn-ckpt checkpoints/dqn_final.pt]
                       [--ppo-ckpt checkpoints/ppo_final.pt]
                       [--lstm-ckpt checkpoints/lstm_final.keras]
                       [--n-eval 1000] [--synthetic]

Runs:
  1. Evaluate CM-MTD (proposed) across n_eval episodes × 5 seeds
  2. Evaluate all baselines (STATIC, HAM, RM, RRT+FRVM, DQN-RM+FRVM)
  3. Statistical comparison: paired t-test, Wilcoxon, Cohen's d
  4. Save all metrics to results/metrics/evaluation_report.json
  5. Print publication-ready summary table
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
from utils.seed_utils import set_global_seed, get_device
from utils.logger import setup_logger
from utils.statistical_analysis import summarize_all_baselines
from utils.metrics import compute_confidence_interval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CM-MTD vs baselines")
    p.add_argument("--config",    default="config/config.yaml")
    p.add_argument("--dqn-ckpt", default="checkpoints/dqn_final.pt")
    p.add_argument("--ppo-ckpt", default="checkpoints/ppo_final.pt")
    p.add_argument("--lstm-ckpt",default="checkpoints/lstm_final.keras")
    p.add_argument("--n-eval",   type=int, default=1000)
    p.add_argument("--all-seeds",action="store_true")
    p.add_argument("--synthetic",action="store_true")
    p.add_argument("--out-dir",  default="results")
    p.add_argument("--device",   default=None)
    return p.parse_args()


def make_env(config: Dict, attack_seq: np.ndarray, lstm=None, seed: int = 42):
    """Build a fresh DTMN environment."""
    from environment.dtmn_env import DTMNEnvironment
    return DTMNEnvironment(
        network_config=config["network"],
        reward_config=config["reward"],
        lstm_predictor=lstm,
        attack_sequence=attack_seq,
        sequence_length=config["data"].get("sequence_length", 10),
        n_steps_per_episode=config["training"].get("n_steps_per_episode", 25),
        K_steps=config["training"].get("K_steps", 5),
        seed=seed,
    )


def evaluate_method(
    agent,
    env,
    n_episodes: int,
    seeds: List[int],
    train_dqn_rm: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Evaluate an agent across multiple seeds.

    Returns:
        Dict with 'dsr', 'rtt', 'plr' arrays of shape [n_seeds, n_episodes].
    """
    from agents.baselines import run_baseline_episode, BaselineAgent
    from agents.hdrl_agent import HDRLAgent

    all_dsr, all_rtt, all_plr, all_reward = [], [], [], []

    for seed in seeds:
        set_global_seed(seed)
        seed_dsr, seed_rtt, seed_plr, seed_rwd = [], [], [], []

        for ep in range(n_episodes):
            ep_seed = seed * 100_000 + ep

            if isinstance(agent, HDRLAgent):
                state, _ = env.reset(seed=ep_seed)
                done = False
                ep_reward = 0.0
                while not done:
                    macro = agent.dqn.select_action(state, deterministic=True)
                    state, r, done, _, _ = env.step(macro)
                    ep_reward += r
                seed_dsr.append(env.get_episode_dsr())
                seed_rtt.append(float(np.mean(env._ep_rtt)) if env._ep_rtt else 0.0)
                seed_plr.append(float(np.mean(env._ep_plr)) if env._ep_plr else 0.0)
                seed_rwd.append(ep_reward)
            else:
                metrics = run_baseline_episode(
                    baseline=agent,
                    env=env,
                    seed=ep_seed,
                    train_dqn_rm=train_dqn_rm,
                )
                seed_dsr.append(metrics["dsr"])
                seed_rtt.append(metrics["rtt"])
                seed_plr.append(metrics["plr"])
                seed_rwd.append(metrics["reward"])

        all_dsr.append(seed_dsr)
        all_rtt.append(seed_rtt)
        all_plr.append(seed_plr)
        all_reward.append(seed_rwd)

    return {
        "dsr":    np.array(all_dsr),
        "rtt":    np.array(all_rtt),
        "plr":    np.array(all_plr),
        "reward": np.array(all_reward),
    }


def run(args: argparse.Namespace) -> None:
    logger = setup_logger("cm_mtd_eval", log_dir="logs")
    config = load_config(args.config)
    device = args.device or get_device()
    config["experiment"]["device"] = device

    seeds = config["experiment"]["seeds"] if args.all_seeds else [42]
    n_eval = args.n_eval

    logger.info("=" * 60)
    logger.info(f"CM-MTD Evaluation | n_eval={n_eval} | seeds={seeds}")
    logger.info("=" * 60)

    # ── Build attack sequence ─────────────────────────────────────────────────
    from datasets.cicids2017_loader import N_CLASSES
    if args.synthetic:
        from datasets.sequence_builder import SyntheticDataGenerator
        gen = SyntheticDataGenerator(n_classes=N_CLASSES, seed=42)
        _, attack_seq = gen.generate_event_sequence(
            n_nodes=config["network"]["n_nodes"],
            n_steps=200_000,
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
        except Exception:
            logger.warning("Falling back to synthetic attack sequence.")
            args.synthetic = True
            from datasets.sequence_builder import SyntheticDataGenerator
            gen = SyntheticDataGenerator(n_classes=N_CLASSES, seed=42)
            _, attack_seq = gen.generate_event_sequence(
                n_nodes=config["network"]["n_nodes"], n_steps=200_000,
                sequence_length=config["data"].get("sequence_length", 10),
            )

    # ── LSTM predictor ────────────────────────────────────────────────────────
    lstm = None
    if Path(args.lstm_ckpt).exists():
        from models.lstm_predictor import LSTMAttackPredictor
        lstm = LSTMAttackPredictor.from_config(config)
        try:
            lstm.load(args.lstm_ckpt)
        except Exception:
            lstm = None

    env = make_env(config, attack_seq, lstm=lstm, seed=42)
    obs_dim = env.get_obs_dim()

    # ── Build CM-MTD agent ────────────────────────────────────────────────────
    from agents.upper_layer_dqn import UpperLayerDQN
    from agents.lower_layer_ppo import LowerLayerPPO
    from agents.hdrl_agent import HDRLAgent

    dqn = UpperLayerDQN.from_config(config, state_dim=obs_dim)
    ppo = LowerLayerPPO.from_config(config, state_dim=obs_dim)
    cm_mtd = HDRLAgent.from_config(config, env, dqn, ppo)

    # Load checkpoints if they exist
    if Path(args.dqn_ckpt).exists() and Path(args.ppo_ckpt).exists():
        cm_mtd.load_checkpoint(args.dqn_ckpt, args.ppo_ckpt)
        logger.info("CM-MTD checkpoints loaded.")
    else:
        logger.warning(
            "No CM-MTD checkpoint found. Running with untrained agent "
            "(results will not match paper). Train first with: python train_hdrl.py"
        )

    # ── Build baselines ───────────────────────────────────────────────────────
    from agents.baselines import build_all_baselines
    baselines = build_all_baselines(
        state_dim=obs_dim, config=config, device=device, seed=42
    )

    # ── Evaluate all methods ──────────────────────────────────────────────────
    all_results: Dict[str, Dict] = {}

    # CM-MTD
    logger.info("\nEvaluating CM-MTD (proposed)...")
    all_results["CM_MTD"] = evaluate_method(cm_mtd, env, n_eval, seeds)
    _log_result(logger, "CM_MTD", all_results["CM_MTD"])

    # Baselines
    for name, baseline in baselines.items():
        logger.info(f"\nEvaluating {name}...")
        train_dqn = (name == "DQN_RM_FRVM")
        all_results[name] = evaluate_method(baseline, env, n_eval, seeds, train_dqn_rm=train_dqn)
        _log_result(logger, name, all_results[name])

    # ── Statistical comparisons ───────────────────────────────────────────────
    logger.info("\n── Statistical Analysis ──────────────────────────────────")
    dsr_results = {k: v["dsr"] for k, v in all_results.items()}
    stats = summarize_all_baselines(dsr_results, proposed_key="CM_MTD")

    for baseline_name, comparison in stats.items():
        tt  = comparison["paired_ttest"]
        wil = comparison["wilcoxon"]
        d   = comparison["cohen_d"]
        imp = comparison["improvement_pct"]
        logger.info(
            f"  CM-MTD vs {baseline_name:14s} | "
            f"Improvement: +{imp:.1f}% | "
            f"t-test p={tt['p_value']:.4f} ({'*' if tt['significant'] else 'ns'}) | "
            f"Wilcoxon p={wil['p_value']:.4f} ({'*' if wil['significant'] else 'ns'}) | "
            f"Cohen's d={d:.3f} ({comparison['effect_size_label']})"
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    logger.info("\n── DSR Summary Table ─────────────────────────────────────")
    logger.info(f"{'Method':<18} {'Mean DSR':>10} {'Std':>7} {'95% CI':>20}")
    logger.info("-" * 60)
    for name, res in all_results.items():
        dsr_arr = res["dsr"].flatten()
        mean, lo, hi = compute_confidence_interval(dsr_arr)
        std = float(np.std(dsr_arr, ddof=1))
        marker = " ◀ proposed" if name == "CM_MTD" else ""
        logger.info(f"  {name:<16} {mean:>8.2f}%  {std:>5.2f}%  [{lo:>6.2f}, {hi:>6.2f}]{marker}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "config":  {"n_eval": n_eval, "seeds": seeds},
        "results": {
            name: {
                "dsr_mean": float(np.mean(v["dsr"])),
                "dsr_std":  float(np.std(v["dsr"], ddof=1)),
                "rtt_mean": float(np.mean(v["rtt"])),
                "plr_mean": float(np.mean(v["plr"])),
            }
            for name, v in all_results.items()
        },
        "statistical_tests": {
            k: {
                "improvement_pct": v["improvement_pct"],
                "ttest_p":         v["paired_ttest"]["p_value"],
                "wilcoxon_p":      v["wilcoxon"]["p_value"],
                "cohen_d":         v["cohen_d"],
                "effect_size":     v["effect_size_label"],
            }
            for k, v in stats.items()
        },
    }
    report_path = out_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nFull evaluation report saved to {report_path}")

    # Save raw DSR arrays for figure generation
    dsr_path = out_dir / "dsr_arrays.npz"
    np.savez(dsr_path, **{k: v["dsr"] for k, v in all_results.items()})
    logger.info(f"DSR arrays saved to {dsr_path}")

    logger.info("\nEvaluation complete.")
    return all_results, stats


def _log_result(logger: logging.Logger, name: str, res: Dict) -> None:
    dsr = res["dsr"].flatten()
    logger.info(
        f"  {name:<16}: DSR={np.mean(dsr):.2f}%±{np.std(dsr, ddof=1):.2f}% | "
        f"RTT={np.mean(res['rtt']):.2f}ms | PLR={np.mean(res['plr']):.2f}%"
    )


if __name__ == "__main__":
    run(parse_args())
