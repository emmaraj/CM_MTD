"""
main.py — Complete CM-MTD Research Pipeline.

Running `python main.py` executes the full pipeline:
  1. Preprocess CICIDS2017 data (or generate synthetic)
  2. Train LSTM attack predictor
  3. Train HDRL agent (CM-MTD)
  4. Evaluate CM-MTD vs all baselines with statistical tests
  5. Generate all paper figures (Figs 7–11)
  6. Save all metrics and models

Usage:
    # Full pipeline with real data (place CSVs in data/cicids2017/ first):
    python main.py

    # Full pipeline with synthetic data (no downloads needed):
    python main.py --synthetic

    # Only specific stages:
    python main.py --stage lstm          # train LSTM only
    python main.py --stage hdrl          # train HDRL only (needs LSTM)
    python main.py --stage evaluate      # evaluate only (needs checkpoints)
    python main.py --stage figures       # figures only

    # Research-grade: 5 seeds, large network:
    python main.py --synthetic --all-seeds --n-episodes 10000

    # Quick smoke test:
    python main.py --synthetic --n-episodes 200 --n-eval 50
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CM-MTD: Collaborative Mutation-based Moving Target Defense",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Pipeline control
    p.add_argument(
        "--stage",
        choices=["all", "preprocess", "lstm", "hdrl", "evaluate", "figures"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    # Data
    p.add_argument("--config",    default="config/config.yaml")
    p.add_argument("--data-dir",  default="data/cicids2017",
                   help="Directory with CICIDS2017 CSV files")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic data instead of CICIDS2017")
    # Training
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--all-seeds",  action="store_true",
                   help="Run all 5 seeds (longer but statistically sound)")
    p.add_argument("--n-episodes", type=int, default=None,
                   help="Override training episodes (default: from config)")
    # Evaluation
    p.add_argument("--n-eval",     type=int, default=1000,
                   help="Episodes per method for evaluation")
    # Output
    p.add_argument("--out-dir",    default="results")
    p.add_argument("--device",     default=None, help="cuda / cpu (auto-detected)")
    p.add_argument("--log-level",  default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    # Quick options
    p.add_argument("--skip-figures",  action="store_true")
    p.add_argument("--skip-baselines",action="store_true")
    return p.parse_args()


def stage_preprocess(args, config, logger) -> bool:
    """Stage 1: Validate and preprocess dataset."""
    logger.info("\n" + "━"*60)
    logger.info("STAGE 1: Data Preprocessing")
    logger.info("━"*60)

    if args.synthetic:
        logger.info("Using synthetic data — skipping real data preprocessing.")
        return True

    data_dir = Path(args.data_dir)
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(
            f"No CSV files found in {data_dir}.\n"
            "To use real CICIDS2017 data:\n"
            "  1. Download from https://www.unb.ca/cic/datasets/ids-2017.html\n"
            "  2. Place the 8 CSV files in data/cicids2017/\n"
            "  3. Re-run without --synthetic\n\n"
            "Continuing with synthetic data..."
        )
        args.synthetic = True
        return True

    logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")
    logger.info("Data preprocessing will happen within LSTM training.")
    return True


def stage_lstm(args, config, logger) -> bool:
    """Stage 2: Train LSTM attack predictor."""
    logger.info("\n" + "━"*60)
    logger.info("STAGE 2: LSTM Attack Prediction Training")
    logger.info("━"*60)

    import train_lstm as lstm_module

    lstm_args = argparse.Namespace(
        config=args.config,
        seed=args.seed,
        data_dir=args.data_dir,
        synthetic=args.synthetic,
        epochs=None,
        out_dir=args.out_dir,
        no_figures=args.skip_figures,
    )
    try:
        result = lstm_module.run(lstm_args)
        logger.info("LSTM training complete.")
        return True
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def stage_hdrl(args, config, logger) -> bool:
    """Stage 3: Train HDRL agent (CM-MTD)."""
    logger.info("\n" + "━"*60)
    logger.info("STAGE 3: HDRL Training (Algorithm 1 — CM-MTD)")
    logger.info("━"*60)

    import train_hdrl as hdrl_module

    hdrl_args = argparse.Namespace(
        config=args.config,
        seed=args.seed,
        all_seeds=args.all_seeds,
        lstm_ckpt="checkpoints/lstm_final.keras",
        synthetic=args.synthetic,
        n_episodes=args.n_episodes,
        out_dir=args.out_dir,
        device=args.device,
        small_net=True,
    )
    try:
        hdrl_module.run(hdrl_args)
        logger.info("HDRL training complete.")
        return True
    except Exception as e:
        logger.error(f"HDRL training failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def stage_evaluate(args, config, logger) -> bool:
    """Stage 4: Evaluate CM-MTD vs all baselines."""
    if args.skip_baselines:
        logger.info("Skipping baseline evaluation (--skip-baselines).")
        return True

    logger.info("\n" + "━"*60)
    logger.info("STAGE 4: Evaluation vs Baselines")
    logger.info("━"*60)

    import evaluate as eval_module

    eval_args = argparse.Namespace(
        config=args.config,
        dqn_ckpt="checkpoints/dqn_final.pt",
        ppo_ckpt="checkpoints/ppo_final.pt",
        lstm_ckpt="checkpoints/lstm_final.keras",
        n_eval=args.n_eval,
        all_seeds=args.all_seeds,
        synthetic=args.synthetic,
        out_dir=args.out_dir,
        device=args.device,
    )
    try:
        eval_module.run(eval_args)
        logger.info("Evaluation complete.")
        return True
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def stage_figures(args, config, logger) -> bool:
    """Stage 5: Generate all paper figures."""
    if args.skip_figures:
        logger.info("Skipping figure generation (--skip-figures).")
        return True

    logger.info("\n" + "━"*60)
    logger.info("STAGE 5: Paper Figure Reproduction")
    logger.info("━"*60)

    import reproduce_all_figures as fig_module

    fig_args = argparse.Namespace(
        results_dir=args.out_dir,
        synthetic_curves=args.synthetic,
        font_size=12,
        smooth=100,
        n_episodes=args.n_episodes or 10_000,
    )
    try:
        fig_module.run(fig_args)
        logger.info("All figures generated.")
        return True
    except Exception as e:
        logger.error(f"Figure generation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def main() -> None:
    args = parse_args()

    # Setup
    from utils.logger import setup_logger
    from utils.seed_utils import set_global_seed, get_device
    from config import load_config

    logger = setup_logger("cm_mtd_main", log_dir="logs", level=args.log_level)
    config = load_config(args.config)
    set_global_seed(args.seed)

    if args.device is None:
        args.device = get_device()
    config["experiment"]["device"] = args.device

    # Banner
    t0 = time.time()
    logger.info("╔" + "═"*58 + "╗")
    logger.info("║   CM-MTD: Collaborative Mutation-based MTD Framework   ║")
    logger.info("║   IEEE JSAC 2023 — Reproducible Research Pipeline      ║")
    logger.info("╚" + "═"*58 + "╝")
    logger.info(f"  Config:    {args.config}")
    logger.info(f"  Stage:     {args.stage}")
    logger.info(f"  Synthetic: {args.synthetic}")
    logger.info(f"  Device:    {args.device}")
    logger.info(f"  Seed:      {args.seed} {'(+ 4 more)' if args.all_seeds else ''}")

    # Ensure output directories exist
    for d in ["results/metrics", "results/figures", "checkpoints", "logs", "data/processed"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Run pipeline stages ───────────────────────────────────────────────────
    stage_map = {
        "preprocess": [stage_preprocess],
        "lstm":       [stage_preprocess, stage_lstm],
        "hdrl":       [stage_preprocess, stage_hdrl],
        "evaluate":   [stage_evaluate],
        "figures":    [stage_figures],
        "all":        [stage_preprocess, stage_lstm, stage_hdrl, stage_evaluate, stage_figures],
    }

    stages = stage_map[args.stage]
    results = {}
    for stage_fn in stages:
        name = stage_fn.__name__
        ok = stage_fn(args, config, logger)
        results[name] = "✓ OK" if ok else "✗ FAILED"
        if not ok:
            logger.warning(f"Stage {name} failed — continuing with remaining stages.")

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("\n" + "═"*60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info("\nStage results:")
    for stage, status in results.items():
        logger.info(f"  {stage:<20} {status}")
    logger.info("\nOutputs:")
    logger.info(f"  Models:   checkpoints/")
    logger.info(f"  Metrics:  {args.out_dir}/metrics/")
    logger.info(f"  Figures:  {args.out_dir}/figures/{{png,pdf,svg}}/")
    logger.info(f"  Logs:     logs/")
    logger.info("═"*60)

    # Save pipeline summary
    summary = {
        "args": vars(args),
        "stages": results,
        "elapsed_seconds": elapsed,
    }
    with open(Path(args.out_dir) / "metrics" / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
