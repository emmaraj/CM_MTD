"""
main.py
-------
Entry point for the CM-MTD pipeline. Orchestrates:
  1. Loading config + dataset (dataset-agnostic; shapes inferred at
     runtime via src/datasets/'s adapter registry)
  2. Pretraining the Stage-2 attack predictor (LSTM or Transformer,
     Section VI-A)
  3. Running the hierarchical DQN/PPO training loop (Algorithm 1)
  4. Evaluation: prediction fidelity (Eq. 18) and Defense Success Ratio (Eq. 19)

experiment.dataset ("cicids2017" | "5g_nidd") and experiment.predictor
("transformer" | "lstm") are the pipeline's two independent experimental
variables -- override either from config.yaml or the CLI flags below.
Checkpoints/results are namespaced by both (see
config_parser.get_run_paths) so switching one doesn't clobber the other's
outputs.

Usage:
    python -m src.main --config config/config.yaml --mode all
    python -m src.main --config config/config.yaml --mode train_lstm
    python -m src.main --config config/config.yaml --mode train_rl
    python -m src.main --config config/config.yaml --mode evaluate

    # Override the config file's dataset/predictor without editing it:
    python -m src.main --config config/config.yaml --mode all --dataset 5g_nidd --predictor lstm
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_parser import (
    load_config, apply_cli_overrides, active_dataset_cfg, active_predictor_cfg,
    get_run_paths, set_global_seed, load_dataset, compute_class_weights,
    configure_device, reset_tf_session, setup_logging,
)
from src.environment import (
    DigitalTwinNetworkEnv, AttackPredictionStateWrapper, NUM_MACRO_ACTIONS,
)
from src.models import LSTMAttackPredictor, TransformerAttackPredictor, EventClassifier, DQNAgent, PPOAgent
from src import metrics as metrics_mod


def _build_predictor(predictor_type: str, num_classes: int, cfg: dict):
    if predictor_type == "transformer":
        return TransformerAttackPredictor(num_classes, cfg["transformer"])
    elif predictor_type == "lstm":
        return LSTMAttackPredictor(num_classes, cfg["lstm"])
    else:
        raise ValueError(f"Unknown predictor {predictor_type!r} (expected 'transformer' or 'lstm')")


def train_attack_predictor(cfg: dict, dataset, run_paths: dict, logger):
    """
    Trains the full two-stage attack predictor:
      Stage 1 (EventClassifier): raw features -> classified event label.
      Stage 2 (LSTMAttackPredictor or TransformerAttackPredictor, per
        config.experiment.predictor): recent label history -> next label.
    Also computes Fairness/Trust/Energy metrics (src/metrics.py) alongside
    the paper's own fidelity (Eq. 18) evaluation, per config.evaluation.
    See models.py's EventClassifier / TransformerAttackPredictor
    docstrings for why this replaced a single LSTM trained directly on
    raw feature sequences.
    """
    reset_tf_session()
    predictor_type = cfg["experiment"]["predictor"]
    stage2_cfg = active_predictor_cfg(cfg)

    # --- Stage 1: classify every row from its raw features -------------
    clf_cfg = cfg["event_classifier"]
    classifier = EventClassifier(
        n_estimators=clf_cfg["n_estimators"], max_depth=clf_cfg["max_depth"],
        seed=cfg["experiment"]["seed"],
    )
    classifier.fit(dataset.X_train, dataset.y_train)

    train_pred_labels = classifier.predict(dataset.X_train)
    test_pred_labels = classifier.predict(dataset.X_test)
    stage1_train_acc = float(np.mean(train_pred_labels == dataset.y_train))
    stage1_test_acc = float(np.mean(test_pred_labels == dataset.y_test))
    logger.info("Stage-1 EventClassifier accuracy: train=%.4f test=%.4f", stage1_train_acc, stage1_test_acc)

    # --- Stage 2: forecast the next REAL label from Stage-1's observed --
    # label history. Input is Stage 1's own predictions (what a deployed
    # system would actually have logged); target is the true y, so Stage 2
    # learns to predict the real future event, not just extrapolate
    # Stage 1's mistakes.
    class_weight = compute_class_weights(
        dataset.y_train, dataset.num_classes,
        strategy=stage2_cfg["class_weight_strategy"],
        max_ratio=stage2_cfg["max_class_weight_ratio"],
    )

    eval_cfg = cfg.get("evaluation", {})
    energy_report = None
    if eval_cfg.get("compute_energy", True):
        with metrics_mod.track_energy(f"{predictor_type}_training") as energy_report:
            predictor = _build_predictor(predictor_type, dataset.num_classes, cfg)
            history = predictor.fit(train_pred_labels, dataset.y_train, class_weight=class_weight,
                                     seed=cfg["experiment"]["seed"])
    else:
        predictor = _build_predictor(predictor_type, dataset.num_classes, cfg)
        history = predictor.fit(train_pred_labels, dataset.y_train, class_weight=class_weight,
                                 seed=cfg["experiment"]["seed"])

    fidelity_metrics = predictor.compute_fidelity(test_pred_labels, dataset.y_test)
    logger.info("%s test fidelity (Eq. 18): %.4f over %d windows",
                predictor_type, fidelity_metrics["fidelity"], fidelity_metrics["n"])
    for c, acc in fidelity_metrics["per_class_accuracy"].items():
        name = dataset.class_names[c] if c < len(dataset.class_names) else str(c)
        logger.info("  per-class accuracy [%s]: %.4f", name, acc)

    results_dir = run_paths["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    eval_report = {
        "dataset": cfg["experiment"]["dataset"],
        "predictor_type": predictor_type,
        "fidelity": fidelity_metrics["fidelity"],
    }

    # --- Fairness ---------------------------------------------------------
    if eval_cfg.get("compute_fairness", True):
        fairness = metrics_mod.class_fairness(fidelity_metrics["per_class_accuracy"])
        eval_report["fairness_class"] = fairness
        logger.info("Fairness (class recall parity): gap=%.4f equity_ratio=%s",
                    fairness["recall_gap"],
                    f"{fairness['equity_ratio']:.4f}" if fairness["equity_ratio"] is not None else "N/A")

    # --- Trust --------------------------------------------------------------
    if eval_cfg.get("compute_trust", True):
        windows, targets = predictor.build_sliding_windows(test_pred_labels, dataset.y_test)
        probs = predictor.predict_proba(windows)
        ece_result = metrics_mod.expected_calibration_error(
            probs, targets, n_bins=eval_cfg.get("calibration_bins", 10)
        )
        eval_report["calibration_ece"] = ece_result["ece"]
        logger.info("Trust (calibration): ECE=%.4f", ece_result["ece"])

        stability = metrics_mod.prediction_stability(
            predictor, test_pred_labels, dataset.y_test,
            benign_class=eval_cfg.get("stability_benign_class", 0),
            n_samples=eval_cfg.get("stability_perturbation_samples", 500),
            seed=cfg["experiment"]["seed"],
        )
        eval_report["prediction_stability"] = stability
        if stability["flip_rate"] is not None:
            logger.info("Trust (stability): flip_rate=%.4f over %d perturbation tests",
                        stability["flip_rate"], stability["n_tested"])

        if predictor_type == "transformer":
            attn = predictor.get_attention_weights(windows[:min(2000, len(windows))])
            attn_entropy = metrics_mod.attention_entropy(attn)
            eval_report["attention_entropy"] = attn_entropy
            logger.info("Trust (attention entropy, normalized): mean=%.4f",
                        attn_entropy["mean_normalized_entropy"])

    # --- Energy -------------------------------------------------------------
    if energy_report is not None:
        n_params = metrics_mod.count_trainable_params(predictor.model)
        sample_window = predictor.build_sliding_windows(test_pred_labels[:100], dataset.y_test[:100])[0][:1]
        latency_ms = metrics_mod.benchmark_inference_latency(predictor.predict_next_events, sample_window)
        eval_report["energy"] = {
            "training_wall_clock_seconds": energy_report.wall_clock_seconds,
            "training_estimated_kwh": energy_report.estimated_energy_kwh,
            "training_estimated_co2_kg": energy_report.estimated_co2_kg,
            "energy_measurement_method": energy_report.method,
            "trainable_params": n_params,
            "inference_ms_per_call": latency_ms,
        }
        logger.info("Energy: training=%.1fs (%s), %d params, inference=%.3fms/call",
                    energy_report.wall_clock_seconds, energy_report.method, n_params, latency_ms)

    with open(os.path.join(results_dir, f"evaluation_report_{predictor_type}.json"), "w") as f:
        json.dump(eval_report, f, indent=2, default=float)
    logger.info("Saved Fairness/Trust/Energy report to %s/evaluation_report_%s.json", results_dir, predictor_type)

    # Fig. 7 material: per-epoch accuracy/loss (paper plots these against
    # "episode" for the LSTM curve -- we treat training epoch as that axis).
    with open(os.path.join(results_dir, "stage2_history.json"), "w") as f:
        json.dump(history.history, f, indent=2)

    # Fig. 8 / Table II material: raw (y_true, y_pred) pairs over the test set.
    y_true, y_pred = predictor.predict_labels(test_pred_labels, dataset.y_test)
    np.savez(
        os.path.join(results_dir, "stage2_confusion.npz"),
        y_true=y_true, y_pred=y_pred,
        class_names=np.array(dataset.class_names),
        fidelity=fidelity_metrics["fidelity"],
    )
    logger.info("Saved training history + confusion data to %s/", results_dir)

    ckpt_dir = run_paths["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    classifier.save(os.path.join(ckpt_dir, "event_classifier.joblib"))
    predictor.model.save(os.path.join(ckpt_dir, f"{predictor_type}_predictor.keras"))
    logger.info("Saved Stage-1 classifier and Stage-2 %s predictor to %s/", predictor_type, ckpt_dir)
    return classifier, predictor


def train_rl(cfg: dict, dataset, classifier, predictor, run_paths: dict, logger):
    seed = cfg["experiment"]["seed"]
    net_cfg = cfg["network"]
    reward_cfg = cfg["reward"]
    dataset_cfg = active_dataset_cfg(cfg)

    base_env = DigitalTwinNetworkEnv(dataset, net_cfg, reward_cfg, dataset_cfg, seed=seed)
    predictor_type = cfg["experiment"]["predictor"]
    seq_len = active_predictor_cfg(cfg)["sequence_length"]
    env = AttackPredictionStateWrapper(base_env, classifier, predictor, seq_len)

    state_dim = base_env.n_nodes
    dqn = DQNAgent(state_dim=state_dim, num_actions=NUM_MACRO_ACTIONS, cfg=cfg["dqn"], seed=seed)
    ppo = PPOAgent(
        state_dim=state_dim, n_nodes=base_env.n_nodes, num_ip_pools=net_cfg["num_ip_pools"],
        num_flows=net_cfg["num_flows"], num_route_candidates=3, cfg=cfg["ppo"], seed=seed,
    )

    M = cfg["training"]["num_episodes"]
    T = cfg["training"]["steps_per_episode"]
    K = cfg["ppo"]["steps_per_macro_action"]
    eval_every = cfg["training"]["eval_every_episodes"]
    save_every = cfg["training"]["save_every_episodes"]
    ckpt_dir = run_paths["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    upper_rewards_log, lower_rewards_log, dsr_log = [], [], []

    for episode in range(1, M + 1):
        obs, _ = env.reset()
        obs = obs.astype(np.float32)
        episode_upper_reward, episode_lower_reward = 0.0, 0.0

        for _t in range(T):
            macro = dqn.select_action(obs)

            ppo_states, ppo_ip_actions, ppo_route_actions = [], [], []
            ppo_log_probs, ppo_values, ppo_rewards, ppo_dones = [], [], [], []

            current_obs = obs
            upper_reward_accum = 0.0
            for _k in range(K):
                micro_action, log_prob, value = ppo.select_action(current_obs)
                full_action = {
                    "macro": macro,
                    "ip_assignment": micro_action["ip_assignment"],
                    "route_assignment": micro_action["route_assignment"],
                }
                next_obs, reward, terminated, truncated, _info = env.step(full_action)
                next_obs = next_obs.astype(np.float32)

                ppo_states.append(current_obs)
                ppo_ip_actions.append(micro_action["ip_assignment"])
                ppo_route_actions.append(micro_action["route_assignment"])
                ppo_log_probs.append(log_prob)
                ppo_values.append(value)
                ppo_rewards.append(reward)
                ppo_dones.append(float(terminated or truncated))

                upper_reward_accum += reward
                episode_lower_reward += reward
                current_obs = next_obs

            _, _, last_value = ppo.select_action(current_obs)
            advantages, returns = ppo.compute_gae(
                np.array(ppo_rewards, dtype=np.float32),
                np.array(ppo_values, dtype=np.float32),
                np.array(ppo_dones, dtype=np.float32),
                last_value,
            )
            ppo.update(
                np.array(ppo_states), np.array(ppo_ip_actions), np.array(ppo_route_actions),
                np.array(ppo_log_probs), advantages, returns,
            )

            dqn.store(obs, macro, upper_reward_accum, current_obs, False)
            dqn.train_step()

            episode_upper_reward += upper_reward_accum
            obs = current_obs

        dqn.decay_epsilon()
        upper_rewards_log.append(episode_upper_reward)
        lower_rewards_log.append(episode_lower_reward)

        if episode % eval_every == 0:
            dsr = base_env.stats.dsr()
            dsr_log.append((episode, dsr))
            logger.info(
                "Episode %d/%d | upper_R=%.2f | lower_R=%.2f | DSR(Eq.19)=%.3f | epsilon=%.3f",
                episode, M, episode_upper_reward, episode_lower_reward, dsr, dqn.epsilon,
            )
            base_env.stats.reset()  # windowed DSR, matching the paper's per-1000-episode averaging

        if episode % save_every == 0:
            dqn.q_network.save(os.path.join(ckpt_dir, f"dqn_q_network_ep{episode}.keras"))
            ppo.actor.save(os.path.join(ckpt_dir, f"ppo_actor_ep{episode}.keras"))
            ppo.critic.save(os.path.join(ckpt_dir, f"ppo_critic_ep{episode}.keras"))
            logger.info("Saved RL checkpoints at episode %d", episode)

    results_dir = run_paths["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    dsr_episodes = np.array([e for e, _ in dsr_log])
    dsr_values = np.array([d for _, d in dsr_log])
    np.savez(
        os.path.join(results_dir, "rl_training_curves.npz"),
        upper_rewards=np.array(upper_rewards_log),   # Fig. 11(a)/(c)/(e) material
        lower_rewards=np.array(lower_rewards_log),   # Fig. 11(b)/(d)/(f) material
        dsr_episodes=dsr_episodes,
        dsr_values=dsr_values,                        # Fig. 9-style DSR-over-training material
    )
    logger.info("Saved RL training curves to %s/rl_training_curves.npz", results_dir)

    return {
        "dqn": dqn, "ppo": ppo, "base_env": base_env,
        "upper_rewards": upper_rewards_log, "lower_rewards": lower_rewards_log,
        "dsr_log": dsr_log,
    }


def main():
    parser = argparse.ArgumentParser(description="CM-MTD training pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--mode", type=str, default="all",
                         choices=["train_lstm", "train_rl", "all", "evaluate"])
    parser.add_argument("--dataset", type=str, default=None, choices=["cicids2017", "5g_nidd"],
                         help="Override config.experiment.dataset without editing config.yaml.")
    parser.add_argument("--predictor", type=str, default=None, choices=["transformer", "lstm"],
                         help="Override config.experiment.predictor without editing config.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, dataset=args.dataset, predictor=args.predictor)
    run_paths = get_run_paths(cfg)

    logger = setup_logging(run_paths["log_dir"])
    set_global_seed(cfg["experiment"]["seed"])
    configure_device(cfg["experiment"]["device"])

    logger.info("Run: dataset=%s predictor=%s (checkpoints -> %s, results -> %s)",
                cfg["experiment"]["dataset"], cfg["experiment"]["predictor"],
                run_paths["checkpoint_dir"], run_paths["results_dir"])

    logger.info("Loading dataset...")
    dataset = load_dataset(cfg)
    logger.info("Dataset ready: input_dim=%d, num_classes=%d, classes=%s",
                dataset.input_dim, dataset.num_classes, dataset.class_names)

    predictor = None
    classifier = None
    predictor_type = cfg["experiment"]["predictor"]
    ckpt_dir = run_paths["checkpoint_dir"]
    predictor_ckpt_path = os.path.join(ckpt_dir, f"{predictor_type}_predictor.keras")
    classifier_ckpt_path = os.path.join(ckpt_dir, "event_classifier.joblib")

    if args.mode in ("train_lstm", "all"):
        classifier, predictor = train_attack_predictor(cfg, dataset, run_paths, logger)
    else:
        import tensorflow as tf
        if not (os.path.exists(predictor_ckpt_path) and os.path.exists(classifier_ckpt_path)):
            raise FileNotFoundError(
                f"No checkpoint found at {predictor_ckpt_path} / {classifier_ckpt_path}. "
                f"Run --mode train_lstm first (with dataset={cfg['experiment']['dataset']!r}, "
                f"predictor={predictor_type!r})."
            )
        logger.info("Loading Stage-1 classifier from %s", classifier_ckpt_path)
        classifier = EventClassifier()
        classifier.load(classifier_ckpt_path)
        logger.info("Loading Stage-2 %s predictor from %s", predictor_type, predictor_ckpt_path)
        keras_model = tf.keras.models.load_model(predictor_ckpt_path, compile=False)
        predictor = _build_predictor(predictor_type, dataset.num_classes, cfg)
        predictor.model = keras_model
        # _infer_fn was tf.function-compiled against the freshly-initialized
        # model at construction time; rebind it to the loaded model so
        # inference (and, for the Transformer, attention extraction) uses
        # the trained weights rather than the discarded random init.
        predictor._infer_fn = tf.function(
            lambda x: predictor.model(x, training=False), reduce_retracing=True
        )

    if args.mode in ("train_rl", "all"):
        results = train_rl(cfg, dataset, classifier, predictor, run_paths, logger)
        final_dsr = results["base_env"].stats.dsr()
        logger.info("Training complete. Final-window DSR (Eq. 19): %.3f", final_dsr)

    if args.mode == "evaluate":
        test_pred_labels = classifier.predict(dataset.X_test)
        metrics = predictor.compute_fidelity(test_pred_labels, dataset.y_test)
        logger.info("Evaluation fidelity (Eq. 18): %.4f", metrics["fidelity"])


if __name__ == "__main__":
    main()
