"""
main.py
-------
Entry point for the CM-MTD pipeline. Orchestrates:
  1. Loading config + dataset (dataset-agnostic; shapes inferred at runtime)
  2. Pretraining the LSTM attack predictor (Section VI-A)
  3. Running the hierarchical DQN/PPO training loop (Algorithm 1)
  4. Evaluation: prediction fidelity (Eq. 18) and Defense Success Ratio (Eq. 19)

Usage:
    python -m src.main --config config/config.yaml --mode all
    python -m src.main --config config/config.yaml --mode train_lstm
    python -m src.main --config config/config.yaml --mode train_rl
    python -m src.main --config config/config.yaml --mode evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_parser import (
    load_config, set_global_seed, load_dataset, compute_class_weights,
    configure_device, reset_tf_session, setup_logging,
)
from src.environment import (
    DigitalTwinNetworkEnv, LSTMStatePredictionWrapper, NUM_MACRO_ACTIONS,
)
from src.models import LSTMAttackPredictor, DQNAgent, PPOAgent


def train_lstm(cfg: dict, dataset, logger):
    reset_tf_session()
    predictor = LSTMAttackPredictor(dataset.input_dim, dataset.num_classes, cfg["lstm"])

    class_weight = compute_class_weights(
        dataset.y_train, dataset.num_classes,
        strategy=cfg["lstm"]["class_weight_strategy"],
        max_ratio=cfg["lstm"]["max_class_weight_ratio"],
    )
    history = predictor.fit(dataset.X_train, dataset.y_train, class_weight=class_weight,
                             seed=cfg["experiment"]["seed"])

    metrics = predictor.compute_fidelity(dataset.X_test, dataset.y_test)
    logger.info("LSTM test fidelity (Eq. 18): %.4f over %d windows", metrics["fidelity"], metrics["n"])
    for c, acc in metrics["per_class_accuracy"].items():
        name = dataset.class_names[c] if c < len(dataset.class_names) else str(c)
        logger.info("  per-class accuracy [%s]: %.4f", name, acc)

    results_dir = cfg["experiment"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # Fig. 7 material: per-epoch accuracy/loss (paper plots these against
    # "episode" for the LSTM curve -- we treat training epoch as that axis).
    with open(os.path.join(results_dir, "lstm_history.json"), "w") as f:
        json.dump(history.history, f, indent=2)

    # Fig. 8 / Table II material: raw (y_true, y_pred) pairs over the test set.
    y_true, y_pred = predictor.predict_labels(dataset.X_test, dataset.y_test)
    np.savez(
        os.path.join(results_dir, "lstm_confusion.npz"),
        y_true=y_true, y_pred=y_pred,
        class_names=np.array(dataset.class_names),
        fidelity=metrics["fidelity"],
    )
    logger.info("Saved LSTM training history + confusion data to %s/", results_dir)

    ckpt_dir = cfg["experiment"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    predictor.model.save(os.path.join(ckpt_dir, "lstm_predictor.keras"))
    logger.info("Saved LSTM predictor to %s", os.path.join(ckpt_dir, "lstm_predictor.keras"))
    return predictor


def train_rl(cfg: dict, dataset, predictor, logger):
    seed = cfg["experiment"]["seed"]
    net_cfg = cfg["network"]
    reward_cfg = cfg["reward"]

    base_env = DigitalTwinNetworkEnv(dataset, net_cfg, reward_cfg, cfg["data"], seed=seed)
    env = LSTMStatePredictionWrapper(base_env, predictor, cfg["lstm"]["sequence_length"])

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
    ckpt_dir = cfg["experiment"]["checkpoint_dir"]
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

    results_dir = cfg["experiment"]["results_dir"]
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg["experiment"]["log_dir"])
    set_global_seed(cfg["experiment"]["seed"])
    configure_device(cfg["experiment"]["device"])

    logger.info("Loading dataset...")
    dataset = load_dataset(cfg)
    logger.info("Dataset ready: input_dim=%d, num_classes=%d, classes=%s",
                dataset.input_dim, dataset.num_classes, dataset.class_names)

    predictor = None
    ckpt_dir = cfg["experiment"]["checkpoint_dir"]
    lstm_ckpt_path = os.path.join(ckpt_dir, "lstm_predictor.keras")

    if args.mode in ("train_lstm", "all"):
        predictor = train_lstm(cfg, dataset, logger)
    else:
        import tensorflow as tf
        if not os.path.exists(lstm_ckpt_path):
            raise FileNotFoundError(
                f"No LSTM checkpoint found at {lstm_ckpt_path}. Run --mode train_lstm first."
            )
        logger.info("Loading LSTM predictor from %s", lstm_ckpt_path)
        keras_model = tf.keras.models.load_model(lstm_ckpt_path)
        predictor = LSTMAttackPredictor(dataset.input_dim, dataset.num_classes, cfg["lstm"])
        predictor.model = keras_model

    if args.mode in ("train_rl", "all"):
        results = train_rl(cfg, dataset, predictor, logger)
        final_dsr = results["base_env"].stats.dsr()
        logger.info("Training complete. Final-window DSR (Eq. 19): %.3f", final_dsr)

    if args.mode == "evaluate":
        metrics = predictor.compute_fidelity(dataset.X_test, dataset.y_test)
        logger.info("Evaluation fidelity (Eq. 18): %.4f", metrics["fidelity"])


if __name__ == "__main__":
    main()
