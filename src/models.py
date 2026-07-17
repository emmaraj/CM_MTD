"""
models.py
---------
TensorFlow/Keras implementations of the three learned components of
CM-MTD: the LSTM attack predictor (Section VI-A), the upper-layer DQN
over macro-actions (Section VI-B, Eq. 13-14), and the lower-layer PPO
over micro-actions (Section VI-B, Eq. 15-17).

Everything here is pure TensorFlow/Keras (no PyTorch). A prior iteration
of this project mixed TF (LSTM/DQN) with PyTorch (PPO) and hit GPU memory
contention between the two runtimes; standardizing on one framework
avoids that class of bug entirely, and keeps `config_parser.configure_device`
sufficient to control every model in the pipeline.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger("cm_mtd")


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# =============================================================================
# LSTM Attack Predictor (Section VI-A, Eq. 11-12, Fig. 4)
# =============================================================================

class LSTMAttackPredictor:
    """
    Predicts the next security-event class for a node from its recent
    history of CICIDS-2017 (or any config-swapped dataset's) feature
    vectors. input_dim is inferred from the dataset, never hardcoded.

    Architecture: per-timestep Dense projection ("event embedding layer",
    Fig. 4) -> stacked LSTM -> Dense -> Softmax(num_classes).
    """

    def __init__(self, input_dim: int, num_classes: int, cfg: dict):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.cfg = cfg
        self.sequence_length = cfg["sequence_length"]
        self.model = self._build_model()
        # Plain eager __call__ still costs ~30ms/call of pure Python/TF
        # dispatch overhead regardless of input size -- irrelevant for
        # training (one call per batch) but fatal for this method, which
        # is invoked once per RL environment step (millions of times over
        # a full run). tf.function traces the graph once (input shape here
        # never changes) and then runs at native speed (~1ms/call).
        self._infer_fn = tf.function(
            lambda x: self.model(x, training=False), reduce_retracing=True
        )

    def _build_model(self) -> keras.Model:
        cfg = self.cfg
        inputs = layers.Input(shape=(self.sequence_length, self.input_dim), name="event_sequence")

        x = layers.TimeDistributed(
            layers.Dense(cfg["embedding_dim"], activation="relu"), name="event_embedding"
        )(inputs)

        lstm_units = cfg["lstm_units"]
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            x = layers.LSTM(units, return_sequences=return_sequences, dropout=cfg["dropout"],
                             name=f"lstm_{i}")(x)

        for units in cfg["dense_units"]:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(cfg["dropout"])(x)

        outputs = layers.Dense(self.num_classes, activation="softmax", name="event_softmax")(x)

        model = keras.Model(inputs, outputs, name="lstm_attack_predictor")
        loss = cfg.get("loss_function", "categorical_crossentropy")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
            loss=loss,
            metrics=["accuracy"],
        )
        return model

    def build_sliding_windows(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Turn a flat (n_samples, input_dim) trace into (window, next_label)
        supervised pairs, per Eq. 12: X^t_k = {E^{t-L}_k,...,E^{t-1}_k},
        Y^t_k = E^t_k. Windows are built with plain slicing over the
        existing empirical order -- no shuffling, no synthetic rows.
        """
        L = self.sequence_length
        n = len(X) - L
        if n <= 0:
            raise ValueError(
                f"Dataset has only {len(X)} rows, fewer than sequence_length={L}; "
                f"cannot build any training windows."
            )
        X_windows = np.stack([X[i:i + L] for i in range(n)], axis=0)
        y_targets = y[L:L + n]
        return X_windows, y_targets

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, class_weight: Optional[dict] = None,
            seed: int = 42):
        X_windows, y_targets = self.build_sliding_windows(X_train, y_train)

        # keras.Model.fit's validation_split takes a contiguous slice off
        # the END of whatever array you pass it -- it does NOT shuffle.
        # X_windows is built from the full, genuinely time-ordered dataset
        # (by design, so each window's internal sequence is real), which
        # means the last validation_split fraction of *windows* is a
        # single contiguous chunk of the original file -- e.g. whichever
        # attack burst happened to land at the end. That produces a
        # validation set that can be wildly unrepresentative (and
        # explains val_accuracy swinging by 70 points epoch to epoch: a
        # small decision-boundary shift flips an entire homogeneous
        # block at once). Shuffling WINDOWS (not raw rows -- each window's
        # own internal seq_len ordering is untouched) before the split
        # fixes this while keeping every window's temporal content valid.
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(X_windows))
        X_windows, y_targets = X_windows[perm], y_targets[perm]

        y_onehot = keras.utils.to_categorical(y_targets, num_classes=self.num_classes)
        logger.info("Training LSTM predictor on %d sequence windows (seq_len=%d, input_dim=%d)",
                    len(X_windows), self.sequence_length, self.input_dim)
        history = self.model.fit(
            X_windows, y_onehot,
            batch_size=self.cfg["batch_size"],
            epochs=self.cfg["epochs"],
            validation_split=self.cfg["validation_split"],
            class_weight=class_weight,
            verbose=2,
        )
        return history

    def predict_next_events(self, feature_window: np.ndarray) -> np.ndarray:
        """
        feature_window: (n_nodes, seq_len, input_dim) -> returns the
        argmax predicted class id per node, shape (n_nodes,). This is
        exactly the SMDP network state S_t (Section III-C-1).
        """
        x = tf.convert_to_tensor(feature_window, dtype=tf.float32)
        probs = self._infer_fn(x).numpy()
        return np.argmax(probs, axis=-1)

    def predict_proba(self, feature_window: np.ndarray) -> np.ndarray:
        x = tf.convert_to_tensor(feature_window, dtype=tf.float32)
        return self._infer_fn(x).numpy()

    def predict_labels(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (y_true, y_pred) over sliding windows of (X, y) — the raw
        material for a confusion matrix (paper's Fig. 8) or any other
        metric beyond the single fidelity scalar in compute_fidelity().
        """
        X_windows, y_targets = self.build_sliding_windows(X, y)
        probs = self.model.predict(X_windows, verbose=0)
        preds = np.argmax(probs, axis=-1)
        return y_targets, preds

    def compute_fidelity(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Prediction accuracy fidelity, Eq. 18: Fidelity = sum_i Y(p_i,y_i) / |N|.
        Also returns per-class accuracy for the confusion-matrix-style
        breakdown reported in the paper's Table II.
        """
        y_targets, preds = self.predict_labels(X, y)

        fidelity = float(np.mean(preds == y_targets))
        per_class = {}
        for c in range(self.num_classes):
            mask = y_targets == c
            if mask.sum() > 0:
                per_class[c] = float(np.mean(preds[mask] == c))
        return {"fidelity": fidelity, "per_class_accuracy": per_class, "n": len(y_targets)}


# =============================================================================
# Upper layer: DQN over macro-actions (Section VI-B, Eq. 13-14, Table I)
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.buffer: deque = deque(maxlen=capacity)
        self._rng = np.random.RandomState(seed)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        idx = self._rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Upper-layer agent choosing macro-actions O = {o_c, o_a, o_r, o_s}
    (Section VI-B, Eq. 13-14). Standard DQN with target network and
    replay buffer, per Algorithm 1 lines 1, 5-6, 14-18, 35-36, 42.
    """

    def __init__(self, state_dim: int, num_actions: int, cfg: dict, seed: int = 0):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.cfg = cfg
        self.gamma = cfg["gamma"]
        self.epsilon = cfg["epsilon_start"]
        self.epsilon_end = cfg["epsilon_end"]
        self.epsilon_decay = (cfg["epsilon_start"] - cfg["epsilon_end"]) / max(cfg["epsilon_decay_episodes"], 1)
        self.batch_size = cfg["batch_size"]
        self.min_replay_before_train = cfg["min_replay_before_train"]
        self.target_update_every = cfg["target_update_every"]

        self.replay = ReplayBuffer(cfg["replay_buffer_size"], seed=seed)
        self._rng = np.random.RandomState(seed)

        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.target_network.set_weights(self.q_network.get_weights())
        self._train_steps = 0

        # Same eager-overhead problem as the LSTM predictor: select_action
        # is called once per macro-step, train_step's two forward passes
        # once per training step -- both add up fast across 10k+ episodes.
        # tf.function traces each network's graph once; set_weights() on
        # target_network later mutates the underlying tf.Variables in
        # place, so this compiled function keeps seeing fresh weights
        # without needing to retrace.
        self._q_infer_fn = tf.function(
            lambda x: self.q_network(x, training=False), reduce_retracing=True
        )
        self._target_infer_fn = tf.function(
            lambda x: self.target_network(x, training=False), reduce_retracing=True
        )

    def _build_network(self) -> keras.Model:
        inputs = layers.Input(shape=(self.state_dim,))
        x = inputs
        for units in self.cfg["hidden_layers"]:
            x = layers.Dense(units, activation=self.cfg["activation"])(x)
        outputs = layers.Dense(self.num_actions, activation="linear")(x)
        model = keras.Model(inputs, outputs, name="dqn_q_network")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.cfg["learning_rate"]), loss="mse")
        return model

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy macro-action selection, Algorithm 1 lines 12-18."""
        if not greedy and self._rng.random() <= self.epsilon:
            return int(self._rng.randint(self.num_actions))
        state_t = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        q_values = self._q_infer_fn(state_t).numpy()[0]
        return int(np.argmax(q_values))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def store(self, state, action, reward, next_state, done) -> None:
        self.replay.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """One gradient step, Algorithm 1 lines 35-36. Returns the loss, or
        None if there isn't enough replay data yet.

        Uses direct __call__ (not .predict()) and train_on_batch (not
        .fit()) throughout -- both .predict() and .fit() rebuild
        significant internal machinery on every call (batching pipeline,
        callbacks, progress bars), which is fine for occasional large
        calls but disastrous when called every training step across
        thousands of episodes. train_on_batch() does exactly one gradient
        update with none of that overhead.
        """
        if len(self.replay) < max(self.batch_size, self.min_replay_before_train):
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states = states.astype(np.float32)
        next_states = next_states.astype(np.float32)

        target_q_next = self._target_infer_fn(tf.convert_to_tensor(next_states)).numpy()
        max_target_q = np.max(target_q_next, axis=1)
        targets = self._q_infer_fn(tf.convert_to_tensor(states)).numpy()
        targets[np.arange(self.batch_size), actions] = rewards + (1.0 - dones) * self.gamma * max_target_q

        loss = self.q_network.train_on_batch(states, targets)

        self._train_steps += 1
        if self._train_steps % self.target_update_every == 0:
            self.target_network.set_weights(self.q_network.get_weights())  # Algorithm 1 line 42

        return float(loss if np.isscalar(loss) else loss[0])


# =============================================================================
# Lower layer: PPO over micro-actions (Section VI-B, Eq. 15-17, Table I)
# =============================================================================

class PPOAgent:
    """
    Lower-layer agent choosing micro-actions: which IP pool each node is
    assigned, and which precomputed candidate route each flow uses
    (Section V, Eq. 4-9). Deployed only for the mutation schemes selected
    by the current macro-action.

    Design note: the paper's own complexity analysis (Section VII)
    describes the actor as outputting "the parameters of a normal
    distribution with 2 neurons" -- i.e. a single-scalar Gaussian policy.
    Our micro-action is inherently a discrete, multi-dimensional choice
    (one IP-pool index per node, one route index per flow), so a single
    continuous scalar cannot parameterize it directly. We instead use a
    multi-head categorical policy (one softmax head per node, one per
    flow) and apply the same clipped-surrogate PPO objective (Eq. 16)
    over the joint log-probability. This is a deliberate architecture
    adaptation to a discrete micro-action space, not a paper ambiguity.
    """

    def __init__(self, state_dim: int, n_nodes: int, num_ip_pools: int,
                 num_flows: int, num_route_candidates: int, cfg: dict, seed: int = 0):
        self.state_dim = state_dim
        self.n_nodes = n_nodes
        self.num_ip_pools = num_ip_pools
        self.num_flows = num_flows
        self.num_route_candidates = num_route_candidates
        self.cfg = cfg
        self.gamma = cfg["gamma"]
        self.gae_lambda = cfg["gae_lambda"]
        self.clip_epsilon = cfg["clip_epsilon"]
        self.entropy_coef = cfg["entropy_coef"]
        self.value_coef = cfg["value_coef"]

        self._rng = np.random.RandomState(seed)

        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=cfg["learning_rate"])
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=cfg["learning_rate"])

        # select_action() runs once per environment step -- the same
        # millions-of-calls frequency as the LSTM predictor -- and
        # update()'s inner loop runs one gradient step per minibatch per
        # epoch, which also adds up fast. Plain eager __call__ costs
        # ~30ms/call of pure dispatch overhead regardless of input size;
        # tf.function compiles each graph once and runs at native speed.
        self._actor_infer_fn = tf.function(
            lambda x: self.actor(x, training=False), reduce_retracing=True
        )
        self._critic_infer_fn = tf.function(
            lambda x: self.critic(x, training=False), reduce_retracing=True
        )
        self._train_step_fn = tf.function(self._train_step_impl, reduce_retracing=True)

    def _build_actor(self) -> keras.Model:
        inputs = layers.Input(shape=(self.state_dim,))
        x = inputs
        for units in self.cfg["hidden_layers"]:
            x = layers.Dense(units, activation=self.cfg["activation"])(x)
        ip_logits = layers.Dense(self.n_nodes * self.num_ip_pools, name="ip_logits_flat")(x)
        ip_logits = layers.Reshape((self.n_nodes, self.num_ip_pools), name="ip_logits")(ip_logits)
        route_logits = layers.Dense(self.num_flows * self.num_route_candidates, name="route_logits_flat")(x)
        route_logits = layers.Reshape((self.num_flows, self.num_route_candidates), name="route_logits")(route_logits)
        return keras.Model(inputs, [ip_logits, route_logits], name="ppo_actor")

    def _build_critic(self) -> keras.Model:
        inputs = layers.Input(shape=(self.state_dim,))
        x = inputs
        for units in self.cfg["hidden_layers"]:
            x = layers.Dense(units, activation=self.cfg["activation"])(x)
        value = layers.Dense(1, activation="linear")(x)
        return keras.Model(inputs, value, name="ppo_critic")

    def select_action(self, state: np.ndarray):
        """Returns (action_dict, log_prob, value) for a single state."""
        state_batch = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        ip_logits, route_logits = self._actor_infer_fn(state_batch)
        ip_logits = ip_logits.numpy()[0]        # (n_nodes, num_ip_pools)
        route_logits = route_logits.numpy()[0]  # (num_flows, num_route_candidates)

        ip_probs = _softmax(ip_logits, axis=-1)
        route_probs = _softmax(route_logits, axis=-1)

        ip_assignment = np.array([
            self._rng.choice(self.num_ip_pools, p=ip_probs[i]) for i in range(self.n_nodes)
        ])
        route_assignment = np.array([
            self._rng.choice(self.num_route_candidates, p=route_probs[f]) for f in range(self.num_flows)
        ])

        log_prob = (
            np.sum(np.log(ip_probs[np.arange(self.n_nodes), ip_assignment] + 1e-8))
            + np.sum(np.log(route_probs[np.arange(self.num_flows), route_assignment] + 1e-8))
        )
        value = float(self._critic_infer_fn(state_batch).numpy()[0, 0])

        action = {"ip_assignment": ip_assignment, "route_assignment": route_assignment}
        return action, float(log_prob), value

    def compute_gae(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: float):
        """
        Generalized Advantage Estimation, matching the TD-error/advantage
        recursion in Algorithm 1 lines 24-27 (Â_k = sum (γξ)^{q-k} δ_q).
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        values_ext = np.append(values, last_value)
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values_ext[t + 1] * (1.0 - dones[t]) - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def _train_step_impl(self, states, ip_actions, route_actions, old_log_probs, advantages, returns):
        """
        One actor + one critic gradient step on a single minibatch.
        Wrapped in tf.function via self._train_step_fn (set in __init__) --
        this is called once per minibatch per epoch inside update() below,
        which without compilation would mean millions of eager
        GradientTape passes (~30ms of pure dispatch overhead each) over a
        full training run. Compiled, each call runs at native graph speed.
        """
        with tf.GradientTape() as tape:
            ip_logits, route_logits = self.actor(states, training=True)
            ip_log_probs_all = tf.nn.log_softmax(ip_logits, axis=-1)
            route_log_probs_all = tf.nn.log_softmax(route_logits, axis=-1)

            ip_selected = tf.gather(ip_log_probs_all, ip_actions, batch_dims=2)
            route_selected = tf.gather(route_log_probs_all, route_actions, batch_dims=2)
            new_log_probs = tf.reduce_sum(ip_selected, axis=1) + tf.reduce_sum(route_selected, axis=1)

            ratio = tf.exp(new_log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

            ip_probs = tf.nn.softmax(ip_logits, axis=-1)
            route_probs = tf.nn.softmax(route_logits, axis=-1)
            entropy = (
                -tf.reduce_mean(tf.reduce_sum(ip_probs * ip_log_probs_all, axis=-1))
                - tf.reduce_mean(tf.reduce_sum(route_probs * route_log_probs_all, axis=-1))
            )
            actor_loss = policy_loss - self.entropy_coef * entropy

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        with tf.GradientTape() as tape:
            values_pred = tf.squeeze(self.critic(states, training=True), axis=-1)
            critic_loss = self.value_coef * tf.reduce_mean(tf.square(returns - values_pred))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return actor_loss, critic_loss

    def update(self, states, ip_actions, route_actions, old_log_probs, advantages, returns):
        """
        Clipped-surrogate PPO update, Eq. 16-17, Algorithm 1 lines 37-40.
        Minibatch size is fixed by config (default steps_per_macro_action
        divides evenly by minibatch_size), so every call to
        self._train_step_fn sees the same shape and reuses the one
        compiled graph traced on the first call -- no retracing overhead
        in the default configuration. If you change config such that the
        last minibatch of an epoch has a different size than the rest,
        that one shape will trigger an extra (one-time) retrace; still
        correct, just marginally slower.
        """
        states = tf.convert_to_tensor(np.asarray(states), dtype=tf.float32)
        ip_actions = tf.convert_to_tensor(np.asarray(ip_actions), dtype=tf.int32)
        route_actions = tf.convert_to_tensor(np.asarray(route_actions), dtype=tf.int32)
        old_log_probs = tf.convert_to_tensor(np.asarray(old_log_probs), dtype=tf.float32)

        advantages = np.asarray(advantages, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(np.asarray(returns), dtype=tf.float32)

        n = states.shape[0]
        minibatch_size = min(self.cfg["minibatch_size"], n)
        actor_losses, critic_losses = [], []

        for _ in range(self.cfg["update_epochs"]):
            idx_all = np.arange(n)
            self._rng.shuffle(idx_all)
            for start in range(0, n, minibatch_size):
                mb_idx = idx_all[start:start + minibatch_size]
                actor_loss, critic_loss = self._train_step_fn(
                    tf.gather(states, mb_idx),
                    tf.gather(ip_actions, mb_idx),
                    tf.gather(route_actions, mb_idx),
                    tf.gather(old_log_probs, mb_idx),
                    tf.gather(advantages, mb_idx),
                    tf.gather(returns, mb_idx),
                )
                actor_losses.append(float(actor_loss))
                critic_losses.append(float(critic_loss))

        return float(np.mean(actor_losses)), float(np.mean(critic_losses))
