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
import keras
from tensorflow.keras import layers

logger = logging.getLogger("cm_mtd")


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


@keras.saving.register_keras_serializable(package="cm_mtd")
def categorical_focal_loss(gamma: float = 2.0):
    """
    Focal loss for multi-class classification (Lin et al., 2017, "Focal
    Loss for Dense Object Detection").

    A static class_weight multiplier applies the SAME fixed factor to
    every example of a class for the entire training run. Empirically on
    this project that produced a narrow, unstable corridor: a mild weight
    collapsed to always-predict-majority-class, a stronger one
    overcorrected to mostly-predict-minority-class, and there was no
    value in between that gave genuine learning -- two collapse modes,
    no stable middle ground.

    Focal loss instead down-weights individual EXAMPLES the model is
    already confidently correct on (via the (1-p_t)^gamma factor,
    regardless of class) and concentrates gradient signal on whatever's
    currently hard to classify. This adapts continuously through
    training rather than fixing one ratio up front, and composes with
    class_weight (Keras applies class_weight as a per-sample multiplier
    on top of whatever loss function returns, so the two aren't
    mutually exclusive -- though typically you'd use one or the other).

    Registered via @keras.saving.register_keras_serializable so a saved
    model using this loss can be reloaded in a fresh process (plain
    closures aren't deserializable -- without this, loading a checkpoint
    trained with loss_function="focal" raises
    "Could not locate function 'loss_fn'").
    """
    @keras.saving.register_keras_serializable(package="cm_mtd")
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
        cross_entropy = -y_true * tf.math.log(y_pred)
        modulating_factor = tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(modulating_factor * cross_entropy, axis=-1)
    return loss_fn


# =============================================================================
# Stage 1: Event Classifier (per-row raw features -> classified event label)
# =============================================================================

class EventClassifier:
    """
    Stage 1 of the attack-prediction pipeline: classifies each row's raw
    features into a security-event class. The paper's Section IV frames
    this as "detection logs" -- in a real deployment this role is played
    by existing IDS/firewall/NetFlow tooling; here it's a Random Forest.

    This exists because an earlier design fed raw per-row features
    directly into the sequence model's sliding window (reasoning: richer
    input should help). scripts/diagnose_separability.py proved that
    backwards: a plain Random Forest gets very high recall directly from
    these features, while the identical features framed as a sequence
    collapsed to a majority-class predictor no matter how the
    loss/class-weighting was tuned. The features were never the problem;
    treating flow-level rows (no inherent row-to-row temporal coherence)
    as a time series was. This classifier does the part the data is
    actually good for -- per-row classification -- and hands its output
    to Stage 2 below, which does the part that's genuinely sequential:
    predicting the next label from recent label history.

    input_dim/num_classes are inferred from the data, never hardcoded.
    """

    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = 16, seed: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("Training Stage-1 event classifier (RandomForest, n_estimators=%d, max_depth=%s) on %d rows",
                    self.model.n_estimators, self.model.max_depth, len(X))
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("EventClassifier.fit() must be called before predict().")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("EventClassifier.fit() must be called before predict_proba().")
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        import joblib
        self.model = joblib.load(path)
        self._fitted = True


# =============================================================================
# Stage 2: LSTM Attack Predictor -- next-event forecasting over LABEL
# sequences (Section VI-A, Eq. 11-12, Fig. 4)
# =============================================================================

class LSTMAttackPredictor:
    """
    Predicts the next security-event class from the recent SEQUENCE of
    per-row event labels already classified by Stage 1 (EventClassifier)
    above. This matches the paper's own description (Section IV, Fig. 4):
    the LSTM's input is already-classified discrete events from detection
    logs, not raw continuous features -- see EventClassifier's docstring
    for why this project moved to a two-stage design.

    Architecture: Embedding(num_classes, embedding_dim) ("event embedding
    layer", Fig. 4) -> stacked LSTM -> Dense -> Softmax(num_classes).
    """

    def __init__(self, num_classes: int, cfg: dict):
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
        inputs = layers.Input(shape=(self.sequence_length,), dtype="int32", name="event_label_sequence")
        x = layers.Embedding(self.num_classes, cfg["embedding_dim"], name="event_embedding")(inputs)

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
        loss_name = cfg.get("loss_function", "categorical_crossentropy")
        loss = categorical_focal_loss(gamma=cfg.get("focal_gamma", 2.0)) if loss_name == "focal" else loss_name
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
            loss=loss,
            metrics=["accuracy"],
        )
        return model

    def build_sliding_windows(self, event_labels: np.ndarray,
                               target_labels: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        event_labels: 1D array of per-row classified event ids (Stage 1's
        output), in genuine chronological order.
        target_labels: the REAL ground-truth label to predict for each
        window (defaults to event_labels itself if not given, i.e. pure
        self-prediction). Normally you pass the true y aligned to the
        same rows, so Stage 2 learns to predict the actual future event
        from Stage 1's (possibly imperfect) observed history -- not just
        extrapolate Stage 1's own mistakes.
        """
        if target_labels is None:
            target_labels = event_labels
        L = self.sequence_length
        n = len(event_labels) - L
        if n <= 0:
            raise ValueError(
                f"Only {len(event_labels)} rows, fewer than sequence_length={L}; "
                f"cannot build any training windows."
            )
        windows = np.stack([event_labels[i:i + L] for i in range(n)], axis=0).astype(np.int32)
        targets = target_labels[L:L + n]
        return windows, targets

    def fit(self, event_labels: np.ndarray, target_labels: Optional[np.ndarray] = None,
            class_weight: Optional[dict] = None, seed: int = 42):
        windows, targets = self.build_sliding_windows(event_labels, target_labels)

        # Same validation-representativeness fix as before: Keras's
        # validation_split slices a contiguous tail off whatever array you
        # pass it. Shuffling WINDOWS (each window's own internal seq_len
        # ordering untouched) before the split keeps that tail
        # representative instead of one homogeneous chunk.
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(windows))
        windows, targets = windows[perm], targets[perm]

        y_onehot = keras.utils.to_categorical(targets, num_classes=self.num_classes)
        logger.info("Training LSTM predictor on %d label-sequence windows (seq_len=%d, num_classes=%d)",
                    len(windows), self.sequence_length, self.num_classes)
        history = self.model.fit(
            windows, y_onehot,
            batch_size=self.cfg["batch_size"],
            epochs=self.cfg["epochs"],
            validation_split=self.cfg["validation_split"],
            class_weight=class_weight,
            verbose=2,
        )
        return history

    def predict_next_events(self, label_window: np.ndarray) -> np.ndarray:
        """
        label_window: (n_nodes, seq_len) integer event-label sequences ->
        returns the argmax predicted class id per node, shape (n_nodes,).
        This is exactly the SMDP network state S_t (Section III-C-1).
        """
        x = tf.convert_to_tensor(label_window, dtype=tf.int32)
        probs = self._infer_fn(x).numpy()
        return np.argmax(probs, axis=-1)

    def predict_proba(self, label_window: np.ndarray) -> np.ndarray:
        x = tf.convert_to_tensor(label_window, dtype=tf.int32)
        return self._infer_fn(x).numpy()

    def predict_labels(self, event_labels: np.ndarray,
                        target_labels: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (y_true, y_pred) over sliding windows -- the raw material
        for a confusion matrix (paper's Fig. 8) or any metric beyond the
        single fidelity scalar in compute_fidelity().
        """
        windows, targets = self.build_sliding_windows(event_labels, target_labels)
        probs = self.model.predict(windows, verbose=0)
        preds = np.argmax(probs, axis=-1)
        return targets, preds

    def compute_fidelity(self, event_labels: np.ndarray, target_labels: Optional[np.ndarray] = None) -> dict:
        """
        Prediction accuracy fidelity, Eq. 18: Fidelity = sum_i Y(p_i,y_i) / |N|.
        Also returns per-class accuracy for the confusion-matrix-style
        breakdown reported in the paper's Table II.
        """
        targets, preds = self.predict_labels(event_labels, target_labels)

        fidelity = float(np.mean(preds == targets))
        per_class = {}
        for c in range(self.num_classes):
            mask = targets == c
            if mask.sum() > 0:
                per_class[c] = float(np.mean(preds[mask] == c))
        return {"fidelity": fidelity, "per_class_accuracy": per_class, "n": len(targets)}


# =============================================================================
# Stage 2 (alternative): Transformer Attack Predictor
# =============================================================================

@keras.saving.register_keras_serializable(package="cm_mtd")
class PositionalEmbedding(layers.Layer):
    """
    A learned per-position embedding table, added directly to the token
    embeddings it's called on. Deliberately NOT implemented as a separate
    Embedding layer called on a tf.range() constant -- that constant has
    no dependency on the model's Input, so Keras's functional-model graph
    tracing does not reliably keep it retrievable via get_layer() after a
    save/load round trip (confirmed by testing: it silently vanishes from
    model.layers). Implementing this as a proper Layer with its own
    add_weight(), called directly on the token embeddings (which DO derive
    from Input), keeps it correctly connected and serializable.
    """

    def __init__(self, sequence_length: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name="pos_embedding_table",
            shape=(self.sequence_length, self.d_model),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, token_embeddings):
        return token_embeddings + self.pos_embedding[tf.newaxis, :, :]

    def get_config(self):
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length, "d_model": self.d_model})
        return config


@keras.saving.register_keras_serializable(package="cm_mtd")
class LastPositionSlice(layers.Layer):
    """
    Extracts the last timestep's representation from a
    (batch, seq_len, d_model) tensor -- the Transformer's analogue of an
    LSTM's final hidden state. NOT implemented as layers.Lambda(lambda
    t: t[:, -1, :]): Keras 3 refuses to deserialize a Lambda wrapping a
    Python closure by default (arbitrary-code-execution risk), which
    would break loading a saved checkpoint in a fresh process -- the same
    class of bug as the focal-loss closure serialization issue elsewhere
    in this file. A plain registered Layer subclass has no such problem.
    """

    def call(self, x):
        return x[:, -1, :]


class TransformerAttackPredictor:
    """
    Drop-in alternative to LSTMAttackPredictor -- same interface
    (fit/predict_next_events/predict_proba/predict_labels/compute_fidelity),
    same input (Stage 1's event-label sequences), different sequence model.
    See the methodology report for full justification; summary, INCLUDING
    an honest negative result from scripts/compare_architectures.py:

      - Predictive fidelity and class-fairness are statistically tied with
        the LSTM on matched synthetic validation data (both architectures
        learned the same short, low-cardinality event sequence equally
        well) -- this is a "no regression" result, not a predictive win.
      - Initial justification for this change assumed attention's
        parallelizability would train faster than an LSTM's sequential
        recurrence. Measured head-to-head (scripts/compare_architectures.py,
        CPU, seq_len=10), that did NOT hold: the Transformer took ~1.7-2x
        longer to train and used more estimated energy than the LSTM at
        this scale. Self-attention's O(L^2) cost across 2 blocks x 4 heads
        outweighs recurrence's sequential-but-O(L) cost when L is this
        short; the parallelization advantage this architecture is known
        for is real but needs longer sequences and/or GPU execution to
        manifest, neither of which apply here. Reported plainly rather
        than omitted.
      - The genuine, scale-independent advantage is interpretability:
        self-attention gives a directly inspectable weight over "which
        past events mattered for this prediction" (see
        get_attention_weights() / metrics.py::attention_entropy) that an
        LSTM's opaque final hidden state does not offer without a separate
        post-hoc method (e.g. SHAP/LIME) bolted on. This is the actual
        basis for preferring it here, not speed.

    Architecture: token Embedding(num_classes, d_model) + learned
    positional Embedding(seq_len, d_model) -> N x [MultiHeadAttention ->
    Add&Norm -> position-wise FeedForward -> Add&Norm] -> take the last
    position's contextualized representation (analogous to an LSTM's
    final hidden state) -> Dense -> Softmax(num_classes).
    """

    def __init__(self, num_classes: int, cfg: dict):
        self.num_classes = num_classes
        self.cfg = cfg
        self.sequence_length = cfg["sequence_length"]
        self.d_model = cfg.get("d_model", cfg.get("embedding_dim", 32))
        self.num_heads = cfg.get("num_heads", 4)
        self.num_blocks = cfg.get("num_transformer_blocks", 2)
        self.d_ff = cfg.get("d_ff", self.d_model * 4)
        self.model = self._build_model()
        self._infer_fn = tf.function(
            lambda x: self.model(x, training=False), reduce_retracing=True
        )

    def _build_model(self) -> keras.Model:
        cfg = self.cfg
        L = self.sequence_length

        inputs = layers.Input(shape=(L,), dtype="int32", name="event_label_sequence")

        token_emb = layers.Embedding(self.num_classes, self.d_model, name="event_embedding")(inputs)
        # Learned positional embedding: with a short, FIXED sequence length,
        # a learned table is simpler than sinusoidal encoding and just as
        # effective -- there's no need to generalize beyond length L. See
        # PositionalEmbedding's docstring for why this is a custom Layer
        # rather than a separate Embedding(...)(tf.range(...)) call.
        x = PositionalEmbedding(L, self.d_model, name="position_embedding")(token_emb)

        for i in range(self.num_blocks):
            attn_out = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.d_model // self.num_heads,
                dropout=cfg["dropout"], name=f"self_attention_{i}",
            )(x, x)  # self-attention: query=key=value=x
            x = layers.LayerNormalization(epsilon=1e-6, name=f"attn_norm_{i}")(x + attn_out)

            ffn = keras.Sequential([
                layers.Dense(self.d_ff, activation="relu"),
                layers.Dense(self.d_model),
            ], name=f"ffn_{i}")
            ffn_out = ffn(x)
            x = layers.LayerNormalization(epsilon=1e-6, name=f"ffn_norm_{i}")(x + ffn_out)

        # Last position's contextualized representation -- it has attended
        # over the full window and plays the same role an LSTM's final
        # hidden state would (the target we predict is the row immediately
        # AFTER this window, so there's no leakage in using full
        # bidirectional attention within the window itself).
        x = LastPositionSlice(name="last_position")(x)

        for units in cfg["dense_units"]:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(cfg["dropout"])(x)

        outputs = layers.Dense(self.num_classes, activation="softmax", name="event_softmax")(x)

        model = keras.Model(inputs, outputs, name="transformer_attack_predictor")
        loss_name = cfg.get("loss_function", "categorical_crossentropy")
        loss = categorical_focal_loss(gamma=cfg.get("focal_gamma", 2.0)) if loss_name == "focal" else loss_name
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
            loss=loss,
            metrics=["accuracy"],
        )
        return model

    # -- Same windowing/fit/predict interface as LSTMAttackPredictor -----

    def build_sliding_windows(self, event_labels: np.ndarray,
                               target_labels: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        if target_labels is None:
            target_labels = event_labels
        L = self.sequence_length
        n = len(event_labels) - L
        if n <= 0:
            raise ValueError(
                f"Only {len(event_labels)} rows, fewer than sequence_length={L}; "
                f"cannot build any training windows."
            )
        windows = np.stack([event_labels[i:i + L] for i in range(n)], axis=0).astype(np.int32)
        targets = target_labels[L:L + n]
        return windows, targets

    def fit(self, event_labels: np.ndarray, target_labels: Optional[np.ndarray] = None,
            class_weight: Optional[dict] = None, seed: int = 42):
        windows, targets = self.build_sliding_windows(event_labels, target_labels)

        # Same validation-representativeness fix as LSTMAttackPredictor:
        # shuffle at the window level before Keras's contiguous-tail split.
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(windows))
        windows, targets = windows[perm], targets[perm]

        y_onehot = keras.utils.to_categorical(targets, num_classes=self.num_classes)
        logger.info("Training Transformer predictor on %d label-sequence windows (seq_len=%d, num_classes=%d)",
                    len(windows), self.sequence_length, self.num_classes)
        history = self.model.fit(
            windows, y_onehot,
            batch_size=self.cfg["batch_size"],
            epochs=self.cfg["epochs"],
            validation_split=self.cfg["validation_split"],
            class_weight=class_weight,
            verbose=2,
        )
        return history

    def predict_next_events(self, label_window: np.ndarray) -> np.ndarray:
        x = tf.convert_to_tensor(label_window, dtype=tf.int32)
        probs = self._infer_fn(x).numpy()
        return np.argmax(probs, axis=-1)

    def predict_proba(self, label_window: np.ndarray) -> np.ndarray:
        x = tf.convert_to_tensor(label_window, dtype=tf.int32)
        return self._infer_fn(x).numpy()

    def predict_labels(self, event_labels: np.ndarray,
                        target_labels: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        windows, targets = self.build_sliding_windows(event_labels, target_labels)
        probs = self.model.predict(windows, verbose=0)
        preds = np.argmax(probs, axis=-1)
        return targets, preds

    def compute_fidelity(self, event_labels: np.ndarray, target_labels: Optional[np.ndarray] = None) -> dict:
        targets, preds = self.predict_labels(event_labels, target_labels)
        fidelity = float(np.mean(preds == targets))
        per_class = {}
        for c in range(self.num_classes):
            mask = targets == c
            if mask.sum() > 0:
                per_class[c] = float(np.mean(preds[mask] == c))
        return {"fidelity": fidelity, "per_class_accuracy": per_class, "n": len(targets)}

    # -- Transformer-specific: attention extraction for the Trust metric -

    def get_attention_weights(self, label_window: np.ndarray, block: int = -1) -> np.ndarray:
        """
        Returns self-attention weights for the given batch of windows,
        shape (batch, num_heads, seq_len, seq_len). block=-1 uses the
        last transformer block (closest to the prediction). This is the
        raw material for metrics.py's attention_entropy (Trust dimension)
        and for plotting "what the model looked at" per prediction.

        Layers are looked up fresh via self.model.get_layer() every call,
        rather than using references captured at construction time -- this
        is deliberate: after loading a saved checkpoint (predictor.model
        gets replaced with the deserialized model), stored references from
        the original build would still point to the randomly-initialized
        construction-time layers, silently returning attention weights
        from untrained weights with no error. get_layer() always reflects
        whatever weights self.model currently holds.
        """
        target_idx = block if block >= 0 else self.num_blocks + block
        x = tf.convert_to_tensor(label_window, dtype=tf.int32)

        h = self.model.get_layer("event_embedding")(x)
        h = self.model.get_layer("position_embedding")(h)

        for i in range(target_idx + 1):
            attn_layer = self.model.get_layer(f"self_attention_{i}")
            if i < target_idx:
                attn_out = attn_layer(h, h)
                h = self.model.get_layer(f"attn_norm_{i}")(h + attn_out)
                ffn_out = self.model.get_layer(f"ffn_{i}")(h)
                h = self.model.get_layer(f"ffn_norm_{i}")(h + ffn_out)
            else:
                _, scores = attn_layer(h, h, return_attention_scores=True)
                return scores.numpy()
        raise RuntimeError("Unreachable")


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
