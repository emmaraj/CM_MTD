"""
LSTM Attack Prediction Model (LSTMNet) — Section VI-A of the paper.

Architecture:
    Security Logs → Preprocess Layer → Event Embedding Layer → LSTM Layer → Softmax

    Embedding(n_classes, embedding_dim)
        → LSTM(128) → Dropout
        → LSTM(64)  → Dropout
        → Dense(128) → Dense(64)
        → Dense(n_classes, activation='softmax')

Loss function: sparse categorical cross-entropy (mean square error variant
               from the paper is approximated via soft targets; standard
               cross-entropy is used here for multi-class classification).
Optimizer:     Adam (paper Eq. references [40]).

The LSTM cell equations (Eq. 11 in paper):
    f_t = σ(ω_f · [x_t, h_{t-1}] + b_f)      ← forget gate
    i_t = σ(ω_i · [x_t, h_{t-1}] + b_i)      ← input gate
    o_t = σ(ω_o · [x_t, h_{t-1}] + b_o)      ← output gate
    C̃_t = tanh(ω_c · [x_t, h_{t-1}] + b_c)  ← cell candidate
    C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t         ← cell state
    h_t = o_t ⊙ tanh(C_t)                     ← hidden state
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("cm_mtd.lstm")

# Deferred import — supports both tf.keras (TF≥2.x) and standalone keras (≥3.x)
_tf = None
_keras = None


def _get_tf():
    """Lazy-load TensorFlow or standalone Keras. Raises ImportError if neither found."""
    global _tf, _keras
    if _keras is not None:
        return _tf, _keras

    # Try standalone keras first (keras ≥ 3.0 ships independently)
    try:
        import keras as _k
        _keras = _k
        _tf = _k  # keras 3 is self-contained
        logger.info("Using standalone keras %s", _k.__version__)
        return _tf, _keras
    except ImportError:
        pass

    # Fall back to tf.keras
    try:
        import tensorflow as tf
        _tf = tf
        _keras = tf.keras
        logger.info("Using tf.keras (TF %s)", tf.__version__)
        return _tf, _keras
    except ImportError:
        raise ImportError(
            "Neither 'keras' nor 'tensorflow' is installed.\n"
            "Install with: pip install tensorflow>=2.15\n"
            "or:           pip install keras>=3.0"
        )


class LSTMAttackPredictor:
    """
    LSTM-based attack prediction model.

    Predicts the next security event (attack type) for each network node
    given a history of L previous events.

    Args:
        n_classes: Number of attack classes (default 8 for CICIDS2017).
        sequence_length: Input sequence length L.
        embedding_dim: Dimension of event embedding vectors.
        lstm_units: List of hidden units per LSTM layer.
        dense_units: List of units per Dense layer after LSTM.
        dropout_rate: Dropout probability.
        learning_rate: Adam optimizer learning rate.
        l2_reg: L2 regularization coefficient.
    """

    def __init__(
        self,
        n_classes: int = 8,
        sequence_length: int = 10,
        embedding_dim: int = 64,
        lstm_units: List[int] = None,
        dense_units: List[int] = None,
        dropout_rate: float = 0.3,
        recurrent_dropout: float = 0.2,
        learning_rate: float = 1e-3,
        l2_reg: float = 1e-4,
    ) -> None:
        self.n_classes = n_classes
        self.L = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units or [128, 64]
        self.dense_units = dense_units or [128, 64]
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg

        self.model: Optional[object] = None
        self._history: Optional[object] = None

    # ─── Model Construction ─────────────────────────────────────────────────

    def build(self) -> None:
        """Build the Keras model graph."""
        tf, keras = _get_tf()
        reg = keras.regularizers.l2(self.l2_reg)

        inp = keras.Input(shape=(self.L,), name="event_sequence")

        # ── Preprocess / Embedding Layer ────────────────────────────────────
        # Maps discrete event IDs → dense vectors (context-based representation)
        x = keras.layers.Embedding(
            input_dim=self.n_classes,
            output_dim=self.embedding_dim,
            name="event_embedding",
        )(inp)

        # ── LSTM Layers ─────────────────────────────────────────────────────
        for idx, units in enumerate(self.lstm_units):
            return_seq = idx < len(self.lstm_units) - 1  # only last layer returns single vector
            x = keras.layers.LSTM(
                units,
                return_sequences=return_seq,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=reg,
                name=f"lstm_{idx}",
            )(x)
            x = keras.layers.BatchNormalization(name=f"bn_lstm_{idx}")(x)

        # ── Dense Layers ─────────────────────────────────────────────────────
        for idx, units in enumerate(self.dense_units):
            x = keras.layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=reg,
                name=f"dense_{idx}",
            )(x)
            x = keras.layers.Dropout(self.dropout_rate, name=f"dropout_dense_{idx}")(x)

        # ── Output Layer — Softmax over attack classes ────────────────────
        out = keras.layers.Dense(
            self.n_classes,
            activation="softmax",
            name="attack_prediction",
        )(x)

        self.model = keras.Model(inputs=inp, outputs=out, name="LSTMNet")

        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=1.0,
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        logger.info(f"LSTMNet built. Parameters: {self.model.count_params():,}")
        self.model.summary(print_fn=logger.debug)

    # ─── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        verbose: int = 1,
        log_dir: str = "logs",
        checkpoint_path: str = "checkpoints/lstm_best.keras",
        use_class_weights: bool = True,
    ) -> Dict:
        """
        Train the LSTM model with early stopping and LR scheduling.

        Args:
            X_train: Training sequences [n_train, L].
            y_train: Training labels [n_train].
            X_val:   Validation sequences.
            y_val:   Validation labels.
            epochs:  Maximum training epochs.
            batch_size: Mini-batch size.
            patience: Early stopping patience.
            lr_patience: Patience for LR reduction.
            lr_factor: LR reduction factor.
            min_lr: Minimum learning rate.
            verbose: Keras verbosity level.
            log_dir: TensorBoard log directory.
            checkpoint_path: Path to save best model weights.

        Returns:
            Training history dictionary.
        """
        if self.model is None:
            self.build()

        tf, keras = _get_tf()
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=lr_factor,
                patience=lr_patience,
                min_lr=min_lr,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss" if X_val is not None else "loss",
                save_best_only=True,
                verbose=0,
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(Path(log_dir) / "lstm"),
                histogram_freq=0,
            ),
        ]

        val_data = (X_val, y_val) if X_val is not None else None

        # ── Class weights — critical for imbalanced attack datasets ──────────
        # Cap at max_weight=20 to prevent extreme upweighting of classes
        # with very few samples (e.g. Infiltration has only 36 in CICIDS-2017).
        # Without the cap, weights of 8000x cause gradient instability and
        # accuracy collapses below random chance.
        class_weight_dict = None
        if use_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            classes_present = np.unique(y_train)

            # Drop classes with fewer than 50 samples — too few to learn
            min_samples = 50
            valid_classes = np.array([
                c for c in classes_present
                if np.sum(y_train == c) >= min_samples
            ])
            if len(valid_classes) < len(classes_present):
                dropped = set(classes_present) - set(valid_classes)
                logger.warning(
                    "Dropping %d class(es) with < %d samples: %s "
                    "(these are replaced by majority vote during prediction).",
                    len(dropped), min_samples, dropped
                )

            weights = compute_class_weight(
                class_weight="balanced",
                classes=valid_classes,
                y=y_train[np.isin(y_train, valid_classes)],
            )

            # Hard cap — prevents any single class dominating gradients
            max_weight = 20.0
            weights = np.clip(weights, 0.1, max_weight)

            class_weight_dict = {int(c): float(w)
                                 for c, w in zip(valid_classes, weights)}
            logger.info("Class weights (capped at %.0fx): %s",
                        max_weight,
                        {k: f"{v:.2f}" for k, v in class_weight_dict.items()})

        self._history = self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            class_weight=class_weight_dict,
        )

        logger.info("LSTM training complete.")
        return self._history.history

    # ─── Inference ──────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        Predict attack class labels (argmax of softmax output).

        Args:
            X: Input sequences [n, L].

        Returns:
            Integer label array [n].
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build() or load().")
        probs = self.model.predict(X, batch_size=batch_size, verbose=0)
        return np.argmax(probs, axis=-1)

    def predict_proba(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        Predict attack class probabilities (softmax output).

        Used as the SMDP network state S_t = {e^{t+1}_1, ..., e^{t+1}_n}.

        Args:
            X: Input sequences [n, L].

        Returns:
            Probability matrix [n, n_classes].
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build() or load().")
        return self.model.predict(X, batch_size=batch_size, verbose=0)

    def predict_next_event(self, recent_events: np.ndarray) -> int:
        """
        Predict the next single event from a recent history sequence.

        Args:
            recent_events: Shape [L] — recent event IDs.

        Returns:
            Predicted attack class integer.
        """
        x = recent_events.reshape(1, self.L)
        probs = self.predict_proba(x)
        return int(np.argmax(probs[0]))

    # ─── Evaluation ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[List[str]] = None,
        batch_size: int = 512,
    ) -> Dict:
        """
        Full evaluation: fidelity, confusion matrix, per-class metrics.

        Implements Eq. 18: Fidelity = Σ 𝕐(p_i, y_i) / |N|

        Args:
            X_test: Test sequences [n_test, L].
            y_test: Ground truth labels [n_test].
            class_names: Optional class name list.
            batch_size: Inference batch size.

        Returns:
            Metrics dictionary including per-class precision/recall/F1.
        """
        from utils.metrics import compute_classification_metrics
        y_pred = self.predict(X_test, batch_size=batch_size)
        metrics = compute_classification_metrics(y_test, y_pred, class_names)
        logger.info(
            f"Evaluation — Fidelity: {metrics['fidelity']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f}"
        )
        return metrics

    def get_training_curves(self) -> Dict[str, List[float]]:
        """Return training history for plotting Figure 7."""
        if self._history is None:
            return {}
        return self._history.history

    # ─── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights and architecture."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"LSTM model saved to {path}")

    def load(self, path: str) -> None:
        """Load a saved model."""
        tf, keras = _get_tf()
        self.model = keras.models.load_model(path)
        logger.info(f"LSTM model loaded from {path}")

    # ─── Configuration Helpers ───────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: Dict) -> "LSTMAttackPredictor":
        """Instantiate from a config dict (e.g., from config.yaml)."""
        lstm_cfg = config.get("lstm", {})
        data_cfg = config.get("data", {})
        attack_cfg = config.get("attack_labels", {})
        return cls(
            n_classes=lstm_cfg.get("n_classes", len(attack_cfg) or 8),
            sequence_length=data_cfg.get("sequence_length", 10),
            embedding_dim=lstm_cfg.get("embedding_dim", 64),
            lstm_units=lstm_cfg.get("lstm_units", [128, 64]),
            dense_units=lstm_cfg.get("dense_units", [128, 64]),
            dropout_rate=lstm_cfg.get("dropout_rate", 0.3),
            recurrent_dropout=lstm_cfg.get("recurrent_dropout", 0.2),
            learning_rate=lstm_cfg.get("learning_rate", 1e-3),
            l2_reg=lstm_cfg.get("l2_regularization", 1e-4),
        )
