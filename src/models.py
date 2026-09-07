"""
models.py
---------
PyTorch implementations of the three learned components of CM-MTD: the
attack predictor (LSTM or Transformer, Section VI-A), the upper-layer DQN
over macro-actions (Section VI-B, Eq. 13-14), and the lower-layer PPO
over micro-actions (Section VI-B, Eq. 15-17).

MIGRATION NOTE (TensorFlow/Keras -> PyTorch): this file was previously
pure TensorFlow/Keras, itself a deliberate standardization after an
EARLIER iteration mixed TF (LSTM/DQN) with PyTorch (PPO) and hit GPU
memory contention between the two runtimes (see README's "Known failure
modes"). That principle -- one framework, end-to-end, never two runtimes
sharing a GPU -- still holds; only WHICH framework changed. Everything
below is pure PyTorch (no TensorFlow), so `config_parser.configure_device`
remains sufficient to control every model in the pipeline, same as before.

The public interface of every class here (constructor signature minus
the explicit `device` argument, `fit`/`predict_next_events`/
`predict_proba`/`predict_labels`/`compute_fidelity` for the predictors,
`select_action`/`store`/`train_step` for DQNAgent, `select_action`/
`compute_gae`/`update` for PPOAgent) is unchanged from the Keras version,
so environment.py and main.py's calling code did not need to change
shape -- only how each class is built and checkpointed.

Performance note carried over from the Keras version: `predict_next_events`
(LSTMAttackPredictor/TransformerAttackPredictor) and `select_action`
(DQNAgent/PPOAgent) are each called once per RL environment step --
millions of times across a full run (T×K×M ≈ 6.25M at config.yaml's
defaults). The Keras version wrapped every such call in `tf.function` to
avoid ~30ms/call of eager dispatch overhead. PyTorch's eager mode has
materially less per-call Python overhead than TF1-style eager execution
did, and every hot-path inference call below runs under
`torch.inference_mode()` (a stricter/faster variant of `torch.no_grad()`)
rather than full autograd-tracked eager mode. This has NOT been
independently re-benchmarked against the old tf.function numbers at real
config.yaml scale (10,000 episodes) -- if a full run's wall-clock time
looks materially worse than the README's ~14-hour CPU figure, profile
`select_action`/`predict_next_events` first; `torch.compile()` on the
relevant `nn.Module`s is the natural next lever, deliberately not applied
here since `torch.compile` on Windows/CPU (Emma's dev machine) is a much
newer, less battle-tested path than the same feature on Linux, and this
migration is large enough already without also debugging that.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("cm_mtd")


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class SimpleHistory:
    """
    Minimal stand-in for keras.callbacks.History -- just a `.history`
    dict of per-epoch metric lists ({"loss": [...], "accuracy": [...],
    "val_loss": [...], "val_accuracy": [...]}). main.py does
    `json.dump(history.history, ...)` and generate_figures.py reads
    `history["loss"]`/`history["accuracy"]`/etc. straight out of that
    JSON -- keeping this shape identical to Keras's History means neither
    of those call sites needed to change for the PyTorch migration.
    """

    def __init__(self):
        self.history: dict = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    def record(self, loss, accuracy, val_loss, val_accuracy) -> None:
        self.history["loss"].append(float(loss))
        self.history["accuracy"].append(float(accuracy))
        self.history["val_loss"].append(float(val_loss))
        self.history["val_accuracy"].append(float(val_accuracy))


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
    class_weight (applied below as a per-sample multiplier on top of
    whatever this returns, same as Keras's class_weight semantics --
    though typically you'd use one or the other).

    Returns a callable (logits, target_idx, sample_weight=None) -> mean
    scalar loss, matching how _build_loss_fn's other branches are shaped.
    """
    def loss_fn(logits: torch.Tensor, target_idx: torch.Tensor,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_classes = logits.shape[-1]
        probs = F.softmax(logits, dim=-1).clamp(1e-8, 1.0 - 1e-8)
        log_probs = torch.log(probs)
        target_onehot = F.one_hot(target_idx, num_classes=num_classes).float()
        cross_entropy = -target_onehot * log_probs
        modulating_factor = (1.0 - probs) ** gamma
        per_sample = (modulating_factor * cross_entropy).sum(dim=-1)
        if sample_weight is not None:
            per_sample = per_sample * sample_weight
        return per_sample.mean()
    return loss_fn


def _build_loss_fn(loss_name: str, focal_gamma: float = 2.0):
    """
    Returns a callable (logits, target_idx, sample_weight=None) -> mean
    scalar loss for whichever loss_function config.yaml names.
    class_weight is applied as a PER-SAMPLE multiplier looked up by each
    sample's true class -- see _class_weight_tensor below -- matching
    Keras's class_weight semantics (a per-sample, not per-batch, scale).
    """
    if loss_name == "focal":
        return categorical_focal_loss(gamma=focal_gamma)

    if loss_name == "categorical_crossentropy":
        def loss_fn(logits, target_idx, sample_weight=None):
            per_sample = F.cross_entropy(logits, target_idx, reduction="none")
            if sample_weight is not None:
                per_sample = per_sample * sample_weight
            return per_sample.mean()
        return loss_fn

    if loss_name == "mse":
        def loss_fn(logits, target_idx, sample_weight=None):
            num_classes = logits.shape[-1]
            probs = F.softmax(logits, dim=-1)
            target_onehot = F.one_hot(target_idx, num_classes=num_classes).float()
            per_sample = F.mse_loss(probs, target_onehot, reduction="none").mean(dim=-1)
            if sample_weight is not None:
                per_sample = per_sample * sample_weight
            return per_sample.mean()
        return loss_fn

    raise ValueError(f"Unknown loss_function: {loss_name!r} (expected 'focal', "
                      f"'categorical_crossentropy', or 'mse')")


def _class_weight_tensor(class_weight: Optional[dict], num_classes: int, device: torch.device) -> Optional[torch.Tensor]:
    if class_weight is None:
        return None
    w = torch.ones(num_classes, dtype=torch.float32, device=device)
    for c, weight in class_weight.items():
        w[int(c)] = float(weight)
    return w


# =============================================================================
# Stage 1: Event Classifier (per-row raw features -> classified event label)
# =============================================================================

class EventClassifier:
    """
    Stage 1 of the attack-prediction pipeline: classifies each row's raw
    features into a security-event class. The paper's Section IV frames
    this as "detection logs" -- in a real deployment this role is played
    by existing IDS/firewall/NetFlow tooling; here it's a Random Forest.

    Unaffected by the TF->PyTorch migration -- always scikit-learn, never
    part of either deep learning framework.

    This exists because an earlier design fed raw per-row features
    directly into the sequence model's sliding window (reasoning: richer
    input should help). scripts/diagnose_separability.py proved that
    backwards: a plain Random Forest gets very high recall directly from
    these features, while the identical features framed as a sequence
    collapsed to a majority-class predictor no matter how the
    loss/class-weighting was tuned. This classifier does the part the
    data is actually good for -- per-row classification -- and hands its
    output to Stage 2 below, which does the part that's genuinely
    sequential: predicting the next label from recent label history.

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

class _LSTMNet(nn.Module):
    """
    Embedding(num_classes, embedding_dim) ("event embedding layer", Fig.
    4) -> stacked LSTM -> Dense -> raw logits (softmax applied outside,
    by the loss function / predict_proba, for numerically-stable
    log-softmax during training).

    One nn.Dropout is applied to the INPUT of each LSTM block rather
    than passed as nn.LSTM's own `dropout=` kwarg -- PyTorch's nn.LSTM
    only applies its internal dropout BETWEEN stacked layers within a
    single nn.LSTM instance (no-op for a single-layer module, which is
    what each stacked block is here), unlike Keras's layers.LSTM(dropout=...)
    which applies recurrent input dropout regardless of stack depth. An
    explicit nn.Dropout on each block's input is the direct PyTorch
    equivalent of what the Keras version's per-layer `dropout=cfg["dropout"]`
    was doing.
    """

    def __init__(self, num_classes: int, embedding_dim: int, lstm_units: list,
                 dense_units: list, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

        self.lstm_dropouts = nn.ModuleList()
        self.lstm_blocks = nn.ModuleList()
        input_size = embedding_dim
        for units in lstm_units:
            self.lstm_dropouts.append(nn.Dropout(dropout))
            self.lstm_blocks.append(nn.LSTM(input_size, units, batch_first=True))
            input_size = units
        self.post_lstm_dropout = nn.Dropout(dropout)

        dense_layers = []
        prev = input_size
        for units in dense_units:
            dense_layers += [nn.Linear(prev, units), nn.ReLU(), nn.Dropout(dropout)]
            prev = units
        self.dense = nn.Sequential(*dense_layers)
        self.out = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)  # (batch, seq_len, embedding_dim)
        for drop, lstm in zip(self.lstm_dropouts, self.lstm_blocks):
            h = drop(h)
            h, _ = lstm(h)
        h_last = h[:, -1, :]  # final timestep -- analogue of Keras's return_sequences=False on the last layer
        h_last = self.post_lstm_dropout(h_last)
        h_last = self.dense(h_last)
        return self.out(h_last)


class LSTMAttackPredictor:
    """
    Predicts the next security-event class from the recent SEQUENCE of
    per-row event labels already classified by Stage 1 (EventClassifier)
    above. This matches the paper's own description (Section IV, Fig. 4):
    the LSTM's input is already-classified discrete events from detection
    logs, not raw continuous features -- see EventClassifier's docstring
    for why this project moved to a two-stage design.
    """

    def __init__(self, num_classes: int, cfg: dict, device: Optional[torch.device] = None):
        self.num_classes = num_classes
        self.cfg = cfg
        self.sequence_length = cfg["sequence_length"]
        self.device = device or torch.device("cpu")
        self.model = _LSTMNet(
            num_classes=num_classes, embedding_dim=cfg["embedding_dim"],
            lstm_units=cfg["lstm_units"], dense_units=cfg["dense_units"], dropout=cfg["dropout"],
        ).to(self.device)
        self._loss_fn = _build_loss_fn(cfg.get("loss_function", "categorical_crossentropy"),
                                        cfg.get("focal_gamma", 2.0))
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"])

    def build_sliding_windows(self, event_labels: np.ndarray,
                               target_labels: Optional[np.ndarray] = None) -> tuple:
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
        windows = np.stack([event_labels[i:i + L] for i in range(n)], axis=0).astype(np.int64)
        targets = target_labels[L:L + n]
        return windows, targets

    def fit(self, event_labels: np.ndarray, target_labels: Optional[np.ndarray] = None,
            class_weight: Optional[dict] = None, seed: int = 42) -> SimpleHistory:
        windows, targets = self.build_sliding_windows(event_labels, target_labels)

        # Same validation-representativeness fix as the Keras version:
        # Keras's validation_split slices a contiguous tail off whatever
        # array you pass it, so this project shuffles at the WINDOW level
        # (each window's own internal seq_len ordering untouched) before
        # taking that tail, keeping it representative instead of one
        # homogeneous chunk. Kept identical here even though this training
        # loop is hand-rolled, not Keras's -- the tail-slice-as-val-set
        # convention is preserved for continuity with past history/figures.
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(windows))
        windows, targets = windows[perm], targets[perm]

        val_split = self.cfg["validation_split"]
        n_val = int(len(windows) * val_split)
        n_train = len(windows) - n_val
        X_train, y_train = windows[:n_train], targets[:n_train]
        X_val, y_val = windows[n_train:], targets[n_train:]

        weight_tensor = _class_weight_tensor(class_weight, self.num_classes, self.device)

        logger.info("Training LSTM predictor on %d label-sequence windows (seq_len=%d, num_classes=%d), "
                    "%d train / %d val", len(windows), self.sequence_length, self.num_classes, n_train, n_val)

        history = SimpleHistory()
        batch_size = self.cfg["batch_size"]
        epochs = self.cfg["epochs"]

        X_train_t = torch.as_tensor(X_train, dtype=torch.long, device=self.device)
        y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        X_val_t = torch.as_tensor(X_val, dtype=torch.long, device=self.device)
        y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=self.device)

        for epoch in range(epochs):
            self.model.train()
            epoch_rng = np.random.RandomState(seed + epoch)
            perm_t = torch.as_tensor(epoch_rng.permutation(n_train), device=self.device)
            running_loss, running_correct = 0.0, 0
            for start in range(0, n_train, batch_size):
                idx = perm_t[start:start + batch_size]
                xb, yb = X_train_t[idx], y_train_t[idx]
                sw = weight_tensor[yb] if weight_tensor is not None else None

                self._optimizer.zero_grad()
                logits = self.model(xb)
                loss = self._loss_fn(logits, yb, sw)
                loss.backward()
                self._optimizer.step()

                running_loss += loss.detach().item() * len(xb)
                running_correct += int((logits.argmax(dim=-1) == yb).sum())

            train_loss = running_loss / max(n_train, 1)
            train_acc = running_correct / max(n_train, 1)

            self.model.eval()
            with torch.inference_mode():
                val_logits = self._batched_forward(X_val_t, batch_size)
                val_loss = self._loss_fn(val_logits, y_val_t, None).item() if n_val > 0 else float("nan")
                val_acc = float((val_logits.argmax(dim=-1) == y_val_t).float().mean()) if n_val > 0 else float("nan")

            history.record(train_loss, train_acc, val_loss, val_acc)
            logger.info("Epoch %d/%d - loss: %.4f - accuracy: %.4f - val_loss: %.4f - val_accuracy: %.4f",
                        epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc)

        return history

    def _batched_forward(self, X: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Runs self.model over X in chunks (no_grad context is the caller's responsibility)."""
        outs = []
        for start in range(0, len(X), batch_size):
            outs.append(self.model(X[start:start + batch_size]))
        return torch.cat(outs, dim=0) if outs else torch.empty(0, self.num_classes, device=self.device)

    def predict_next_events(self, label_window: np.ndarray) -> np.ndarray:
        """
        label_window: (n_nodes, seq_len) integer event-label sequences ->
        returns the argmax predicted class id per node, shape (n_nodes,).
        This is exactly the SMDP network state S_t (Section III-C-1).

        Runs under torch.inference_mode() rather than plain eager
        autograd-tracked mode -- this is called once per RL environment
        step (millions of times over a full run), the same hot path the
        Keras version's tf.function compilation targeted. See this
        module's docstring for the honest caveat on relative throughput.
        """
        self.model.eval()
        x = torch.as_tensor(label_window, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(x)
            return logits.argmax(dim=-1).cpu().numpy()

    def predict_proba(self, label_window: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.as_tensor(label_window, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(x)
            return F.softmax(logits, dim=-1).cpu().numpy()

    def predict_labels(self, event_labels: np.ndarray,
                        target_labels: Optional[np.ndarray] = None) -> tuple:
        """
        Returns (y_true, y_pred) over sliding windows -- the raw material
        for a confusion matrix (paper's Fig. 8) or any metric beyond the
        single fidelity scalar in compute_fidelity().
        """
        windows, targets = self.build_sliding_windows(event_labels, target_labels)
        self.model.eval()
        x = torch.as_tensor(windows, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self._batched_forward(x, self.cfg["batch_size"])
            preds = logits.argmax(dim=-1).cpu().numpy()
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

class _TransformerBlock(nn.Module):
    """One [MultiHeadAttention -> Add&Norm -> position-wise FFN -> Add&Norm] block."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads,
                                           dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        attn_out, attn_weights = self.attn(x, x, x, need_weights=need_weights, average_attn_weights=False)
        x = self.attn_norm(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)
        return x, attn_weights


class _TransformerNet(nn.Module):
    """
    Token Embedding(num_classes, d_model) + learned positional embedding
    (nn.Parameter, analogous to the Keras version's custom
    PositionalEmbedding layer -- a plain nn.Parameter needs none of that
    layer's serialization workaround, since PyTorch's state_dict
    save/load has no equivalent to Keras's functional-graph-tracing
    quirk that motivated it) -> N x _TransformerBlock -> last position's
    contextualized representation -> Dense -> raw logits.
    """

    def __init__(self, num_classes: int, sequence_length: int, d_model: int, num_heads: int,
                 num_blocks: int, d_ff: int, dense_units: list, dropout: float):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_blocks = num_blocks
        self.embedding = nn.Embedding(num_classes, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(sequence_length, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_blocks)
        ])

        dense_layers = []
        prev = d_model
        for units in dense_units:
            dense_layers += [nn.Linear(prev, units), nn.ReLU(), nn.Dropout(dropout)]
            prev = units
        self.dense = nn.Sequential(*dense_layers)
        self.out = nn.Linear(prev, num_classes)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) + self.pos_embedding.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._embed(x)
        for block in self.blocks:
            h, _ = block(h, need_weights=False)
        h_last = h[:, -1, :]  # last position's contextualized representation (LSTM final-hidden-state analogue)
        h_last = self.dense(h_last)
        return self.out(h_last)

    def forward_with_attention(self, x: torch.Tensor, target_block: int) -> torch.Tensor:
        """Runs blocks [0, target_block] and returns THAT block's attention weights."""
        h = self._embed(x)
        for i, block in enumerate(self.blocks):
            need_weights = (i == target_block)
            h, attn_weights = block(h, need_weights=need_weights)
            if need_weights:
                return attn_weights
        raise ValueError(f"target_block={target_block} out of range for {self.num_blocks} blocks")


class TransformerAttackPredictor:
    """
    Drop-in alternative to LSTMAttackPredictor -- same interface
    (fit/predict_next_events/predict_proba/predict_labels/compute_fidelity),
    same input (Stage 1's event-label sequences), different sequence model.
    See the methodology report for full justification; summary, INCLUDING
    an honest negative result from scripts/compare_architectures.py:

      - Predictive fidelity and class-fairness are statistically tied with
        the LSTM on matched synthetic validation data -- this is a "no
        regression" result, not a predictive win.
      - The Transformer is NOT faster: self-attention's O(L^2) cost
        across multiple blocks/heads outweighs recurrence's
        sequential-but-O(L) cost at this short a sequence length. The
        parallelization advantage this architecture is known for is real
        but needs longer sequences and/or GPU execution to manifest.
      - The genuine, scale-independent advantage is interpretability:
        self-attention gives a directly inspectable weight over "which
        past events mattered for this prediction" (see
        get_attention_weights() / metrics.py::attention_entropy) that an
        LSTM's opaque final hidden state does not offer without a separate
        post-hoc method (e.g. SHAP/LIME) bolted on.
    """

    def __init__(self, num_classes: int, cfg: dict, device: Optional[torch.device] = None):
        self.num_classes = num_classes
        self.cfg = cfg
        self.sequence_length = cfg["sequence_length"]
        self.d_model = cfg.get("d_model", cfg.get("embedding_dim", 32))
        self.num_heads = cfg.get("num_heads", 4)
        self.num_blocks = cfg.get("num_transformer_blocks", 2)
        self.d_ff = cfg.get("d_ff", self.d_model * 4)
        self.device = device or torch.device("cpu")
        self.model = _TransformerNet(
            num_classes=num_classes, sequence_length=self.sequence_length, d_model=self.d_model,
            num_heads=self.num_heads, num_blocks=self.num_blocks, d_ff=self.d_ff,
            dense_units=cfg["dense_units"], dropout=cfg["dropout"],
        ).to(self.device)
        self._loss_fn = _build_loss_fn(cfg.get("loss_function", "categorical_crossentropy"),
                                        cfg.get("focal_gamma", 2.0))
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"])

    # -- Same windowing/fit/predict interface as LSTMAttackPredictor -----

    def build_sliding_windows(self, event_labels: np.ndarray,
                               target_labels: Optional[np.ndarray] = None) -> tuple:
        if target_labels is None:
            target_labels = event_labels
        L = self.sequence_length
        n = len(event_labels) - L
        if n <= 0:
            raise ValueError(
                f"Only {len(event_labels)} rows, fewer than sequence_length={L}; "
                f"cannot build any training windows."
            )
        windows = np.stack([event_labels[i:i + L] for i in range(n)], axis=0).astype(np.int64)
        targets = target_labels[L:L + n]
        return windows, targets

    def fit(self, event_labels: np.ndarray, target_labels: Optional[np.ndarray] = None,
            class_weight: Optional[dict] = None, seed: int = 42) -> SimpleHistory:
        windows, targets = self.build_sliding_windows(event_labels, target_labels)

        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(windows))
        windows, targets = windows[perm], targets[perm]

        val_split = self.cfg["validation_split"]
        n_val = int(len(windows) * val_split)
        n_train = len(windows) - n_val
        X_train, y_train = windows[:n_train], targets[:n_train]
        X_val, y_val = windows[n_train:], targets[n_train:]

        weight_tensor = _class_weight_tensor(class_weight, self.num_classes, self.device)

        logger.info("Training Transformer predictor on %d label-sequence windows (seq_len=%d, num_classes=%d), "
                    "%d train / %d val", len(windows), self.sequence_length, self.num_classes, n_train, n_val)

        history = SimpleHistory()
        batch_size = self.cfg["batch_size"]
        epochs = self.cfg["epochs"]

        X_train_t = torch.as_tensor(X_train, dtype=torch.long, device=self.device)
        y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        X_val_t = torch.as_tensor(X_val, dtype=torch.long, device=self.device)
        y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=self.device)

        for epoch in range(epochs):
            self.model.train()
            epoch_rng = np.random.RandomState(seed + epoch)
            perm_t = torch.as_tensor(epoch_rng.permutation(n_train), device=self.device)
            running_loss, running_correct = 0.0, 0
            for start in range(0, n_train, batch_size):
                idx = perm_t[start:start + batch_size]
                xb, yb = X_train_t[idx], y_train_t[idx]
                sw = weight_tensor[yb] if weight_tensor is not None else None

                self._optimizer.zero_grad()
                logits = self.model(xb)
                loss = self._loss_fn(logits, yb, sw)
                loss.backward()
                self._optimizer.step()

                running_loss += loss.detach().item() * len(xb)
                running_correct += int((logits.argmax(dim=-1) == yb).sum())

            train_loss = running_loss / max(n_train, 1)
            train_acc = running_correct / max(n_train, 1)

            self.model.eval()
            with torch.inference_mode():
                val_logits = self._batched_forward(X_val_t, batch_size)
                val_loss = self._loss_fn(val_logits, y_val_t, None).item() if n_val > 0 else float("nan")
                val_acc = float((val_logits.argmax(dim=-1) == y_val_t).float().mean()) if n_val > 0 else float("nan")

            history.record(train_loss, train_acc, val_loss, val_acc)
            logger.info("Epoch %d/%d - loss: %.4f - accuracy: %.4f - val_loss: %.4f - val_accuracy: %.4f",
                        epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc)

        return history

    def _batched_forward(self, X: torch.Tensor, batch_size: int) -> torch.Tensor:
        outs = []
        for start in range(0, len(X), batch_size):
            outs.append(self.model(X[start:start + batch_size]))
        return torch.cat(outs, dim=0) if outs else torch.empty(0, self.num_classes, device=self.device)

    def predict_next_events(self, label_window: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.as_tensor(label_window, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(x)
            return logits.argmax(dim=-1).cpu().numpy()

    def predict_proba(self, label_window: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.as_tensor(label_window, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(x)
            return F.softmax(logits, dim=-1).cpu().numpy()

    def predict_labels(self, event_labels: np.ndarray,
                        target_labels: Optional[np.ndarray] = None) -> tuple:
        windows, targets = self.build_sliding_windows(event_labels, target_labels)
        self.model.eval()
        x = torch.as_tensor(windows, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self._batched_forward(x, self.cfg["batch_size"])
            preds = logits.argmax(dim=-1).cpu().numpy()
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

        Unlike the Keras version (which had to re-fetch layers via
        model.get_layer() on every call specifically to avoid stale
        references after a checkpoint reload replaced predictor.model),
        this always calls through self.model directly, so there's no
        equivalent staleness risk to guard against -- reassigning
        self.model.load_state_dict(...) mutates the SAME nn.Module
        instance in place rather than swapping in a new object.
        """
        target_idx = block if block >= 0 else self.num_blocks + block
        self.model.eval()
        x = torch.as_tensor(label_window, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            attn_weights = self.model.forward_with_attention(x, target_idx)
        return attn_weights.cpu().numpy()


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


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_layers: list, out_dim: int, activation: str):
        super().__init__()
        act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh}.get(activation)
        if act_cls is None:
            raise ValueError(f"Unknown activation: {activation!r} (expected 'relu' or 'tanh')")
        layers_list = []
        prev = in_dim
        for units in hidden_layers:
            layers_list += [nn.Linear(prev, units), act_cls()]
            prev = units
        layers_list.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """
    Upper-layer agent choosing macro-actions O = {o_c, o_a, o_r, o_s}
    (Section VI-B, Eq. 13-14). Standard DQN with target network and
    replay buffer, per Algorithm 1 lines 1, 5-6, 14-18, 35-36, 42.
    """

    def __init__(self, state_dim: int, num_actions: int, cfg: dict, seed: int = 0,
                 device: Optional[torch.device] = None):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.gamma = cfg["gamma"]
        self.epsilon = cfg["epsilon_start"]
        self.epsilon_end = cfg["epsilon_end"]
        self.epsilon_decay = (cfg["epsilon_start"] - cfg["epsilon_end"]) / max(cfg["epsilon_decay_episodes"], 1)
        self.batch_size = cfg["batch_size"]
        self.min_replay_before_train = cfg["min_replay_before_train"]
        self.target_update_every = cfg["target_update_every"]

        self.replay = ReplayBuffer(cfg["replay_buffer_size"], seed=seed)
        self._rng = np.random.RandomState(seed)

        self.q_network = _MLP(state_dim, cfg["hidden_layers"], num_actions, cfg["activation"]).to(self.device)
        self.target_network = _MLP(state_dim, cfg["hidden_layers"], num_actions, cfg["activation"]).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self._optimizer = torch.optim.Adam(self.q_network.parameters(), lr=cfg["learning_rate"])
        self._train_steps = 0

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy macro-action selection, Algorithm 1 lines 12-18."""
        if not greedy and self._rng.random() <= self.epsilon:
            return int(self._rng.randint(self.num_actions))
        self.q_network.eval()
        state_t = torch.as_tensor(state[None, :], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            q_values = self.q_network(state_t)[0].cpu().numpy()
        return int(np.argmax(q_values))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def store(self, state, action, reward, next_state, done) -> None:
        self.replay.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """One gradient step, Algorithm 1 lines 35-36. Returns the loss, or
        None if there isn't enough replay data yet.
        """
        if len(self.replay) < max(self.batch_size, self.min_replay_before_train):
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            target_q_next = self.target_network(next_states_t)
            max_target_q = target_q_next.max(dim=1).values
        y = rewards_t + (1.0 - dones_t) * self.gamma * max_target_q

        self.q_network.train()
        self._optimizer.zero_grad()
        q_pred = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q_pred, y)
        loss.backward()
        self._optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())  # Algorithm 1 line 42

        return loss.detach().item()


# =============================================================================
# Lower layer: PPO over micro-actions (Section VI-B, Eq. 15-17, Table I)
# =============================================================================

class _PPOActor(nn.Module):
    """Shared trunk, two categorical heads: per-node IP-pool logits, per-flow route logits."""

    def __init__(self, state_dim: int, hidden_layers: list, activation: str,
                 n_nodes: int, num_ip_pools: int, num_flows: int, num_route_candidates: int):
        super().__init__()
        act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh}.get(activation)
        if act_cls is None:
            raise ValueError(f"Unknown activation: {activation!r} (expected 'relu' or 'tanh')")
        trunk_layers = []
        prev = state_dim
        for units in hidden_layers:
            trunk_layers += [nn.Linear(prev, units), act_cls()]
            prev = units
        self.trunk = nn.Sequential(*trunk_layers)
        self.n_nodes, self.num_ip_pools = n_nodes, num_ip_pools
        self.num_flows, self.num_route_candidates = num_flows, num_route_candidates
        self.ip_head = nn.Linear(prev, n_nodes * num_ip_pools)
        self.route_head = nn.Linear(prev, num_flows * num_route_candidates)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        ip_logits = self.ip_head(h).view(-1, self.n_nodes, self.num_ip_pools)
        route_logits = self.route_head(h).view(-1, self.num_flows, self.num_route_candidates)
        return ip_logits, route_logits


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
                 num_flows: int, num_route_candidates: int, cfg: dict, seed: int = 0,
                 device: Optional[torch.device] = None):
        self.state_dim = state_dim
        self.n_nodes = n_nodes
        self.num_ip_pools = num_ip_pools
        self.num_flows = num_flows
        self.num_route_candidates = num_route_candidates
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.gamma = cfg["gamma"]
        self.gae_lambda = cfg["gae_lambda"]
        self.clip_epsilon = cfg["clip_epsilon"]
        self.entropy_coef = cfg["entropy_coef"]
        self.value_coef = cfg["value_coef"]

        self._rng = np.random.RandomState(seed)

        self.actor = _PPOActor(state_dim, cfg["hidden_layers"], cfg["activation"],
                                n_nodes, num_ip_pools, num_flows, num_route_candidates).to(self.device)
        self.critic = _MLP(state_dim, cfg["hidden_layers"], 1, cfg["activation"]).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg["learning_rate"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg["learning_rate"])

    def select_action(self, state: np.ndarray):
        """Returns (action_dict, log_prob, value) for a single state."""
        self.actor.eval()
        self.critic.eval()
        state_batch = torch.as_tensor(state[None, :], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            ip_logits, route_logits = self.actor(state_batch)
            value = float(self.critic(state_batch)[0, 0])
        ip_logits = ip_logits[0].cpu().numpy()        # (n_nodes, num_ip_pools)
        route_logits = route_logits[0].cpu().numpy()  # (num_flows, num_route_candidates)

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
        """One actor + one critic gradient step on a single minibatch."""
        self.actor.train()
        ip_logits, route_logits = self.actor(states)
        ip_log_probs_all = F.log_softmax(ip_logits, dim=-1)
        route_log_probs_all = F.log_softmax(route_logits, dim=-1)

        ip_selected = ip_log_probs_all.gather(2, ip_actions.unsqueeze(-1)).squeeze(-1)
        route_selected = route_log_probs_all.gather(2, route_actions.unsqueeze(-1)).squeeze(-1)
        new_log_probs = ip_selected.sum(dim=1) + route_selected.sum(dim=1)

        ratio = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()

        ip_probs = F.softmax(ip_logits, dim=-1)
        route_probs = F.softmax(route_logits, dim=-1)
        entropy = (
            -(ip_probs * ip_log_probs_all).sum(dim=-1).mean()
            - (route_probs * route_log_probs_all).sum(dim=-1).mean()
        )
        actor_loss = policy_loss - self.entropy_coef * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic.train()
        values_pred = self.critic(states).squeeze(-1)
        critic_loss = self.value_coef * F.mse_loss(values_pred, returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.detach().item(), critic_loss.detach().item()

    def update(self, states, ip_actions, route_actions, old_log_probs, advantages, returns):
        """
        Clipped-surrogate PPO update, Eq. 16-17, Algorithm 1 lines 37-40.
        """
        states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        ip_actions_t = torch.as_tensor(np.asarray(ip_actions), dtype=torch.long, device=self.device)
        route_actions_t = torch.as_tensor(np.asarray(route_actions), dtype=torch.long, device=self.device)
        old_log_probs_t = torch.as_tensor(np.asarray(old_log_probs), dtype=torch.float32, device=self.device)

        advantages = np.asarray(advantages, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(np.asarray(returns), dtype=torch.float32, device=self.device)

        n = states_t.shape[0]
        minibatch_size = min(self.cfg["minibatch_size"], n)
        actor_losses, critic_losses = [], []

        for _ in range(self.cfg["update_epochs"]):
            idx_all = np.arange(n)
            self._rng.shuffle(idx_all)
            for start in range(0, n, minibatch_size):
                mb_idx = torch.as_tensor(idx_all[start:start + minibatch_size], device=self.device)
                actor_loss, critic_loss = self._train_step_impl(
                    states_t[mb_idx], ip_actions_t[mb_idx], route_actions_t[mb_idx],
                    old_log_probs_t[mb_idx], advantages_t[mb_idx], returns_t[mb_idx],
                )
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

        return float(np.mean(actor_losses)), float(np.mean(critic_losses))
