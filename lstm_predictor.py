"""
lstm_predictor.py
=================
PyTorch implementation of the LSTMNet attack predictor described in
Section VI-A of the paper.

Architecture (Fig. 4):
  Preprocess layer  →  Event Embedding layer  →  LSTM layer  →  Softmax

Key design choices matching the paper:
  - Shared LSTM trunk; per-node linear output heads.
  - Loss: CrossEntropy (equivalent to MSE on one-hot in the paper's framing).
  - Optimizer: Adam.
  - Fidelity metric: Eq. (18).

All training metrics are written to CSV; NO matplotlib calls here.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class EventEmbedding(nn.Module):
    """
    Maps discrete security-event IDs to dense vectors (event embedding layer).
    Input  : (batch, seq_len, n_nodes)  — LongTensor of event class IDs.
    Output : (batch, seq_len, n_nodes × embed_dim) — Float context vectors.
    """

    def __init__(self, n_event_types: int = 3, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_event_types, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N)  →  embedded: (B, T, N, D)
        emb = self.embedding(x)
        B, T, N, D = emb.shape
        return emb.view(B, T, N * D)          # (B, T, N*D)


class LSTMNet(nn.Module):
    """
    Full LSTMNet: Embedding → LSTM → per-node softmax heads.

    Parameters
    ----------
    n_nodes       : number of network nodes (IoT + cloud)
    n_event_types : number of security event classes (default 3)
    embed_dim     : dimension of the event embedding
    hidden_size   : LSTM hidden state dimension
    n_lstm_layers : number of stacked LSTM layers
    dropout       : dropout between LSTM layers (if n_lstm_layers > 1)
    """

    def __init__(
        self,
        n_nodes:       int = 12,
        n_event_types: int = 3,
        embed_dim:     int = 16,
        hidden_size:   int = 128,
        n_lstm_layers: int = 2,
        dropout:       float = 0.2,
    ):
        super().__init__()
        self.n_nodes       = n_nodes
        self.n_event_types = n_event_types

        self.event_embedding = EventEmbedding(n_event_types, embed_dim)

        lstm_input_size = n_nodes * embed_dim
        self.lstm = nn.LSTM(
            input_size  = lstm_input_size,
            hidden_size = hidden_size,
            num_layers  = n_lstm_layers,
            batch_first = True,
            dropout     = dropout if n_lstm_layers > 1 else 0.0,
        )

        # One linear head per node  →  avoids parameter explosion vs. a single
        # big linear layer (n_nodes × n_event_types outputs)
        self.output_heads = nn.ModuleList(
            [nn.Linear(hidden_size, n_event_types) for _ in range(n_nodes)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : LongTensor, shape (B, seq_len, n_nodes)

        Returns
        -------
        logits : FloatTensor, shape (B, n_nodes, n_event_types)
        """
        embedded  = self.event_embedding(x)         # (B, T, N*D)
        lstm_out, _ = self.lstm(embedded)            # (B, T, H)
        last_h    = lstm_out[:, -1, :]              # (B, H)
        logits    = torch.stack(
            [head(last_h) for head in self.output_heads], dim=1
        )                                            # (B, N, C)
        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Argmax prediction. x: (B, T, N) → (B, N) int64 predicted classes."""
        return self.forward(x).argmax(dim=-1)


# ---------------------------------------------------------------------------
# Training / evaluation wrapper
# ---------------------------------------------------------------------------

class LSTMPredictor:
    """
    High-level wrapper that owns a LSTMNet, its optimizer, and CSV logging.

    Log file written to <log_dir>/lstm_training_log.csv with columns:
        episode, phase, loss, fidelity, timestamp
    """

    def __init__(
        self,
        n_nodes:       int   = 12,
        n_event_types: int   = 3,
        embed_dim:     int   = 16,
        hidden_size:   int   = 128,
        n_lstm_layers: int   = 2,
        dropout:       float = 0.20,
        learning_rate: float = 1e-3,
        device:        str   = "cpu",
        log_dir:       str   = "./logs",
    ):
        self.n_nodes       = n_nodes
        self.n_event_types = n_event_types
        self.log_dir       = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.device = torch.device(
            device if torch.cuda.is_available() and device != "cpu" else "cpu"
        )

        self.model = LSTMNet(
            n_nodes       = n_nodes,
            n_event_types = n_event_types,
            embed_dim     = embed_dim,
            hidden_size   = hidden_size,
            n_lstm_layers = n_lstm_layers,
            dropout       = dropout,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

        self.log_path = os.path.join(log_dir, "lstm_training_log.csv")
        self._init_csv(self.log_path, ["episode", "phase", "loss", "fidelity", "timestamp"])

    # -----------------------------------------------------------------------
    # CSV helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _init_csv(path: str, header: List[str]):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    def _append_csv(self, path: str, row: list):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _log_metrics(self, episode: int, phase: str, loss: float, fidelity: float):
        self._append_csv(
            self.log_path,
            [episode, phase, f"{loss:.6f}", f"{fidelity:.6f}",
             datetime.utcnow().isoformat()],
        )

    # -----------------------------------------------------------------------
    # Fidelity — Eq. (18)
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_fidelity(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Fraction of (node, sample) pairs correctly predicted."""
        return float((predictions == targets).sum()) / max(targets.size, 1)

    # -----------------------------------------------------------------------
    # Confusion matrix (flat over all nodes)
    # -----------------------------------------------------------------------

    def compute_confusion_matrix(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """
        Compute (n_event_types × n_event_types) confusion matrix,
        flattened over all nodes and samples.
        Row = true class, Column = predicted class.
        """
        cm = np.zeros((self.n_event_types, self.n_event_types), dtype=np.int64)
        flat_p = predictions.flatten().astype(int)
        flat_t = targets.flatten().astype(int)
        for t_cls, p_cls in zip(flat_t, flat_p):
            if 0 <= t_cls < self.n_event_types and 0 <= p_cls < self.n_event_types:
                cm[t_cls, p_cls] += 1
        return cm

    # -----------------------------------------------------------------------
    # Per-class precision / recall / F1
    # -----------------------------------------------------------------------

    @staticmethod
    def class_metrics(cm: np.ndarray) -> Dict[str, List[float]]:
        """Derive per-class precision, recall, F1 from confusion matrix."""
        n = cm.shape[0]
        precision, recall, f1 = [], [], []
        for c in range(n):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1_c = 2 * prec * rec / max(prec + rec, 1e-9)
            precision.append(float(prec))
            recall.append(float(rec))
            f1.append(float(f1_c))
        return {"precision": precision, "recall": recall, "f1": f1}

    # -----------------------------------------------------------------------
    # Single epoch
    # -----------------------------------------------------------------------

    def _train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
    ) -> Tuple[float, float]:
        self.model.train()
        n = len(X)
        indices = np.random.permutation(n)
        total_loss = 0.0
        all_preds: List[np.ndarray] = []

        for start in range(0, n, batch_size):
            idx    = indices[start : start + batch_size]
            Xb     = torch.from_numpy(X[idx]).long().to(self.device)    # (B,T,N)
            yb     = torch.from_numpy(y[idx]).long().to(self.device)    # (B,N)
            logits = self.model(Xb)                                      # (B,N,C)

            # Average cross-entropy over all nodes
            loss = sum(
                self.criterion(logits[:, node, :], yb[:, node])
                for node in range(self.n_nodes)
            ) / self.n_nodes

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(idx)
            all_preds.append(logits.argmax(-1).cpu().numpy())  # (B,N)

        preds   = np.vstack(all_preds)
        targets = y[indices[:len(preds)]]
        return total_loss / n, self.compute_fidelity(preds, targets)

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128,
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Returns (loss, fidelity, all_predictions, all_targets).
        """
        self.model.eval()
        n = len(X)
        total_loss = 0.0
        all_preds: List[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, n, batch_size):
                Xb     = torch.from_numpy(X[start:start+batch_size]).long().to(self.device)
                yb     = torch.from_numpy(y[start:start+batch_size]).long().to(self.device)
                logits = self.model(Xb)
                loss   = sum(
                    self.criterion(logits[:, node, :], yb[:, node])
                    for node in range(self.n_nodes)
                ) / self.n_nodes
                total_loss += loss.item() * Xb.size(0)
                all_preds.append(logits.argmax(-1).cpu().numpy())

        preds   = np.vstack(all_preds)
        targets = y[:len(preds)]
        return total_loss / n, self.compute_fidelity(preds, targets), preds, targets

    # -----------------------------------------------------------------------
    # Full training loop
    # -----------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        y_train: np.ndarray,
        y_test:  np.ndarray,
        n_episodes: int = 20,
        batch_size: int = 64,
        verbose:    bool = True,
    ) -> None:
        """
        Train for n_episodes epochs and log every episode to CSV.
        """
        print(f"[LSTMPredictor] Training for {n_episodes} episodes "
              f"on device={self.device}...")

        for ep in range(1, n_episodes + 1):
            tr_loss, tr_fid = self._train_epoch(X_train, y_train, batch_size)
            te_loss, te_fid, _, _ = self.evaluate(X_test, y_test, batch_size)

            self._log_metrics(ep, "train", tr_loss, tr_fid)
            self._log_metrics(ep, "test",  te_loss, te_fid)

            if verbose and (ep % 5 == 0 or ep == 1):
                print(
                    f"  Ep {ep:3d}/{n_episodes}  "
                    f"train_loss={tr_loss:.4f}  train_fid={tr_fid:.4f}  "
                    f"test_loss={te_loss:.4f}  test_fid={te_fid:.4f}"
                )

        print(f"[LSTMPredictor] Done. Log → {self.log_path}")

    # -----------------------------------------------------------------------
    # Online inference (used during HDRL loop)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def predict_state(self, recent_events: np.ndarray) -> np.ndarray:
        """
        Predict next-step security events for all nodes.

        Parameters
        ----------
        recent_events : np.ndarray, shape (seq_len, n_nodes), int64

        Returns
        -------
        predicted_events : np.ndarray, shape (n_nodes,), int64
        """
        self.model.eval()
        x = torch.from_numpy(recent_events).long().unsqueeze(0).to(self.device)
        # shape: (1, seq_len, n_nodes)
        logits = self.model(x)                 # (1, n_nodes, n_event_types)
        return logits.argmax(-1).squeeze(0).cpu().numpy()   # (n_nodes,)

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[LSTMPredictor] Model saved → {path}")

    def load(self, path: str):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        print(f"[LSTMPredictor] Model loaded ← {path}")


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import DataLoader

    dl = DataLoader(mode="direct_ddos", n_nodes=12, n_timesteps=800, seed=0)
    Xtr, Xte, ytr, yte = dl.load()

    lp = LSTMPredictor(n_nodes=12, log_dir="/tmp/cm_mtd_logs")
    lp.fit(Xtr, Xte, ytr, yte, n_episodes=3, batch_size=32)

    _, _, preds, targets = lp.evaluate(Xte, yte)
    cm = lp.compute_confusion_matrix(preds, targets)
    print("Confusion matrix:\n", cm)
