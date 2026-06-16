"""
Temporal Sequence Builder for LSTM Attack Prediction.

Implements the training data construction described in Section VI-A:

    [X | Y] = | E^1_k  E^2_k  ... E^L_k  | E^{L+1}_k |
              | E^2_k  E^3_k  ... E^{L+1}_k | E^{L+2}_k |
              |  ...                         |   ...      |
              | E^{t-L}_k  ...  E^{t-1}_k   | E^t_k     |

Each row is a sliding window of length L (sequence_length) over
the security event sequence for node k. The target Y^t_k = E^t_k
is the predicted next security event (attack type).

In practice (with CICIDS2017), we treat each flow as belonging to
a "network node" (source IP), and construct per-node event sequences.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("cm_mtd.sequence_builder")


class SecurityEventSequenceBuilder:
    """
    Builds sliding-window security event sequences for LSTM.
    
    Args:
        sequence_length: L — look-back window (paper default: 10).
        step_size: Sliding window step size.
        n_classes: Total number of attack classes.
    """

    def __init__(
        self,
        sequence_length: int = 10,
        step_size: int = 1,
        n_classes: int = 8,
    ) -> None:
        self.L = sequence_length
        self.step = step_size
        self.n_classes = n_classes

    # ─── From Raw Labels ────────────────────────────────────────────────────

    def build_from_labels(
        self, y: np.ndarray, group_ids: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build (X, Y) sequence pairs from a flat array of event labels.
        
        If group_ids is provided, sequences are built per unique group
        (e.g., per source IP / node) to avoid mixing different nodes'
        event histories.
        
        Args:
            y: Integer label array of shape [N].
            group_ids: Optional per-sample group identifiers (e.g., node IDs).
        
        Returns:
            X: shape [n_sequences, L] — input sequences (integer event IDs)
            Y: shape [n_sequences]   — target labels (integer event IDs)
        """
        if group_ids is None:
            # Treat all events as a single global sequence
            X_seqs, Y_seqs = self._sliding_window(y)
        else:
            X_list, Y_list = [], []
            unique_groups = np.unique(group_ids)
            for gid in unique_groups:
                mask = group_ids == gid
                group_seq = y[mask]
                if len(group_seq) > self.L:
                    Xg, Yg = self._sliding_window(group_seq)
                    X_list.append(Xg)
                    Y_list.append(Yg)
            if not X_list:
                return np.array([]), np.array([])
            X_seqs = np.concatenate(X_list, axis=0)
            Y_seqs = np.concatenate(Y_list, axis=0)

        logger.info(
            f"Built {len(X_seqs):,} sequences "
            f"(L={self.L}, step={self.step})"
        )
        return X_seqs, Y_seqs

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        label_col: str = "label_id",
        node_col: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build sequences directly from a preprocessed DataFrame.
        
        Args:
            df: DataFrame with integer label column.
            label_col: Name of the integer label column.
            node_col: Optional column identifying network node / source IP.
        
        Returns:
            X, Y sequence arrays.
        """
        y = df[label_col].values.astype(np.int64)
        group_ids = df[node_col].values if node_col and node_col in df.columns else None
        return self.build_from_labels(y, group_ids)

    # ─── Synthetic Sequence Generation ──────────────────────────────────────

    def build_synthetic(
        self,
        n_nodes: int = 12,
        n_time_steps: int = 5000,
        attack_schedule: Optional[List[Dict]] = None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic security event sequences for simulation.
        
        Models realistic attack patterns:
        - Scanning (PortScan) followed by DoS/DDoS
        - Temporal clustering of attacks on same nodes
        - Background benign traffic
        
        Args:
            n_nodes: Number of network nodes.
            n_time_steps: Total time steps to simulate.
            attack_schedule: Optional list of attack phase definitions.
            seed: Random seed.
        
        Returns:
            events: shape [n_nodes, n_time_steps] — per-node event sequences.
            X: shape [n_sequences, L] — LSTM input sequences.
            Y: shape [n_sequences]    — LSTM targets.
        """
        rng = np.random.default_rng(seed)

        # Default attack schedule: scan → DDoS cycles
        if attack_schedule is None:
            attack_schedule = [
                {"start": 0,          "end": 1000, "type": "BENIGN"},
                {"start": 1000,       "end": 2000, "type": "PortScan"},
                {"start": 2000,       "end": 3000, "type": "DoS"},
                {"start": 3000,       "end": 4000, "type": "DDoS"},
                {"start": 4000,       "end": 5000, "type": "BENIGN"},
            ]

        from datasets.cicids2017_loader import ATTACK_CLASS_MAP

        # Build per-node event sequences
        events = np.zeros((n_nodes, n_time_steps), dtype=np.int64)

        for phase in attack_schedule:
            start, end = phase["start"], phase["end"]
            atk_type = phase["type"]
            atk_id = ATTACK_CLASS_MAP.get(atk_type, 0)

            # Randomly select target nodes for each attack phase
            if atk_type == "BENIGN":
                events[:, start:end] = 0  # all nodes benign
            else:
                n_targets = max(1, n_nodes // 2)
                target_nodes = rng.choice(n_nodes, n_targets, replace=False)
                # Spread attack with some randomness
                for t in range(start, min(end, n_time_steps)):
                    for node in target_nodes:
                        if rng.random() < 0.7:  # 70% probability of attack event
                            events[node, t] = atk_id
                        else:
                            events[node, t] = 0  # benign

        # Flatten per-node sequences into global sequence
        all_events = events.flatten()

        # Add some noise (random minor attacks)
        noise_idx = rng.choice(len(all_events), size=len(all_events) // 20, replace=False)
        noise_labels = rng.integers(0, self.n_classes, size=len(noise_idx))
        all_events[noise_idx] = noise_labels

        X, Y = self._sliding_window(all_events)

        logger.info(
            f"Generated {n_nodes} nodes × {n_time_steps} steps → "
            f"{len(X):,} sequences"
        )
        return events, X, Y

    # ─── Train/Val/Test Split ───────────────────────────────────────────────

    def train_val_test_split(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        train_frac: float = 0.70,
        val_frac: float = 0.10,
        test_frac: float = 0.20,
        stratify: bool = True,
        seed: int = 42,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split sequences into train/val/test with stratification.
        
        Paper: 80% train, 20% test (we add a val split here).
        
        Returns:
            Dict {'train': (X, Y), 'val': (X, Y), 'test': (X, Y)}.
        """
        strat = Y if stratify else None

        X_temp, X_test, Y_temp, Y_test = train_test_split(
            X, Y, test_size=test_frac, stratify=strat, random_state=seed
        )
        val_rel = val_frac / (train_frac + val_frac)
        strat2 = Y_temp if stratify else None
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_temp, Y_temp, test_size=val_rel, stratify=strat2, random_state=seed
        )

        splits = {
            "train": (X_train, Y_train),
            "val":   (X_val,   Y_val),
            "test":  (X_test,  Y_test),
        }
        for name, (Xs, Ys) in splits.items():
            dist = np.bincount(Ys, minlength=self.n_classes)
            logger.info(f"  {name.capitalize():5s}: {len(Xs):,} sequences | dist: {dist}")

        return splits

    # ─── Internal Utilities ─────────────────────────────────────────────────

    def _sliding_window(
        self, sequence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply sliding window of length L to a 1D event sequence.
        
        X[i] = sequence[i : i+L]
        Y[i] = sequence[i+L]
        """
        T = len(sequence)
        if T <= self.L:
            return np.array([]).reshape(0, self.L), np.array([])

        n_windows = (T - self.L - 1) // self.step + 1
        X = np.zeros((n_windows, self.L), dtype=np.int64)
        Y = np.zeros(n_windows, dtype=np.int64)

        for i in range(n_windows):
            start = i * self.step
            X[i] = sequence[start: start + self.L]
            Y[i] = sequence[start + self.L]

        return X, Y

    def predict_state_vector(
        self,
        lstm_model,
        recent_events: np.ndarray,
        n_nodes: int = 12,
    ) -> np.ndarray:
        """
        Use LSTM to predict security event probabilities for all nodes.
        
        This generates the SMDP network state S_t = {e^{t+1}_1, ..., e^{t+1}_n}
        (Eq. in Section III-C-1 of paper).
        
        Args:
            lstm_model: Trained LSTM predictor.
            recent_events: Recent event history per node [n_nodes, L].
            n_nodes: Number of network nodes.
        
        Returns:
            State vector of shape [n_nodes, n_classes] with attack probabilities.
        """
        state = np.zeros((n_nodes, self.n_classes), dtype=np.float32)
        for i in range(n_nodes):
            seq = recent_events[i].reshape(1, self.L)
            probs = lstm_model.predict_proba(seq)
            state[i] = probs[0]
        return state


class SyntheticDataGenerator:
    """
    Generates synthetic CICIDS2017-like data when real data is unavailable.
    Used for development, testing, and demonstration purposes.
    """

    def __init__(self, n_classes: int = 8, seed: int = 42) -> None:
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        n_samples: int = 100000,
        class_weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic feature-label pairs.
        
        Args:
            n_samples: Total number of samples.
            class_weights: Class probability weights. Defaults to CICIDS2017-like distribution.
        
        Returns:
            X: Feature matrix [n_samples, 78] (78 = number of CICIDS2017 features).
            y: Label array [n_samples].
        """
        # Approximate CICIDS2017 class distribution
        if class_weights is None:
            class_weights = np.array([
                0.50,   # BENIGN (dominant)
                0.15,   # DoS
                0.12,   # DDoS
                0.10,   # PortScan
                0.01,   # Infiltration
                0.03,   # Bot
                0.05,   # BruteForce
                0.04,   # WebAttack
            ])
            class_weights = class_weights / class_weights.sum()

        # Generate labels
        y = self.rng.choice(self.n_classes, size=n_samples, p=class_weights)

        # Generate per-class feature distributions
        n_features = 78  # matches CICIDS2017
        X = np.zeros((n_samples, n_features), dtype=np.float32)

        for cls in range(self.n_classes):
            mask = y == cls
            n_cls = mask.sum()
            if n_cls == 0:
                continue
            # Each class has distinct statistical signature
            mean = self.rng.uniform(-1, 3, n_features) * (cls + 1)
            std = self.rng.uniform(0.5, 2.0, n_features)
            X[mask] = self.rng.normal(mean, std, (n_cls, n_features))

        # Clip to reasonable range
        X = np.clip(X, -10, 100)

        logger.info(f"Generated {n_samples:,} synthetic samples "
                    f"| Class dist: {np.bincount(y)}")
        return X.astype(np.float32), y

    def generate_event_sequence(
        self,
        n_nodes: int = 12,
        n_steps: int = 10000,
        sequence_length: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic temporal event sequences for LSTM.
        
        Returns:
            X: [n_seqs, sequence_length] event sequences.
            y: [n_seqs] target labels.
        """
        builder = SecurityEventSequenceBuilder(
            sequence_length=sequence_length,
            n_classes=self.n_classes,
        )
        _, X, y = builder.build_synthetic(
            n_nodes=n_nodes,
            n_time_steps=n_steps,
        )
        return X, y
