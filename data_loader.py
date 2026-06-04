"""
data_loader.py
==============
Modular data loader for CM-MTD (Collaborative Mutation Moving Target Defense).

Supports two dataset modes, switchable via the `mode` constructor argument:
  - 'direct_ddos'    : Simulated Direct DDoS + Sequential Scanning
  - 'crossfire_ddos' : Simulated Crossfire DDoS + Sequential Scanning
  - 'cicids2017'     : Real-world CICIDS-2017 dataset (CSV files)

Output contract:
    DataLoader.load() → (X_train, X_test, y_train, y_test)
    where X.shape == (n_samples, sequence_length, n_nodes)  [int64]
          y.shape == (n_samples, n_nodes)                    [int64]
    Labels are SecurityEvent integers (0=BENIGN, 1=INFILTRATION, 2=DOS_DDOS).
"""

from __future__ import annotations

import os
import warnings
from collections import deque
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Security event taxonomy
# ---------------------------------------------------------------------------

class SecurityEvent(IntEnum):
    BENIGN       = 0   # Normal traffic
    INFILTRATION = 1   # Network reconnaissance / IP scanning
    DOS_DDOS     = 2   # Denial-of-Service / Distributed DoS


N_EVENT_TYPES = len(SecurityEvent)

# Map CICIDS-2017 string labels → SecurityEvent integers
CICIDS_LABEL_MAP: Dict[str, int] = {
    "BENIGN"                           : SecurityEvent.BENIGN,
    "Bot"                              : SecurityEvent.INFILTRATION,
    "DDoS"                             : SecurityEvent.DOS_DDOS,
    "DoS GoldenEye"                    : SecurityEvent.DOS_DDOS,
    "DoS Hulk"                         : SecurityEvent.DOS_DDOS,
    "DoS Slowhttptest"                 : SecurityEvent.DOS_DDOS,
    "DoS slowloris"                    : SecurityEvent.DOS_DDOS,
    "FTP-Patator"                      : SecurityEvent.INFILTRATION,
    "Heartbleed"                       : SecurityEvent.INFILTRATION,
    "Infiltration"                     : SecurityEvent.INFILTRATION,
    "PortScan"                         : SecurityEvent.INFILTRATION,
    "SSH-Patator"                      : SecurityEvent.INFILTRATION,
    "Web Attack – Brute Force"         : SecurityEvent.INFILTRATION,
    "Web Attack – Sql Injection"       : SecurityEvent.INFILTRATION,
    "Web Attack – XSS"                 : SecurityEvent.INFILTRATION,
    "Web Attack \x96 Brute Force"      : SecurityEvent.INFILTRATION,
    "Web Attack \x96 Sql Injection"    : SecurityEvent.INFILTRATION,
    "Web Attack \x96 XSS"             : SecurityEvent.INFILTRATION,
}


# ---------------------------------------------------------------------------
# Network topology helper
# ---------------------------------------------------------------------------

class NetworkTopology:
    """
    Lightweight representation of the SDN-IoT network topology in DTMN.
    Nodes represent IoT devices + cloud servers; switches are OpenFlow nodes.
    """

    def __init__(self, n_nodes: int = 12, n_switches: int = 12, seed: int = 42):
        self.n_nodes    = n_nodes
        self.n_switches = n_switches
        rng = np.random.RandomState(seed)

        # Node connectivity degree ~ power-law (high-degree nodes = DDoS targets)
        raw_degrees = rng.zipf(2.0, n_nodes).clip(1, n_switches).astype(float)
        self.node_degrees = raw_degrees / raw_degrees.sum()

        # Random switch adjacency (ring + random cross-links → Waxman-like)
        adj = np.zeros((n_switches, n_switches), dtype=np.int32)
        for i in range(n_switches):
            j = (i + 1) % n_switches
            adj[i, j] = adj[j, i] = 1
        for _ in range(n_switches // 2):
            a, b = rng.choice(n_switches, size=2, replace=False)
            adj[a, b] = adj[b, a] = 1
        self.switch_adjacency = adj

        # Map each node to its access switch
        self.node_switch_map = rng.randint(0, n_switches, size=n_nodes)

    def get_high_degree_nodes(self, top_k: int = 3) -> List[int]:
        """Indices of the top-k highest-degree nodes (primary DDoS targets)."""
        return np.argsort(self.node_degrees)[-top_k:].tolist()


# ---------------------------------------------------------------------------
# Attack simulators
# ---------------------------------------------------------------------------

class AttackSimulator:
    """
    Generates synthetic security-event time series for n_nodes.
    Implements Section II attack strategies from the paper.
    """

    def __init__(self, topology: NetworkTopology, seed: int = 42):
        self.topology = topology
        self.rng = np.random.RandomState(seed)

    # -------- Direct DDoS + Sequential Scanning ----------------------------

    def generate_direct_ddos_sequential_scanning(
        self,
        n_timesteps: int = 5000,
        scan_prob:   float = 0.30,
        ddos_prob:   float = 0.80,
        ddos_len_range: Tuple[int, int] = (40, 80),
        recon_len_range: Tuple[int, int] = (20, 60),
        pause_range:    Tuple[int, int] = (10, 80),
    ) -> np.ndarray:
        """
        Adversary model:
          Phase 1 – Reconnaissance: sequential IP scanning.
          Phase 2 – DDoS execution: flood high-degree nodes.
          Phase 3 – Pause (benign traffic).

        Returns event_array: shape (n_timesteps, n_nodes), dtype int64.
        """
        events = np.zeros((n_timesteps, self.topology.n_nodes), dtype=np.int64)
        high_deg = self.topology.get_high_degree_nodes()
        t = 0
        scan_start = self.rng.randint(0, self.topology.n_nodes)

        while t < n_timesteps:
            # -- Reconnaissance --
            recon_len = self.rng.randint(*recon_len_range)
            for dt in range(recon_len):
                if t >= n_timesteps:
                    break
                idx = (scan_start + dt) % self.topology.n_nodes
                if self.rng.rand() < scan_prob:
                    events[t, idx] = SecurityEvent.INFILTRATION
                t += 1

            # -- DDoS --
            ddos_len = self.rng.randint(*ddos_len_range)
            for dt in range(ddos_len):
                if t >= n_timesteps:
                    break
                for node in high_deg:
                    if self.rng.rand() < ddos_prob:
                        events[t, node] = SecurityEvent.DOS_DDOS
                t += 1

            # -- Pause --
            t += self.rng.randint(*pause_range)
            scan_start = self.rng.randint(0, self.topology.n_nodes)

        return events

    # -------- Crossfire DDoS + Sequential Scanning -------------------------

    def generate_crossfire_ddos_sequential_scanning(
        self,
        n_timesteps: int = 5000,
        scan_prob:   float = 0.30,
        ddos_prob:   float = 0.75,
        ddos_len_range: Tuple[int, int] = (30, 70),
        recon_len_range: Tuple[int, int] = (20, 60),
        pause_range:    Tuple[int, int] = (10, 60),
    ) -> np.ndarray:
        """
        Crossfire DDoS: adversary sends traffic to *neighbor* nodes to throttle
        shared links, indirectly harming the targeted area.
        """
        events = np.zeros((n_timesteps, self.topology.n_nodes), dtype=np.int64)
        high_deg = self.topology.get_high_degree_nodes()
        # Include immediate neighbors (wrap-around)
        neighbors = [(n + 1) % self.topology.n_nodes for n in high_deg]
        attack_nodes = list(set(high_deg + neighbors))
        t = 0
        scan_start = self.rng.randint(0, self.topology.n_nodes)

        while t < n_timesteps:
            # -- Reconnaissance --
            recon_len = self.rng.randint(*recon_len_range)
            for dt in range(recon_len):
                if t >= n_timesteps:
                    break
                idx = (scan_start + dt) % self.topology.n_nodes
                if self.rng.rand() < scan_prob:
                    events[t, idx] = SecurityEvent.INFILTRATION
                t += 1

            # -- Crossfire DDoS --
            ddos_len = self.rng.randint(*ddos_len_range)
            for dt in range(ddos_len):
                if t >= n_timesteps:
                    break
                for node in attack_nodes:
                    if self.rng.rand() < ddos_prob:
                        events[t, node] = SecurityEvent.DOS_DDOS
                t += 1

            t += self.rng.randint(*pause_range)
            scan_start = self.rng.randint(0, self.topology.n_nodes)

        return events

    # -------- CICIDS-2017 synthetic fallback --------------------------------

    def generate_cicids2017_like(self, n_timesteps: int = 5000) -> np.ndarray:
        """
        Synthetic dataset mimicking CICIDS-2017 class distribution:
          ~80% BENIGN, ~15% DOS_DDOS, ~5% INFILTRATION, with temporal bursts.
        Used when the real CSV path is absent.
        """
        events = np.zeros((n_timesteps, self.topology.n_nodes), dtype=np.int64)
        for t in range(n_timesteps):
            for node in range(self.topology.n_nodes):
                r = self.rng.rand()
                if r < 0.80:
                    events[t, node] = SecurityEvent.BENIGN
                elif r < 0.95:
                    events[t, node] = SecurityEvent.DOS_DDOS
                else:
                    events[t, node] = SecurityEvent.INFILTRATION

        # Add temporal correlation bursts (attacks cluster in time)
        n_bursts = n_timesteps // 80
        for _ in range(n_bursts):
            burst_t    = self.rng.randint(0, max(1, n_timesteps - 40))
            atype      = self.rng.choice([SecurityEvent.DOS_DDOS,
                                           SecurityEvent.INFILTRATION])
            burst_nodes = self.rng.choice(
                self.topology.n_nodes,
                size=self.rng.randint(1, 4),
                replace=False,
            )
            length = self.rng.randint(5, 35)
            for dt in range(length):
                if burst_t + dt < n_timesteps:
                    for n in burst_nodes:
                        events[burst_t + dt, n] = atype
        return events


# ---------------------------------------------------------------------------
# Main DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Unified data loader for CM-MTD experiments.

    Parameters
    ----------
    mode : str
        One of 'direct_ddos', 'crossfire_ddos', 'cicids2017'.
    n_nodes : int
        Number of network nodes (IoT devices + cloud servers).
    n_switches : int
        Number of OpenFlow switches.
    sequence_length : int
        Look-back window L (Eq. 12 of the paper).
    train_ratio : float
        Fraction of data used for training (paper uses 80%).
    n_timesteps : int
        Total simulated time steps (simulation modes only).
    cicids_path : str or None
        Directory containing CICIDS-2017 CSV files. If None or missing,
        a synthetic CICIDS-like dataset is generated automatically.
    seed : int
        Global random seed for reproducibility.
    """

    def __init__(
        self,
        mode:            str  = "direct_ddos",
        n_nodes:         int  = 12,
        n_switches:      int  = 12,
        sequence_length: int  = 10,
        train_ratio:     float = 0.80,
        n_timesteps:     int  = 5000,
        cicids_path:     Optional[str] = None,
        seed:            int  = 42,
    ):
        assert mode in ("direct_ddos", "crossfire_ddos", "cicids2017"), (
            f"mode must be 'direct_ddos', 'crossfire_ddos', or 'cicids2017', got '{mode}'"
        )
        self.mode            = mode
        self.n_nodes         = n_nodes
        self.n_switches      = n_switches
        self.sequence_length = sequence_length
        self.train_ratio     = train_ratio
        self.n_timesteps     = n_timesteps
        self.cicids_path     = cicids_path
        self.seed            = seed

        self.topology  = NetworkTopology(n_nodes, n_switches, seed)
        self.simulator = AttackSimulator(self.topology, seed)
        self.raw_events: Optional[np.ndarray] = None  # set after load()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the dataset for the configured mode.

        Returns
        -------
        X_train, X_test : np.ndarray, shape (n_samples, seq_len, n_nodes), int64
        y_train, y_test : np.ndarray, shape (n_samples, n_nodes),           int64
        """
        if self.mode == "direct_ddos":
            self.raw_events = self.simulator.generate_direct_ddos_sequential_scanning(
                n_timesteps=self.n_timesteps
            )
        elif self.mode == "crossfire_ddos":
            self.raw_events = self.simulator.generate_crossfire_ddos_sequential_scanning(
                n_timesteps=self.n_timesteps
            )
        else:  # cicids2017
            self.raw_events = self._load_cicids2017()

        X, y = self._create_sequences(self.raw_events)

        split = int(len(X) * self.train_ratio)
        return X[:split], X[split:], y[:split], y[split:]

    def get_raw_events(self) -> np.ndarray:
        """Return the full raw event array (n_timesteps, n_nodes)."""
        if self.raw_events is None:
            raise RuntimeError("Call load() before accessing raw_events.")
        return self.raw_events

    def get_topology(self) -> NetworkTopology:
        return self.topology

    def get_event_distribution(self) -> Dict[str, float]:
        """Class-frequency statistics over raw_events."""
        if self.raw_events is None:
            raise RuntimeError("Call load() first.")
        total = self.raw_events.size
        return {
            "BENIGN"      : float(np.sum(self.raw_events == SecurityEvent.BENIGN)) / total,
            "INFILTRATION": float(np.sum(self.raw_events == SecurityEvent.INFILTRATION)) / total,
            "DOS_DDOS"    : float(np.sum(self.raw_events == SecurityEvent.DOS_DDOS)) / total,
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _create_sequences(
        self, events: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sliding-window sequence construction as in Eq. (12).
        X[i] = events[i : i+L],  y[i] = events[i+L]
        """
        T, N = events.shape
        n_samples = T - self.sequence_length
        X = np.empty((n_samples, self.sequence_length, N), dtype=np.int64)
        y = np.empty((n_samples, N), dtype=np.int64)
        for i in range(n_samples):
            X[i] = events[i : i + self.sequence_length]
            y[i] = events[i + self.sequence_length]
        return X, y

    def _load_cicids2017(self) -> np.ndarray:
        """
        Load CICIDS-2017 from CSV files and aggregate into (T, n_nodes) array.
        Falls back to synthetic CICIDS-like data if path is unavailable.
        """
        if not self.cicids_path or not os.path.isdir(self.cicids_path):
            print("[DataLoader] CICIDS-2017 path not found — using synthetic fallback.")
            return self.simulator.generate_cicids2017_like(self.n_timesteps)

        csv_files = sorted(
            f for f in os.listdir(self.cicids_path) if f.endswith(".csv")
        )
        if not csv_files:
            print("[DataLoader] No CSV files in CICIDS path — using synthetic fallback.")
            return self.simulator.generate_cicids2017_like(self.n_timesteps)

        frames: List[pd.DataFrame] = []
        for fname in csv_files:
            try:
                df = pd.read_csv(
                    os.path.join(self.cicids_path, fname), low_memory=False
                )
                df.columns = df.columns.str.strip()
                frames.append(df)
                print(f"  [DataLoader] Loaded {fname}: {len(df):,} rows")
            except Exception as exc:
                print(f"  [DataLoader] Skipping {fname}: {exc}")

        if not frames:
            return self.simulator.generate_cicids2017_like(self.n_timesteps)

        data = pd.concat(frames, ignore_index=True)

        # Resolve label column (CICIDS CSVs sometimes have a leading space)
        label_col = next(
            (c for c in data.columns if c.strip() == "Label"), None
        )
        if label_col is None:
            print("[DataLoader] Label column not found — using synthetic fallback.")
            return self.simulator.generate_cicids2017_like(self.n_timesteps)

        data["event_class"] = (
            data[label_col]
            .str.strip()
            .map(lambda x: CICIDS_LABEL_MAP.get(x, SecurityEvent.BENIGN))
            .fillna(SecurityEvent.BENIGN)
            .astype(int)
        )

        # Assign flows to nodes by destination port modulo n_nodes
        port_col = next(
            (c for c in data.columns if "Destination Port" in c), None
        )
        if port_col:
            data["node_id"] = (
                data[port_col].fillna(0).astype(int) % self.n_nodes
            )
        else:
            rng = np.random.RandomState(self.seed)
            data["node_id"] = rng.randint(0, self.n_nodes, len(data))

        # Aggregate into time windows
        win = max(1, len(data) // self.n_timesteps)
        events = np.zeros((self.n_timesteps, self.n_nodes), dtype=np.int64)
        for t in range(self.n_timesteps):
            s, e = t * win, min((t + 1) * win, len(data))
            if s >= len(data):
                break
            window = data.iloc[s:e]
            for nid in range(self.n_nodes):
                node_rows = window[window["node_id"] == nid]
                events[t, nid] = (
                    int(node_rows["event_class"].max())
                    if len(node_rows) > 0
                    else SecurityEvent.BENIGN
                )
        return events


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for mode in ("direct_ddos", "crossfire_ddos", "cicids2017"):
        dl = DataLoader(mode=mode, n_nodes=12, n_timesteps=1000, seed=0)
        Xtr, Xte, ytr, yte = dl.load()
        dist = dl.get_event_distribution()
        print(
            f"[{mode}]  train={Xtr.shape}  test={Xte.shape}  "
            f"dist={dist}"
        )
