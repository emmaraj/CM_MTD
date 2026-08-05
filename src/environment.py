"""
environment.py
---------------
Custom Gymnasium environment simulating the SDN-IoT Digital Twin Mobile
Network (DTMN) from Zhang et al. (IEEE JSAC 2023), Sections III & V.

Design notes (read before modifying reward logic):

* The environment NEVER calls np.random inside step(). Which row of the
  empirical CICIDS-2017 trace is "observed" by each node at each timestep
  is a deterministic function of the timestep and node index (round-robin
  over the dataset). This satisfies the "strictly step through empirical
  data traces, no synthetic/randomized states" requirement.

* "Attacker targeting" (which IP pool / route the adversary is aiming at)
  is derived deterministically from each sample's own feature vector via a
  hash, rather than sampled randomly. This keeps attack outcomes tied to
  real data while still giving the RL agent something non-trivial to learn
  (matching a static IP/route to the hash is exactly the reconnaissance
  problem HAM/RM are designed to disrupt).

* Network topology (Waxman graph) IS generated with a seeded RNG at env
  construction time. That is world-building, not per-step "state
  simulation" — it happens once, like generating a fixed test network.

* Full SMT-based constraint solving (Section V) is out of scope for this
  research-grade sandbox (see project's clarifying questions / README).
  Feasibility of Eq. 4-7 is instead guaranteed *by construction*: routes
  are drawn from precomputed simple paths between each flow's fixed
  source/destination switch, and IP pool indices are bounded categorical
  choices. This is a deliberate simplification, not a paper ambiguity.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger("cm_mtd")

# Macro-action indices, matching paper's O = {o_c, o_a, o_r, o_s}
MACRO_HAM_RM = 0     # o_c: deploy both HAM and RM
MACRO_HAM_ONLY = 1   # o_a: deploy HAM only
MACRO_RM_ONLY = 2    # o_r: deploy RM only
MACRO_STATIC = 3     # o_s: static IP addresses and routes
NUM_MACRO_ACTIONS = 4

_MACRO_TO_FLAGS = {
    MACRO_HAM_RM: (True, True),
    MACRO_HAM_ONLY: (True, False),
    MACRO_RM_ONLY: (False, True),
    MACRO_STATIC: (False, False),
}


def _deterministic_target_index(feature_vector: np.ndarray, modulus: int) -> int:
    """
    Deterministically derive an "attacker-targeted" index (an IP pool or a
    route/switch index) from a real sample's feature vector. Using a hash
    of the actual feature values means the target is fully determined by
    the empirical data row, not by any random draw.
    """
    # Quantize to stabilize the hash against floating point noise, then hash.
    quantized = np.round(feature_vector, 3).tobytes()
    digest = hashlib.md5(quantized).hexdigest()
    return int(digest, 16) % max(modulus, 1)


def build_waxman_topology(num_nodes: int, alpha: float, beta: float, seed: int) -> nx.Graph:
    """
    Build the SDN-IoT topology as a Waxman random graph, matching the
    paper's simulation setup (Section VII, Table I).
    Graph construction uses a seeded RNG — this is one-time world-building,
    not part of the per-step empirical-data stepping logic.
    """
    graph = nx.waxman_graph(num_nodes, alpha=alpha, beta=beta, seed=seed)
    # Waxman graphs can be disconnected; stitch any isolated components
    # together so every node has at least one route to every other node.
    components = list(nx.connected_components(graph))
    rng = np.random.RandomState(seed)
    for i in range(1, len(components)):
        a = rng.choice(list(components[i - 1]))
        b = rng.choice(list(components[i]))
        graph.add_edge(a, b)
    return graph


def build_flow_routes(graph: nx.Graph, num_flows: int, num_candidates: int, seed: int) -> list[list[list[int]]]:
    """
    For each of `num_flows` flows, pick a fixed (source, destination) pair
    of switches and precompute up to `num_candidates` simple paths between
    them. Route mutation (RM) then just selects among these precomputed,
    already-valid (reachability-satisfying, Eq. 6-7) candidate paths.
    """
    rng = np.random.RandomState(seed)
    nodes = list(graph.nodes())
    routes: list[list[list[int]]] = []

    attempts = 0
    while len(routes) < num_flows and attempts < num_flows * 50:
        attempts += 1
        src, dst = rng.choice(nodes, size=2, replace=False)
        if not nx.has_path(graph, src, dst):
            continue
        try:
            candidates = []
            for path in nx.shortest_simple_paths(graph, src, dst):
                candidates.append(path)
                if len(candidates) >= num_candidates:
                    break
            if candidates:
                routes.append(candidates)
        except nx.NetworkXNoPath:
            continue

    if len(routes) < num_flows:
        raise RuntimeError(
            f"Could only construct {len(routes)}/{num_flows} flow routes on this "
            f"topology; increase num_nodes or reduce num_flows in config.yaml."
        )
    return routes


@dataclass
class EpisodeStats:
    """Cumulative counters needed for the Defense Success Ratio, Eq. 19."""
    scanned_nodes_total: int = 0          # L^s_k: total scan attempts observed
    scan_successes_total: int = 0         # N^s_k: scans that compromised a node
    route_exposures_total: int = 0        # L^d_k: total switches on transmission routes
    switch_compromises_total: int = 0     # N^d_k: switches compromised by DDoS

    def dsr(self) -> float:
        """Defense Success Ratio, Eq. 19."""
        denom = self.scanned_nodes_total + self.route_exposures_total
        if denom == 0:
            return 1.0
        numer = self.scan_successes_total + self.switch_compromises_total
        return 1.0 - (numer / denom)

    def reset(self) -> None:
        self.scanned_nodes_total = 0
        self.scan_successes_total = 0
        self.route_exposures_total = 0
        self.switch_compromises_total = 0


class DigitalTwinNetworkEnv(gym.Env):
    """
    Raw environment stepping strictly through the empirical dataset trace.

    Observation (returned to the agent that ultimately consumes it via the
    LSTMStatePredictionWrapper below) is the *ground-truth* security event
    id per node at t+1 -- i.e. exactly the target the LSTM predictor learns
    to forecast. Reward follows Eq. 1-3.
    """

    metadata = {"render_modes": []}

    def __init__(self, dataset, network_cfg: dict, reward_cfg: dict, data_cfg: dict, seed: int = 42):
        super().__init__()
        self.dataset = dataset

        # This environment's own bounded, row-cycling view of the training
        # data -- built here (not in config_parser.load_dataset) so that
        # LSTM training elsewhere always sees the full, genuinely-
        # contiguous dataset. See config_parser.build_env_row_cache's
        # docstring for why sharing one truncated/reordered view between
        # both consumers was a bug.
        from src.config_parser import build_env_row_cache
        self._env_X_train, self._env_y_train = build_env_row_cache(dataset, data_cfg)

        self.n_nodes = network_cfg["num_nodes"]
        self.num_ip_pools = network_cfg["num_ip_pools"]
        self.num_flows = network_cfg["num_flows"]
        self.num_switches = network_cfg["num_openflow_switches"]
        self.reward_cfg = reward_cfg
        self._seed = seed

        self.graph = build_waxman_topology(
            self.n_nodes, network_cfg["waxman_alpha"], network_cfg["waxman_beta"], seed
        )
        self.flow_routes = build_flow_routes(
            self.graph, self.num_flows, num_candidates=3, seed=seed
        )
        # Each flow's destination endpoint maps to one of our n_nodes for
        # the purpose of "which node is this DDoS event targeting".
        self._flow_dest_node = [i % self.n_nodes for i in range(self.num_flows)]

        try:
            self._infiltration_class = dataset.class_names.index("Infiltration")
        except ValueError:
            raise RuntimeError(
                "environment.py requires a class literally named 'Infiltration' in "
                "config.data.class_names to know which security-event label HAM "
                f"defends against, but it's not there. Current class_names: "
                f"{dataset.class_names} (num_classes inferred from data: {dataset.num_classes}).\n\n"
                "This almost always means config.data.class_names doesn't match the "
                "actual number of distinct labels in your .npy files -- "
                "config_parser.Dataset silently falls back to generic 'class_N' names "
                "on a count mismatch, which is what just happened. Check what's "
                "actually in your files:\n"
                "    python3 -c \"import numpy as np; y=np.load('<your y_train path>'); "
                "print(np.unique(y, return_counts=True))\"\n"
                "and confirm the count matches config.data.class_names. A common cause "
                "is x_train_path/y_train_path still pointing at .npy files from an "
                "earlier/different preprocessing run.\n\n"
                "If you're intentionally using a dataset without an Infiltration-style "
                "reconnaissance class, the reward model in this file needs to be "
                "adapted deliberately (see the HAM/RM defense logic below) rather than "
                "silently running with it disabled."
            )
        try:
            self._ddos_class = dataset.class_names.index("DoS/DDoS")
        except ValueError:
            raise RuntimeError(
                "environment.py requires a class literally named 'DoS/DDoS' in "
                "config.data.class_names to know which security-event label RM "
                f"defends against, but it's not there. Current class_names: "
                f"{dataset.class_names} (num_classes inferred from data: {dataset.num_classes}).\n\n"
                "See the 'Infiltration' error above for the likely cause (a class-count "
                "mismatch between config.data.class_names and your actual .npy files) "
                "and how to check it."
            )

        self.action_space = spaces.Dict({
            "macro": spaces.Discrete(NUM_MACRO_ACTIONS),
            "ip_assignment": spaces.MultiDiscrete([self.num_ip_pools] * self.n_nodes),
            "route_assignment": spaces.MultiDiscrete([3] * self.num_flows),  # index into candidates
        })
        self.observation_space = spaces.MultiDiscrete([dataset.num_classes] * self.n_nodes)

        self._t = 0
        self.stats = EpisodeStats()
        self._prev_ip_assignment: Optional[np.ndarray] = None
        self._prev_route_assignment: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _row_indices_for_timestep(self, t: int) -> np.ndarray:
        """Deterministic round-robin mapping from (timestep, node) -> dataset row."""
        n_rows = len(self._env_y_train)
        base = (t * self.n_nodes) % n_rows
        return (base + np.arange(self.n_nodes)) % n_rows

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Starts a new training "episode" for bookkeeping purposes only. The
        underlying data pointer (self._t) is NOT rewound to 0 -- CM-MTD's
        SMDP is continuous-time (Section III-C), so episode boundaries here
        are just chunks of one uninterrupted walk through the trace, not
        independent restarts. Pass options={"full_reset": True} to force a
        rewind (e.g. for a fresh evaluation pass over the test set).
        """
        super().reset(seed=seed)
        if options and options.get("full_reset"):
            self._t = 0
        self.stats.reset()
        self._prev_ip_assignment = None
        self._prev_route_assignment = None
        row_idx = self._row_indices_for_timestep(self._t)
        raw_events = self._env_y_train[row_idx]
        features = self._env_X_train[row_idx]
        info = {"raw_events": raw_events, "features": features, "row_idx": row_idx}
        return raw_events.copy(), info

    def step(self, action: dict):
        row_idx = self._row_indices_for_timestep(self._t)
        raw_events = self._env_y_train[row_idx]
        features = self._env_X_train[row_idx]

        macro = int(action["macro"])
        ham_active, rm_active = _MACRO_TO_FLAGS[macro]
        ip_assignment = np.asarray(action["ip_assignment"])
        route_assignment = np.asarray(action["route_assignment"])

        scans_this_step = 0
        compromised_switches_this_step = 0
        compromised_switch_ids: set[int] = set()

        # --- HAM vs. reconnaissance (Infiltration) events, per node -----
        for i in range(self.n_nodes):
            if raw_events[i] == self._infiltration_class:
                self.stats.scanned_nodes_total += 1
                attacker_target = _deterministic_target_index(features[i], self.num_ip_pools)
                defended = ham_active and (ip_assignment[i] != attacker_target)
                if not defended:
                    scans_this_step += 1
                    self.stats.scan_successes_total += 1

        # --- RM vs. DDoS events, per flow's destination node -------------
        for f in range(self.num_flows):
            dest_node = self._flow_dest_node[f]
            if raw_events[dest_node] == self._ddos_class:
                candidates = self.flow_routes[f]
                chosen_idx = int(route_assignment[f]) % len(candidates)
                route = candidates[chosen_idx]
                self.stats.route_exposures_total += len(route)
                attacker_target_switch = _deterministic_target_index(
                    features[dest_node], self.num_switches
                )
                defended = rm_active and (attacker_target_switch not in route)
                if not defended:
                    compromised_switches_this_step += 1
                    compromised_switch_ids.add(attacker_target_switch)
                    self.stats.switch_compromises_total += 1

        # --- Reward: Eq. 1 (defense) -------------------------------------
        r_cfg = self.reward_cfg
        attacks_successful = (scans_this_step > 0) or (compromised_switches_this_step > 0)
        if attacks_successful:
            r_defense = -r_cfg["alpha1"] * scans_this_step - r_cfg["alpha2"] * compromised_switches_this_step
        else:
            r_defense = r_cfg["positive_constant_C"]

        # --- Reward: Eq. 2 (resource consumption) -------------------------
        r_resource = 0.0
        if macro != MACRO_STATIC:
            n_ip_changes = (
                self.n_nodes if self._prev_ip_assignment is None
                else int(np.sum(ip_assignment != self._prev_ip_assignment))
            )
            n_route_changes = (
                self.num_flows if self._prev_route_assignment is None
                else int(np.sum(route_assignment != self._prev_route_assignment))
            )
            r_resource = -(
                r_cfg["gamma1"] * r_cfg["ham_unit_cost"] * n_ip_changes
                + r_cfg["gamma2"] * r_cfg["rm_unit_cost"] * n_route_changes
            )

        r_total = r_defense + r_resource

        self._prev_ip_assignment = ip_assignment.copy()
        self._prev_route_assignment = route_assignment.copy()

        self._t += 1
        next_row_idx = self._row_indices_for_timestep(self._t)
        next_raw_events = self._env_y_train[next_row_idx]

        terminated = False  # episode length is controlled by the training loop (T steps)
        truncated = False
        next_features = self._env_X_train[next_row_idx]
        info = {
            "raw_events": next_raw_events,
            "features": next_features,
            "row_idx": next_row_idx,
            "r_defense": r_defense,
            "r_resource": r_resource,
            "scans_this_step": scans_this_step,
            "compromised_switches_this_step": compromised_switches_this_step,
            "macro_action": macro,
        }
        return next_raw_events.copy(), r_total, terminated, truncated, info


class AttackPredictionStateWrapper(gym.Wrapper):
    """
    Wraps DigitalTwinNetworkEnv so the observation exposed to the
    hierarchical agents is the two-stage attack predictor's *forecast* of
    the next security-event vector (matching the paper's SMDP state
    definition, Section III-C-1), rather than the raw ground-truth event.

    Works with either LSTMAttackPredictor or TransformerAttackPredictor
    (models.py) interchangeably -- both share the same
    predict_next_events(label_window) interface, so swapping
    config.predictor_type doesn't require touching this wrapper.

    Fig. 5's flowchart is: Environment --observations--> LSTM --state-->
    DQN/PPO. Concretely that arrow is now two hops: raw per-node features
    -> Stage 1 (EventClassifier) turns them into a classified event label
    -> Stage 2 forecasts the NEXT label from recent label history. See
    models.py's EventClassifier docstring for why this is two stages
    instead of one sequence model over raw features.
    """

    def __init__(self, env: DigitalTwinNetworkEnv, event_classifier, stage2_predictor, sequence_length: int):
        super().__init__(env)
        self.event_classifier = event_classifier
        self.stage2_predictor = stage2_predictor
        self.sequence_length = sequence_length
        # History of per-node CLASSIFIED EVENT LABELS (Stage 1's output),
        # shape (n_nodes,) each -- Stage 2 operates on label sequences,
        # not raw features (see models.py).
        self._label_history: list[np.ndarray] = []
        self.observation_space = env.observation_space

    def _classify(self, features: np.ndarray) -> np.ndarray:
        """features: (n_nodes, input_dim) -> (n_nodes,) predicted event labels."""
        return self.event_classifier.predict(features)

    def _push_history(self, labels: np.ndarray) -> None:
        self._label_history.append(labels)
        if len(self._label_history) > self.sequence_length:
            self._label_history.pop(0)

    def _predicted_state(self) -> np.ndarray:
        if len(self._label_history) < self.sequence_length:
            pad = [self._label_history[0]] * (self.sequence_length - len(self._label_history))
            window = np.stack(pad + self._label_history, axis=0)
        else:
            window = np.stack(self._label_history, axis=0)
        # window: (seq_len, n_nodes) -> (n_nodes, seq_len) for a batched,
        # per-node prediction call.
        window = np.transpose(window, (1, 0))
        return self.stage2_predictor.predict_next_events(window)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self._label_history = []
        self._push_history(self._classify(info["features"]))
        predicted_state = self._predicted_state()
        return predicted_state, info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        self._push_history(self._classify(info["features"]))
        predicted_state = self._predicted_state()
        return predicted_state, reward, terminated, truncated, info
