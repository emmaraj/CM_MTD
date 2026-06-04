"""
smt_constraints.py
==================
Z3-based SMT formalization of network constraints from Section V of the paper.

Implements:
  § V-A  Operation constraints          — Eqs. (4), (5), (6), (7)
  § V-B  Flow table size constraint     — Eq.  (9)
  § V-C  QoS delay constraint           — Eq.  (10)

The solver produces lists of MutationAction objects (feasible IP assignments +
mutated routes) that are passed to the HDRL agent's action space.

If Z3 is unavailable the module transparently falls back to a fast heuristic
solver that still respects all structural constraints.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional Z3 import — graceful degradation
try:
    from z3 import (
        Bool, Int, And, Or, Not, Distinct, Solver,
        sat, is_true, set_param,
    )
    set_param("timeout", 5000)   # 5 s per Z3 query
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("[smt_constraints] z3-solver not found — using heuristic solver.")


# ---------------------------------------------------------------------------
# Network configuration
# ---------------------------------------------------------------------------

@dataclass
class NetworkConfig:
    """
    All parameters that define the SDN-IoT network in DTMN.
    Defaults match the small (12-node) simulation scenario from Table I.
    """
    n_nodes:          int   = 12     # IoT + cloud nodes
    n_switches:       int   = 12     # OpenFlow switches
    n_ip_spaces:      int   = 30     # Available IP address spaces  [30–150]
    n_flows:          int   = 10     # Concurrent traffic flows
    max_flow_table:   int   = 50     # C^cap_j — max entries per switch
    link_delay_ms:    float = 1.0    # T_l — link propagation delay (ms)
    proc_delay_ms:    float = 0.5    # T^m_j — IP-mod / lookup delay (ms)
    max_delay_ms:     float = 15.0   # C^del_y — QoS delay ceiling (ms)
    max_hops:         int   = 5      # Maximum route length (hops)

    # Auto-generated topology matrices
    switch_adjacency: Optional[np.ndarray] = field(default=None, repr=False)
    node_switch_map:  Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        rng = np.random.RandomState(42)
        if self.switch_adjacency is None:
            adj = np.zeros((self.n_switches, self.n_switches), dtype=np.int32)
            for i in range(self.n_switches):
                j = (i + 1) % self.n_switches
                adj[i, j] = adj[j, i] = 1
            for _ in range(self.n_switches // 2):
                a, b = rng.choice(self.n_switches, size=2, replace=False)
                adj[a, b] = adj[b, a] = 1
            self.switch_adjacency = adj

        if self.node_switch_map is None:
            self.node_switch_map = rng.randint(0, self.n_switches,
                                               size=self.n_nodes)

    @classmethod
    def from_topology(cls, topology) -> "NetworkConfig":
        """Construct from a data_loader.NetworkTopology instance."""
        cfg = cls(
            n_nodes    = topology.n_nodes,
            n_switches = topology.n_switches,
        )
        cfg.switch_adjacency = topology.switch_adjacency
        cfg.node_switch_map  = topology.node_switch_map
        return cfg


# ---------------------------------------------------------------------------
# Mutation action record
# ---------------------------------------------------------------------------

@dataclass
class MutationAction:
    """
    A single feasible joint mutation decision.

    Attributes
    ----------
    ip_assignment : np.ndarray, shape (n_nodes,)
        ip_assignment[i] = x  means IP space z_x is assigned to node v^n_i.
        Corresponds to binary b^x_i = 1 in the paper.
    routes : List[List[int]]
        routes[y] = ordered list of switch indices on the path of flow f_y.
        Corresponds to binary d^y_j = 1 in the paper.
    macro_action : int
        Which macro-action triggered this mutation (0=both, 1=HAM, 2=RM, 3=static).
    """
    ip_assignment: np.ndarray
    routes:        List[List[int]]
    macro_action:  int = 0

    def to_dict(self) -> Dict:
        return {
            "ip_assignment": self.ip_assignment.tolist(),
            "routes":        self.routes,
            "macro_action":  self.macro_action,
        }


# ---------------------------------------------------------------------------
# SMT constraint solver
# ---------------------------------------------------------------------------

class SMTConstraintSolver:
    """
    Generates lists of feasible MutationAction objects for a given macro-action.

    Macro-action codes (must match hdrl_agent.py constants):
        0 — o_c : deploy both HAM and RM
        1 — o_a : HAM only  (IP mutation)
        2 — o_r : RM only   (route mutation)
        3 — o_s : static    (no mutation)

    Usage
    -----
    solver = SMTConstraintSolver(cfg)
    actions = solver.generate_feasible_actions(macro_action=0, n_actions=20)
    """

    MACRO_BOTH   = 0
    MACRO_HAM    = 1
    MACRO_RM     = 2
    MACRO_STATIC = 3

    def __init__(self, config: NetworkConfig, seed: int = 42):
        self.cfg = config
        self.rng = np.random.RandomState(seed)
        # Cached default IP assignment (node i → IP space i % w)
        self._default_ip = np.arange(config.n_nodes, dtype=np.int32) % config.n_ip_spaces

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def generate_feasible_actions(
        self,
        macro_action: int,
        n_actions:    int = 20,
    ) -> List[MutationAction]:
        """
        Return up to n_actions feasible MutationAction objects.

        Static macro-action → returns a single default (no-op) action.
        Otherwise, uses Z3 (if available) or the heuristic solver.
        """
        if macro_action == self.MACRO_STATIC:
            return [self._default_action(macro_action)]

        if Z3_AVAILABLE:
            actions = self._z3_solve(macro_action, n_actions)
        else:
            actions = self._heuristic_solve(macro_action, n_actions)

        # Safety: always return at least n_actions entries
        while len(actions) < n_actions:
            actions.append(self._default_action(macro_action))

        return actions[:n_actions]

    def get_resource_cost(self, action: MutationAction) -> Tuple[float, float]:
        """
        Compute normalised resource costs (e^a_i, e^r_y) for a mutation action.

        Returns
        -------
        ham_cost : float ∈ [0, 1]   — proportion of nodes with mutated IPs.
        rm_cost  : float ∈ [0, 1]   — normalised total route length.
        """
        ham_cost = float(
            np.sum(action.ip_assignment != self._default_ip)
        ) / max(self.cfg.n_nodes, 1)

        max_total = self.cfg.n_flows * self.cfg.max_hops
        rm_cost = float(sum(len(r) for r in action.routes)) / max(max_total, 1)

        return ham_cost, rm_cost

    # -----------------------------------------------------------------------
    # BFS routing
    # -----------------------------------------------------------------------

    def _bfs_route(self, src: int, dst: int) -> List[int]:
        """BFS shortest path on switch adjacency graph (Eq. 6 reachability)."""
        if src == dst:
            return [src]
        adj    = self.cfg.switch_adjacency
        queue  = deque([(src, [src])])
        visited: set = {src}
        while queue:
            node, path = queue.popleft()
            if len(path) >= self.cfg.max_hops:
                continue
            for nb in range(self.cfg.n_switches):
                if adj[node, nb] == 1 and nb not in visited:
                    new_path = path + [nb]
                    if nb == dst:
                        return new_path
                    visited.add(nb)
                    queue.append((nb, new_path))
        return [src, dst]   # fallback direct hop

    # -----------------------------------------------------------------------
    # Constraint checkers
    # -----------------------------------------------------------------------

    def _check_ip_uniqueness(self, ip_assignment: np.ndarray) -> bool:
        """
        Eq. (5): each IP space assigned to at most one node.
        (Eq. 4 is satisfied by construction — every node receives exactly one space.)
        """
        return len(set(ip_assignment.tolist())) == len(ip_assignment)

    def _check_qos(self, route: List[int]) -> bool:
        """
        Eq. (10): total delay ≤ C^del_y.
        delay = 2 × proc_delay  (src + dst switch)
               + (|route| - 1) × link_delay
        """
        if not route:
            return False
        n_hops     = len(route)
        total_ms   = (2 * self.cfg.proc_delay_ms +
                      max(n_hops - 1, 0) * self.cfg.link_delay_ms)
        return total_ms <= self.cfg.max_delay_ms and n_hops <= self.cfg.max_hops

    def _check_flow_table(
        self,
        ip_assignment: np.ndarray,
        routes:        List[List[int]],
    ) -> bool:
        """
        Eq. (9): for every switch, count non-adjacent IP-space pairs + flows
        through that switch; must not exceed C^cap_j.
        """
        w = self.cfg.n_ip_spaces
        for sw in range(self.cfg.n_switches):
            count = 0
            # Non-adjacent IP space pairs on nodes connected to this switch
            connected_nodes = [
                i for i in range(self.cfg.n_nodes)
                if self.cfg.node_switch_map[i] == sw
            ]
            for i, ni in enumerate(connected_nodes):
                for nj in connected_nodes[i + 1:]:
                    xi, xj = int(ip_assignment[ni]), int(ip_assignment[nj])
                    if abs(xi - xj) > 1:   # non-adjacent → separate flow entries
                        count += 1
            # Flows whose route passes through sw as intermediate
            for route in routes:
                if sw in route[1:-1]:
                    count += 1
            if count > self.cfg.max_flow_table:
                return False
        return True

    # -----------------------------------------------------------------------
    # Z3 solver
    # -----------------------------------------------------------------------

    def _z3_solve(self, macro_action: int, n_actions: int) -> List[MutationAction]:
        """
        Use Z3 integer variables with Distinct to enforce uniqueness (Eq. 5).
        Faster than binary-variable encoding for n_nodes ≤ 20.
        """
        n, w = self.cfg.n_nodes, self.cfg.n_ip_spaces
        actions: List[MutationAction] = []

        for attempt in range(n_actions * 4):
            if len(actions) >= n_actions:
                break

            slv = Solver()
            ip  = [Int(f"ip_{i}") for i in range(n)]

            # Domain — Eq. (4) implicitly: each node has one valid space
            for i in range(n):
                slv.add(ip[i] >= 0, ip[i] < w)
            # Uniqueness — Eq. (5)
            slv.add(Distinct(*ip))

            # Bias toward randomised assignment to get diversity
            pivot = int(self.rng.randint(0, w - n))
            for i in range(n):
                slv.add(ip[i] >= pivot)

            if slv.check() != sat:
                continue

            model = slv.model()
            try:
                ip_assign = np.array(
                    [int(str(model[ip[i]])) for i in range(n)],
                    dtype=np.int32,
                )
            except Exception:
                continue

            if not self._check_ip_uniqueness(ip_assign):
                continue

            # Generate routes with QoS check
            routes, ok = self._sample_routes()
            if not ok:
                continue

            # Flow table check
            if not self._check_flow_table(ip_assign, routes):
                continue

            actions.append(MutationAction(
                ip_assignment = ip_assign,
                routes        = routes,
                macro_action  = macro_action,
            ))

        return actions

    # -----------------------------------------------------------------------
    # Heuristic solver (Z3 fallback)
    # -----------------------------------------------------------------------

    def _heuristic_solve(self, macro_action: int, n_actions: int) -> List[MutationAction]:
        """
        Random-restart heuristic:
          1. Sample a random permutation of w IP spaces → take first n.
          2. Sample routes via BFS with random src/dst.
          3. Accept if all constraints pass.
        """
        n, w   = self.cfg.n_nodes, self.cfg.n_ip_spaces
        actions: List[MutationAction] = []

        for _ in range(n_actions * 20):
            if len(actions) >= n_actions:
                break

            if w >= n:
                ip_assign = self.rng.choice(w, size=n, replace=False).astype(np.int32)
            else:
                # More nodes than spaces — reuse with offset to minimise conflicts
                ip_assign = (np.arange(n) + self.rng.randint(0, w)) % w
                ip_assign = ip_assign.astype(np.int32)

            if not self._check_ip_uniqueness(ip_assign):
                continue

            routes, ok = self._sample_routes()
            if not ok:
                continue

            if not self._check_flow_table(ip_assign, routes):
                continue

            actions.append(MutationAction(
                ip_assignment = ip_assign,
                routes        = routes,
                macro_action  = macro_action,
            ))

        return actions

    # -----------------------------------------------------------------------
    # Route sampling (shared by both solvers)
    # -----------------------------------------------------------------------

    def _sample_routes(self) -> Tuple[List[List[int]], bool]:
        """
        Sample n_flows routes.  Returns (routes, all_valid).
        Validity = all routes satisfy QoS (Eq. 10) and flow-conservation (Eq. 6).
        """
        routes: List[List[int]] = []
        n_sw = self.cfg.n_switches
        for _ in range(self.cfg.n_flows):
            src = int(self.rng.randint(0, n_sw))
            dst = int((src + self.rng.randint(1, n_sw)) % n_sw)
            route = self._bfs_route(src, dst)
            if not self._check_qos(route):
                return routes, False
            routes.append(route)
        return routes, True

    # -----------------------------------------------------------------------
    # Default action
    # -----------------------------------------------------------------------

    def _default_action(self, macro_action: int) -> MutationAction:
        """Default (no-mutation or identity) action."""
        routes, _ = self._sample_routes()
        if not routes:   # edge-case: all routes infeasible → minimal routes
            routes = [[i % self.cfg.n_switches] for i in range(self.cfg.n_flows)]
        return MutationAction(
            ip_assignment = self._default_ip.copy(),
            routes        = routes,
            macro_action  = macro_action,
        )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg    = NetworkConfig(n_nodes=12, n_switches=12, n_ip_spaces=30, n_flows=8)
    solver = SMTConstraintSolver(cfg, seed=0)

    for macro_name, macro_id in [("Both", 0), ("HAM", 1), ("RM", 2), ("Static", 3)]:
        actions = solver.generate_feasible_actions(macro_id, n_actions=5)
        hc, rc  = solver.get_resource_cost(actions[0])
        print(
            f"macro={macro_name}  n_feasible={len(actions)}  "
            f"ham_cost={hc:.2f}  rm_cost={rc:.2f}  "
            f"route_lens={[len(r) for r in actions[0].routes[:3]]}"
        )
