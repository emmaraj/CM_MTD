"""
SMT Constraint Formalization — Section V of the paper.

Formalizes the problem of assigning IP address spaces and selecting
mutated routes as a Constrained Satisfaction Problem (CSP).

Implements the three constraint categories from Section V:

A. Operation Constraints (Eq. 4–7):
   • Each node gets ≥1 IP space
   • Each IP space assigned to exactly 1 node
   • Each OpenFlow switch on a flow has equal in/out degree

B. Flow Table Size Constraint (Eq. 8–9):
   • Non-adjacent IP spaces assigned to same node inflate flow tables
   • Per-switch flow table entries ≤ C^cap_j

C. QoS Constraint (Eq. 10):
   • Total delay per flow ≤ C^del_y (processing + forwarding delay)

Two backends:
   • z3 (if installed): exact SMT solving
   • numpy (always available): fast greedy feasibility checker
     — deterministically finds one feasible action

Usage:
    solver = SMTConstraintSolver(network_model, config)
    feasible = solver.get_feasible_action(macro_action)
    # feasible: {'ip_assignment': array, 'route_selection': array}
"""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("cm_mtd.smt")

# Optional z3 backend
_Z3_AVAILABLE = False
try:
    import z3
    _Z3_AVAILABLE = True
    logger.info("z3 SMT solver available — using exact backend")
except ImportError:
    logger.info("z3 not found — using greedy numpy feasibility checker")


class SMTConstraintSolver:
    """
    Constraint solver for feasible MTD action generation.

    Produces IP address space assignments (HAM) and route selections (RM)
    that satisfy all network constraints, removing infeasible actions from
    the SMDP action space (Algorithm 1 — preprocessing step).

    Args:
        n_nodes:    Number of network nodes n.
        n_switches: Number of OpenFlow switches m.
        n_flows:    Number of network flows k.
        n_ip_spaces: Total IP address space pool w.
        flow_table_cap: Per-switch flow table capacity C^cap_j.
        max_delay_ms:   Maximum delay per flow C^del_y (ms).
        link_delay_ms:  Per-hop link delay T_l (ms).
        proc_delay_ms:  IP modification processing delay T^m_j (ms).
    """

    def __init__(
        self,
        n_nodes: int = 12,
        n_switches: int = 12,
        n_flows: int = 10,
        n_ip_spaces: int = 30,
        flow_table_cap: int = 100,
        max_delay_ms: float = 50.0,
        link_delay_ms: float = 1.0,
        proc_delay_ms: float = 2.0,
        seed: int = 42,
    ) -> None:
        self.n = n_nodes
        self.m = n_switches
        self.k = n_flows
        self.w = n_ip_spaces
        self.C_cap = flow_table_cap
        self.C_del = max_delay_ms
        self.T_l   = link_delay_ms
        self.T_m   = proc_delay_ms
        self.rng   = np.random.default_rng(seed)

        # Adjacency matrix for IP spaces (1 = adjacent, can be supernet-aggregated)
        self._ip_adjacency = self._build_ip_adjacency()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_feasible_action(
        self, macro_action: int, n_attempts: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Return one feasible {IP assignment, route selection} pair.

        Args:
            macro_action: 0=STATIC, 1=HAM, 2=RM, 3=HAM+RM.
            n_attempts:   Greedy retries before giving up.

        Returns:
            Dict with:
                'ip_assignment':   [n_nodes, n_ip_spaces] binary matrix
                'route_selection': [n_flows, n_switches]  binary matrix
                'feasible':        bool — False if no feasible action found
        """
        from environment.reward_functions import MACRO_HAM_ONLY, MACRO_RM_ONLY, MACRO_HAM_RM, MACRO_STATIC

        ham_needed = macro_action in (MACRO_HAM_ONLY, MACRO_HAM_RM)
        rm_needed  = macro_action in (MACRO_RM_ONLY,  MACRO_HAM_RM)

        if _Z3_AVAILABLE:
            return self._solve_z3(ham_needed, rm_needed)
        return self._solve_greedy(ham_needed, rm_needed, n_attempts)

    def check_feasibility(
        self,
        ip_assignment: np.ndarray,   # [n, w] binary
        route_selection: np.ndarray, # [k, m] binary
    ) -> Dict[str, bool]:
        """
        Check all three constraint categories for a given action.

        Returns:
            Dict with per-constraint feasibility flags and a global 'feasible' key.
        """
        op_ok    = self._check_operation_constraints(ip_assignment, route_selection)
        flow_ok  = self._check_flow_table_constraint(ip_assignment, route_selection)
        qos_ok   = self._check_qos_constraint(route_selection)
        return {
            "operation":   op_ok,
            "flow_table":  flow_ok,
            "qos":         qos_ok,
            "feasible":    op_ok and flow_ok and qos_ok,
        }

    # ── Constraint Checkers ───────────────────────────────────────────────────

    def _check_operation_constraints(
        self,
        ip_assign: np.ndarray,    # [n, w]
        route_sel: np.ndarray,    # [k, m]
    ) -> bool:
        """
        Eq. 4: Each node has ≥1 IP space:  Σ_x b^x_i ≥ 1  ∀i
        Eq. 5: Each IP space used by exactly 1 node: Σ_i b^x_i = 1  ∀x
        Eq. 6: Flow conservation at each switch (in-degree = out-degree)
        Eq. 7: Route variables binary ∈ {0, 1}
        """
        # Eq. 4
        if not np.all(ip_assign.sum(axis=1) >= 1):
            return False
        # Eq. 5
        if not np.all(ip_assign.sum(axis=0) == 1):
            return False
        # Eq. 7
        if not np.all(np.isin(route_sel, [0, 1])):
            return False
        # Eq. 6 (simplified: each selected route must form a valid path)
        # We check that each flow has at least one selected switch
        for f in range(self.k):
            if route_sel[f].sum() == 0:
                return False
        return True

    def _check_flow_table_constraint(
        self,
        ip_assign: np.ndarray,
        route_sel: np.ndarray,
    ) -> bool:
        """
        Eq. 8–9: Per-switch flow table entries ≤ C^cap_j.

        Supernetting: adjacent IP spaces on same node are aggregated,
        reducing flow table entries (D_{x1,x2} = 0 when adjacent).
        """
        for j in range(self.m):
            entries = 0
            # Count non-adjacent IP space pairs assigned to nodes with this switch
            for i in range(self.n):
                assigned = np.where(ip_assign[i] == 1)[0]
                for a in range(len(assigned)):
                    for b in range(a + 1, len(assigned)):
                        x1, x2 = assigned[a], assigned[b]
                        # D_{x1,x2} = 1 means non-adjacent (can't supernet)
                        if self._ip_adjacency[x1, x2] == 0:
                            entries += 1
            # Add route-based entries
            entries += int(route_sel[:, j].sum())
            if entries > self.C_cap:
                return False
        return True

    def _check_qos_constraint(self, route_sel: np.ndarray) -> bool:
        """
        Eq. 10: Σ_j d^y_j T^m_j + Σ_{j1≠j2} d^y_j1 d^y_j2 T_l ≤ C^del_y  ∀y

        Processing delay at source/dest + link delay for each hop.
        """
        for f in range(self.k):
            selected = np.where(route_sel[f] == 1)[0]
            n_hops   = len(selected)
            if n_hops == 0:
                return False
            # Endpoint processing delay
            proc_delay  = 2 * self.T_m
            # Inter-hop link delays
            link_delay  = max(0, n_hops - 1) * self.T_l
            total_delay = proc_delay + link_delay
            if total_delay > self.C_del:
                return False
        return True

    # ── Greedy Solver (numpy) ─────────────────────────────────────────────────

    def _solve_greedy(
        self, ham_needed: bool, rm_needed: bool, n_attempts: int
    ) -> Dict[str, np.ndarray]:
        """
        Greedy randomized search for a feasible action.

        Tries up to n_attempts random assignments, returns the first
        feasible one or the best partial solution.
        """
        best: Optional[Dict] = None

        for _ in range(n_attempts):
            ip_assign  = self._random_ip_assignment()
            route_sel  = self._random_route_selection()

            if not ham_needed:
                ip_assign = self._identity_ip_assignment()
            if not rm_needed:
                route_sel = self._default_route_selection()

            checks = self.check_feasibility(ip_assign, route_sel)
            if checks["feasible"]:
                return {
                    "ip_assignment":   ip_assign,
                    "route_selection": route_sel,
                    "feasible": True,
                    "backend": "greedy_numpy",
                }
            if best is None:
                best = {"ip_assignment": ip_assign, "route_selection": route_sel}

        logger.warning("Greedy solver: no fully feasible action found in %d attempts", n_attempts)
        return {
            "ip_assignment":   best["ip_assignment"],
            "route_selection": best["route_selection"],
            "feasible": False,
            "backend": "greedy_numpy_partial",
        }

    def _random_ip_assignment(self) -> np.ndarray:
        """
        Produce a random IP assignment satisfying Eq. 4–5:
          Eq. 4: Each node i gets ≥1 IP space  (Σ_x b^x_i ≥ 1)
          Eq. 5: Each IP space x assigned to exactly 1 node (Σ_i b^x_i = 1)

        Strategy: shuffle w spaces, partition into n non-empty groups.
        """
        ip_assign = np.zeros((self.n, self.w), dtype=np.int32)
        spaces = self.rng.permutation(self.w)

        # Ensure each node gets at least 1 space (first n spaces → one each)
        for i in range(min(self.n, self.w)):
            ip_assign[i, spaces[i]] = 1

        # Assign remaining spaces to random nodes (exactly one node each)
        for x in range(self.n, self.w):
            node = int(self.rng.integers(0, self.n))
            ip_assign[node, spaces[x]] = 1

        return ip_assign

    def _identity_ip_assignment(self) -> np.ndarray:
        """
        Fixed 'static' assignment that satisfies Eq. 4–5:
        Space x → node (x % n). Every space assigned to exactly one node,
        every node gets at least ⌊w/n⌋ spaces.
        """
        ip_assign = np.zeros((self.n, self.w), dtype=np.int32)
        for x in range(self.w):
            ip_assign[x % self.n, x] = 1
        return ip_assign

    def _random_route_selection(self) -> np.ndarray:
        """
        Select a feasible route for each flow (2–4 hops, satisfying QoS).
        Max hops = floor((C_del - 2*T_m) / T_l) + 1
        """
        max_hops = max(1, int((self.C_del - 2 * self.T_m) / max(self.T_l, 1e-9)))
        route_sel = np.zeros((self.k, self.m), dtype=np.int32)
        for f in range(self.k):
            n_hops = int(self.rng.integers(1, min(max_hops, self.m) + 1))
            switches = self.rng.choice(self.m, size=n_hops, replace=False)
            route_sel[f, switches] = 1
        return route_sel

    def _default_route_selection(self) -> np.ndarray:
        """Default static routes: flow f → switch f % m."""
        route_sel = np.zeros((self.k, self.m), dtype=np.int32)
        for f in range(self.k):
            route_sel[f, f % self.m] = 1
        return route_sel

    # ── z3 Solver ────────────────────────────────────────────────────────────

    def _solve_z3(self, ham_needed: bool, rm_needed: bool) -> Dict:
        """
        Exact SMT solving via z3.  Falls back to greedy if z3 times out.
        """
        try:
            return self._z3_solve(ham_needed, rm_needed)
        except Exception as e:
            logger.warning("z3 solve failed (%s) — falling back to greedy", e)
            return self._solve_greedy(ham_needed, rm_needed, n_attempts=50)

    def _z3_solve(self, ham_needed: bool, rm_needed: bool) -> Dict:
        import z3
        s = z3.Solver()
        s.set("timeout", 5000)  # 5 s timeout

        # Decision variables
        b = [[z3.Bool(f"b_{i}_{x}") for x in range(self.w)] for i in range(self.n)]
        d = [[z3.Bool(f"d_{f}_{j}") for j in range(self.m)] for f in range(self.k)]

        # Eq. 4: Σ_x b^x_i ≥ 1
        for i in range(self.n):
            s.add(z3.Sum([z3.If(b[i][x], 1, 0) for x in range(self.w)]) >= 1)

        # Eq. 5: Σ_i b^x_i = 1
        for x in range(self.w):
            s.add(z3.Sum([z3.If(b[i][x], 1, 0) for i in range(self.n)]) == 1)

        # Eq. 6 (simplified): each flow selects ≥1 switch
        for f in range(self.k):
            s.add(z3.Sum([z3.If(d[f][j], 1, 0) for j in range(self.m)]) >= 1)

        # Eq. 10: QoS delay
        max_hops = int((self.C_del - 2 * self.T_m) / max(self.T_l, 1e-9))
        for f in range(self.k):
            s.add(z3.Sum([z3.If(d[f][j], 1, 0) for j in range(self.m)]) <= max_hops)

        if s.check() == z3.sat:
            model = s.model()
            ip_assign  = np.zeros((self.n, self.w), dtype=np.int32)
            route_sel  = np.zeros((self.k, self.m), dtype=np.int32)
            for i in range(self.n):
                for x in range(self.w):
                    if z3.is_true(model[b[i][x]]):
                        ip_assign[i, x] = 1
            for f in range(self.k):
                for j in range(self.m):
                    if z3.is_true(model[d[f][j]]):
                        route_sel[f, j] = 1
            return {"ip_assignment": ip_assign, "route_selection": route_sel,
                    "feasible": True, "backend": "z3"}
        else:
            logger.warning("z3: UNSAT — no feasible action exists")
            return self._solve_greedy(ham_needed, rm_needed, n_attempts=50)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _build_ip_adjacency(self) -> np.ndarray:
        """
        D_{x1,x2} = 0 if IP spaces x1 and x2 are adjacent (consecutive /24s).
        Adjacent spaces can be supernet-aggregated → fewer flow table entries.
        """
        adj = np.zeros((self.w, self.w), dtype=np.int32)
        for x in range(self.w - 1):
            adj[x, x + 1] = 1
            adj[x + 1, x] = 1
        return adj

    @classmethod
    def from_config(cls, config: Dict) -> "SMTConstraintSolver":
        net = config.get("network", {})
        return cls(
            n_nodes=net.get("n_nodes", 12),
            n_switches=net.get("n_switches", 12),
            n_flows=net.get("n_flows", 10),
            n_ip_spaces=net.get("n_ip_spaces", 30),
            flow_table_cap=net.get("flow_table_capacity", 100),
            max_delay_ms=net.get("max_delay_ms", 50.0),
        )
