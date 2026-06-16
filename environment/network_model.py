"""
Network Model — Section III-A of the paper.

Models the SDN-IoT architecture in DTMN as an undirected graph:
    G = {V, M, E}

Where:
    V  = set of IoT devices and cloud nodes vⁿᵢ  (1 ≤ i ≤ n)
    M  = set of OpenFlow switches vˢⱼ              (1 ≤ j ≤ m)
    E  = set of wireless/wired links

Simulates:
    - Network reconnaissance (PortScan) — targeted at nodes V
    - DDoS attacks                       — targeted at switches M / flows F
    - HAM: reduces reconnaissance success probability
    - RM:  reduces DDoS success probability

Waxman topology model (paper Table I: α=0.2, β=0.15).
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("cm_mtd.network")

# Macro-action IDs (must match reward_functions.py)
MACRO_STATIC   = 0
MACRO_HAM_ONLY = 1
MACRO_RM_ONLY  = 2
MACRO_HAM_RM   = 3


class NetworkNode:
    """Represents an IoT device or cloud node vⁿᵢ."""

    def __init__(self, node_id: int, n_ip_spaces: int = 3) -> None:
        self.node_id = node_id
        self.n_ip_spaces = n_ip_spaces
        # Current IP address space assignment (HAM changes this)
        self.current_ip_space: int = node_id % n_ip_spaces
        # Vulnerability level: increases if node is scanned often
        self.vulnerability: float = 0.0
        # Whether this node is currently under reconnaissance
        self.under_recon: bool = False
        # Resource cost of HAM for this node
        self.ham_cost: float = 0.1 + 0.05 * np.random.random()


class OpenFlowSwitch:
    """Represents an OpenFlow switch vˢⱼ."""

    def __init__(self, switch_id: int, flow_table_capacity: int = 100) -> None:
        self.switch_id = switch_id
        self.flow_table_capacity = flow_table_capacity
        self.current_flow_count: int = 0
        # Whether this switch is on a DDoS-targeted route
        self.on_attack_route: bool = False
        # Whether this switch has been compromised
        self.compromised: bool = False


class NetworkModel:
    """
    SDN-IoT network model for DTMN simulation.

    Simulates:
    1. Waxman random topology generation
    2. Threat model: reconnaissance + DDoS (cyber kill chain)
    3. MTD effects: HAM (disrupts recon) and RM (disrupts DDoS)
    4. Defense Success Ratio (DSR) computation (Eq. 19)

    Args:
        n_nodes: Number of IoT/cloud nodes n.
        n_switches: Number of OpenFlow switches m.
        n_flows: Number of flows k.
        n_ip_spaces: Total IP address spaces w.
        waxman_alpha: Waxman model α (paper: 0.2).
        waxman_beta: Waxman model β (paper: 0.15).
        seed: Random seed.
    """

    def __init__(
        self,
        n_nodes: int = 12,
        n_switches: int = 12,
        n_flows: int = 10,
        n_ip_spaces: int = 30,
        waxman_alpha: float = 0.2,
        waxman_beta: float = 0.15,
        flow_table_capacity: int = 100,
        seed: int = 42,
    ) -> None:
        self.n = n_nodes
        self.m = n_switches
        self.k = n_flows
        self.w = n_ip_spaces
        self.alpha = waxman_alpha
        self.beta = waxman_beta
        self.flow_table_cap = flow_table_capacity
        self.rng = np.random.default_rng(seed)

        # Initialize network components
        self.nodes: List[NetworkNode] = [
            NetworkNode(i, n_ip_spaces=3) for i in range(n_nodes)
        ]
        self.switches: List[OpenFlowSwitch] = [
            OpenFlowSwitch(j, flow_table_capacity) for j in range(n_switches)
        ]

        # Topology as adjacency matrix
        self.adj_matrix = self._generate_waxman_topology()

        # Flow routing tables: flow_id → list of switch IDs on route
        self.flow_routes: Dict[int, List[int]] = self._initialize_routes()

        # Attack state
        self.recon_targets: List[int] = []   # nodes under scanning
        self.ddos_targets: List[int] = []    # flow IDs under DDoS

        logger.info(
            f"Network model initialized: {n_nodes} nodes, "
            f"{n_switches} switches, {n_flows} flows"
        )

    # ─── Topology Generation ────────────────────────────────────────────────

    def _generate_waxman_topology(self) -> np.ndarray:
        """
        Generate Waxman random topology adjacency matrix.

        P(u, v) = β · exp(−d(u, v) / (α · L))

        Where d(u,v) is the Euclidean distance between nodes u and v,
        and L is the maximum distance across the graph.
        """
        total = self.n + self.m
        # Random 2D positions
        positions = self.rng.uniform(0, 1, (total, 2))

        # Pairwise distances
        diffs = positions[:, None, :] - positions[None, :, :]
        dist = np.sqrt(np.sum(diffs ** 2, axis=-1))
        L = dist.max()

        # Waxman connection probabilities
        P = self.beta * np.exp(-dist / (self.alpha * L + 1e-8))

        # Sample adjacency
        adj = (self.rng.uniform(0, 1, (total, total)) < P).astype(int)
        np.fill_diagonal(adj, 0)
        adj = np.maximum(adj, adj.T)  # symmetric

        return adj

    def _initialize_routes(self) -> Dict[int, List[int]]:
        """Assign random routes (subset of switches) to each flow."""
        routes = {}
        for f in range(self.k):
            # Each flow passes through 2–4 switches
            n_hops = self.rng.integers(2, min(5, self.m + 1))
            switch_ids = self.rng.choice(self.m, size=n_hops, replace=False).tolist()
            routes[f] = switch_ids
        return routes

    # ─── Attack Simulation ──────────────────────────────────────────────────

    def step_attack(
        self,
        attack_type: int,
        macro_action: int,
        predicted_events: np.ndarray,
    ) -> Dict[str, float]:
        """
        Simulate one time-slot of attack and defense interaction.

        Args:
            attack_type: Current attack class (0=BENIGN, 1=DoS, 2=DDoS,
                         3=PortScan, 4=Infiltration, 5=Bot, 6=BruteForce, 7=WebAttack).
            macro_action: Active MTD macro-action (0–3).
            predicted_events: LSTM output [n_nodes, n_classes].

        Returns:
            Dict with:
                n_scanned:              Θ_t — successful node scans
                n_switches_compromised: Υ_t — compromised switches
                total_scanned:          L^s_t
                total_switches_route:   L^d_t
                attack_success:         bool
                ham_costs:              per-node HAM resource array
                rm_costs:               per-flow RM resource array
        """
        # Reset per-step state
        for node in self.nodes:
            node.under_recon = False
        for sw in self.switches:
            sw.compromised = False

        n_scanned = 0.0
        n_switches_compromised = 0.0
        total_scanned = float(self.n)
        total_switches_on_route = float(sum(len(r) for r in self.flow_routes.values()))
        attack_success = False

        # ── HAM effect: shuffle IP assignments (disrupts reconnaissance) ──
        ham_active = macro_action in {MACRO_HAM_ONLY, MACRO_HAM_RM}
        rm_active  = macro_action in {MACRO_RM_ONLY, MACRO_HAM_RM}

        if ham_active:
            for node in self.nodes:
                # Rotate IP space assignment randomly
                node.current_ip_space = int(self.rng.integers(0, self.w))

        # ── RM effect: mutate routes (disrupts DDoS flows) ────────────────
        if rm_active:
            for f in range(self.k):
                n_hops = len(self.flow_routes[f])
                new_route = self.rng.choice(self.m, size=n_hops, replace=False).tolist()
                self.flow_routes[f] = new_route

        # ── Simulate attack outcome based on type ─────────────────────────
        if attack_type in {3, 4, 6}:  # PortScan / Infiltration / BruteForce → reconnaissance
            for node in self.nodes:
                node.under_recon = True
                # HAM increases difficulty of scanning
                ham_reduction = 0.7 if ham_active else 0.0
                # Base scan success rate depends on vulnerability
                scan_prob = max(0.0, 0.5 + node.vulnerability * 0.1 - ham_reduction)
                if self.rng.random() < scan_prob:
                    n_scanned += 1
                    node.vulnerability = min(1.0, node.vulnerability + 0.05)
                    attack_success = True
                else:
                    node.vulnerability = max(0.0, node.vulnerability - 0.01)

        if attack_type in {1, 2}:  # DoS / DDoS → attack execution
            for f in range(self.k):
                for sw_id in self.flow_routes[f]:
                    sw = self.switches[sw_id]
                    # RM diverts traffic to uncongested routes
                    rm_reduction = 0.65 if rm_active else 0.0
                    ddos_prob = max(0.0, 0.6 - rm_reduction)
                    if self.rng.random() < ddos_prob:
                        sw.compromised = True
                        n_switches_compromised += 1
                        attack_success = True

        if attack_type in {5, 7}:  # Bot / WebAttack — mixed
            # Moderate impact, partially mitigated by both HAM and RM
            base_success = 0.4
            if ham_active:
                base_success -= 0.15
            if rm_active:
                base_success -= 0.15
            if self.rng.random() < max(0.0, base_success):
                n_scanned += self.rng.integers(1, max(2, self.n // 4))
                attack_success = True

        # ── Compute resource costs ─────────────────────────────────────────
        ham_costs = np.array([nd.ham_cost for nd in self.nodes], dtype=np.float32)
        rm_costs = np.array(
            [0.05 + 0.03 * len(self.flow_routes[f]) for f in range(self.k)],
            dtype=np.float32,
        )

        if not ham_active:
            ham_costs[:] = 0.0
        if not rm_active:
            rm_costs[:] = 0.0

        return {
            "n_scanned": n_scanned,
            "n_switches_compromised": n_switches_compromised,
            "total_scanned": total_scanned,
            "total_switches_route": total_switches_on_route,
            "attack_success": attack_success,
            "ham_costs": ham_costs,
            "rm_costs": rm_costs,
        }

    # ─── Network Performance ─────────────────────────────────────────────────

    def compute_rtt(self, macro_action: int) -> float:
        """
        Estimate Round Trip Time (ms) — Figure 10a metric.

        HAM and RM increase RTT due to IP reassignment and route changes.
        Baseline (no mutation) RTT ≈ 1 ms.
        CM-MTD RTT ≈ 2–4 ms (paper result).
        """
        base_rtt = 1.0  # ms, no mutation baseline
        ham_active = macro_action in {MACRO_HAM_ONLY, MACRO_HAM_RM}
        rm_active  = macro_action in {MACRO_RM_ONLY, MACRO_HAM_RM}

        rtt = base_rtt
        if ham_active:
            rtt += self.rng.normal(1.5, 0.5)   # HAM adds IP reassignment delay
        if rm_active:
            rtt += self.rng.normal(1.0, 0.4)   # RM adds route mutation delay

        return float(max(base_rtt, rtt))

    def compute_plr(self, macro_action: int) -> float:
        """
        Estimate Packet Loss Ratio (%) — Figure 10b metric.

        CM-MTD mostly 0% PLR with occasional spikes due to RM/HAM (paper Fig 10b).
        """
        ham_active = macro_action in {MACRO_HAM_ONLY, MACRO_HAM_RM}
        rm_active  = macro_action in {MACRO_RM_ONLY, MACRO_HAM_RM}

        # Base PLR near 0%
        plr = 0.0
        if ham_active and self.rng.random() < 0.05:  # 5% chance of transient packet loss
            plr += self.rng.uniform(5, 15)
        if rm_active and self.rng.random() < 0.03:
            plr += self.rng.uniform(3, 10)

        return float(plr)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset network state for a new episode."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for node in self.nodes:
            node.vulnerability = 0.0
            node.under_recon = False
            node.current_ip_space = node.node_id % 3
        for sw in self.switches:
            sw.compromised = False
            sw.current_flow_count = 0
        self.flow_routes = self._initialize_routes()
