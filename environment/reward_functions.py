"""
Reward Function Implementation — Section III-C of the paper.

Implements:
  R_d  (Eq. 1) — defense effectiveness reward
  R_c  (Eq. 2) — resource consumption penalty
  R_total = R_d + R_c  (Eq. 3)

R_d:
  If attacks are successful:
    R_d = -α₁ Σᵢ Θ_{t,i} - α₂ Υ_t
  Otherwise:
    R_d = C   (positive constant)

Where:
  Θ_{t,i} = number of times node vⁿᵢ was scanned successfully at slot t
  Υ_t      = number of OpenFlow switches compromised by DDoS at slot t
  α₁, α₂  = penalty coefficients
  C        = positive reward for successful defense

R_c:
  If macro-action ∈ {o_c, o_a, o_r} (MTD active):
    R_c = W_t = -γ₁ Σᵢ eᵃᵢ - γ₂ Σₓ eʳₓ
  If macro-action = o_s (static):
    R_c = 0

Where:
  eᵃᵢ = resource consumption of HAM for node vⁿᵢ
  eʳₓ = resource consumption of RM for flow f_y
  γ₁, γ₂ = resource cost coefficients
"""
import logging
from typing import NamedTuple

import numpy as np

logger = logging.getLogger("cm_mtd.reward")

# Macro-action identifiers (matching SMDP macro-action space O)
MACRO_STATIC   = 0   # o_s: static IP + routes
MACRO_HAM_ONLY = 1   # o_a: HAM only
MACRO_RM_ONLY  = 2   # o_r: RM only
MACRO_HAM_RM   = 3   # o_c: both HAM and RM

MTD_ACTIVE_MACROS = {MACRO_HAM_ONLY, MACRO_RM_ONLY, MACRO_HAM_RM}


class RewardComponents(NamedTuple):
    """Container for decomposed reward components."""
    R_total: float
    R_defense: float
    R_resource: float
    attack_success: bool
    n_scanned: float
    n_switches_compromised: float


class RewardFunction:
    """
    Computes the total reward R_total = R_d + R_c at each SMDP time slot.

    Args:
        alpha1: Coefficient for scanning penalty (Eq. 1).
        alpha2: Coefficient for DDoS switch compromise penalty (Eq. 1).
        C_defense: Positive constant reward for successful defense (Eq. 1).
        gamma1: HAM resource cost coefficient (Eq. 2).
        gamma2: RM resource cost coefficient (Eq. 2).
    """

    def __init__(
        self,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        C_defense: float = 10.0,
        gamma1: float = 0.5,
        gamma2: float = 0.5,
    ) -> None:
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.C = C_defense
        self.gamma1 = gamma1
        self.gamma2 = gamma2

    def compute(
        self,
        macro_action: int,
        n_nodes_scanned: float,
        n_switches_compromised: float,
        ham_resource_costs: np.ndarray,
        rm_resource_costs: np.ndarray,
        attack_success: bool,
    ) -> RewardComponents:
        """
        Compute R_total = R_d + R_c.

        Args:
            macro_action:            Selected macro-action (0–3).
            n_nodes_scanned:         Θ_t — total successful node scans at slot t.
            n_switches_compromised:  Υ_t — OpenFlow switches compromised by DDoS.
            ham_resource_costs:      eᵃ array [n_nodes] — HAM resource per node.
            rm_resource_costs:       eʳ array [n_flows] — RM resource per flow.
            attack_success:          True if any attack succeeded at this slot.

        Returns:
            RewardComponents namedtuple.
        """
        # ── Defense Reward R_d (Eq. 1) ────────────────────────────────────
        if attack_success:
            R_d = (
                -self.alpha1 * float(n_nodes_scanned)
                - self.alpha2 * float(n_switches_compromised)
            )
        else:
            R_d = self.C

        # ── Resource Reward R_c (Eq. 2) ───────────────────────────────────
        if macro_action in MTD_ACTIVE_MACROS:
            W_t = (
                -self.gamma1 * float(np.sum(ham_resource_costs))
                - self.gamma2 * float(np.sum(rm_resource_costs))
            )
            R_c = W_t
        else:
            R_c = 0.0

        R_total = R_d + R_c

        return RewardComponents(
            R_total=R_total,
            R_defense=R_d,
            R_resource=R_c,
            attack_success=attack_success,
            n_scanned=float(n_nodes_scanned),
            n_switches_compromised=float(n_switches_compromised),
        )

    @classmethod
    def from_config(cls, config: dict) -> "RewardFunction":
        """Instantiate from config dict."""
        rwd = config.get("reward", {})
        return cls(
            alpha1=rwd.get("alpha1", 1.0),
            alpha2=rwd.get("alpha2", 1.0),
            C_defense=rwd.get("C_defense", 10.0),
            gamma1=rwd.get("gamma1", 0.5),
            gamma2=rwd.get("gamma2", 0.5),
        )
