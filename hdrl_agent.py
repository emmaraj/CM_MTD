"""
hdrl_agent.py
=============
Hierarchical Deep Reinforcement Learning (HDRL) for CM-MTD — Algorithm 1.

Upper layer  →  DQN  selects macro-actions O ∈ {o_c, o_a, o_r, o_s}.
Lower layer  →  PPO  selects specific mutation actions A (IP + route indices).

NetworkEnvironment provides:
  • Simulated reward computation  (Eqs. 1–3)
  • Defense success ratio         (Eq. 19)
  • RTT / PLR estimation
  • Placeholder hooks for Mininet-WiFi / Ryu integration

All neural-network layers follow Table I hyper-parameters:
  DQN  hidden=[256],        activation=ReLU,   lr=1e-3
  PPO  hidden=[256,256],    activation=tanh,   lr=1e-4
"""

from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Macro-action constants  (shared with train_and_eval.py)
# ---------------------------------------------------------------------------

MACRO_BOTH   = 0   # o_c : deploy HAM + RM
MACRO_HAM    = 1   # o_a : HAM only
MACRO_RM     = 2   # o_r : RM  only
MACRO_STATIC = 3   # o_s : no mutation

N_MACRO_ACTIONS = 4

MACRO_NAMES = {
    MACRO_BOTH  : "Both (HAM+RM)",
    MACRO_HAM   : "HAM Only",
    MACRO_RM    : "RM Only",
    MACRO_STATIC: "Static",
}


# ---------------------------------------------------------------------------
# Transition records
# ---------------------------------------------------------------------------

@dataclass
class UpperTransition:
    """One SMDP step at the upper (DQN) layer."""
    state:        np.ndarray
    macro_action: int
    reward:       float
    next_state:   np.ndarray
    done:         bool


@dataclass
class LowerTransition:
    """One time-slot step at the lower (PPO) layer."""
    state:        np.ndarray
    macro_action: int
    action:       int     # index into feasible-action list
    log_prob:     float   # log π_old(A|S,O)
    reward:       float
    next_state:   np.ndarray
    done:         bool


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self._buf: deque = deque(maxlen=capacity)

    def push(self, transition):
        self._buf.append(transition)

    def sample(self, n: int) -> list:
        return random.sample(self._buf, min(n, len(self._buf)))

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Neural networks
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden: List[int], out_dim: int,
         activation: str = "relu") -> nn.Sequential:
    act = nn.ReLU if activation == "relu" else nn.Tanh
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class DQNNetwork(nn.Module):
    """Q(S, O) network — upper layer."""

    def __init__(self, state_dim: int, n_actions: int,
                 hidden: List[int] = None):
        super().__init__()
        hidden = hidden or [256]
        self.net = _mlp(state_dim, hidden, n_actions, "relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPOActor(nn.Module):
    """π(A | S, O) network — lower layer actor."""

    def __init__(self, state_dim: int, n_actions: int,
                 hidden: List[int] = None):
        super().__init__()
        hidden = hidden or [256, 256]
        # Condition on macro-action via one-hot concatenation
        self.net = _mlp(state_dim + N_MACRO_ACTIONS, hidden, n_actions, "tanh")

    def forward(self, state: torch.Tensor,
                macro_oh: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, macro_oh], dim=-1)
        return F.softmax(self.net(x), dim=-1)


class PPOCritic(nn.Module):
    """V(S, O) network — lower layer critic."""

    def __init__(self, state_dim: int, hidden: List[int] = None):
        super().__init__()
        hidden = hidden or [256, 256]
        self.net = _mlp(state_dim + N_MACRO_ACTIONS, hidden, 1, "tanh")

    def forward(self, state: torch.Tensor,
                macro_oh: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, macro_oh], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Simulated network environment
# ---------------------------------------------------------------------------

class NetworkEnvironment:
    """
    Simulated DTMN environment for HDRL training.

    Computes per-step rewards using Eqs. (1)–(3) and tracks DSR (Eq. 19).
    Placeholder hooks are provided for real Mininet-WiFi / Ryu integration.

    Parameters
    ----------
    alpha1 : scanning-success penalty coefficient
    alpha2 : DDoS-success penalty coefficient
    gamma1 : HAM resource-consumption coefficient
    gamma2 : RM  resource-consumption coefficient
    C      : positive constant for successful defense
    """

    def __init__(
        self,
        n_nodes:   int   = 12,
        n_switches: int  = 12,
        alpha1:    float = 1.0,
        alpha2:    float = 1.5,
        gamma1:    float = 0.5,
        gamma2:    float = 0.5,
        C:         float = 10.0,
        seed:      int   = 42,
    ):
        self.n_nodes    = n_nodes
        self.n_switches = n_switches
        self.alpha1     = alpha1
        self.alpha2     = alpha2
        self.gamma1     = gamma1
        self.gamma2     = gamma2
        self.C          = C
        self.rng        = np.random.RandomState(seed)

        # Episode accumulators (reset each episode)
        self.ep_scan_hits:  int = 0
        self.ep_ddos_hits:  int = 0
        self.ep_scan_total: int = 0
        self.ep_ddos_total: int = 0
        self.step_count:    int = 0

    # -----------------------------------------------------------------------
    # Episode lifecycle
    # -----------------------------------------------------------------------

    def reset(self):
        self.ep_scan_hits  = 0
        self.ep_ddos_hits  = 0
        self.ep_scan_total = 0
        self.ep_ddos_total = 0
        self.step_count    = 0

    # -----------------------------------------------------------------------
    # Reward computation
    # -----------------------------------------------------------------------

    def _ham_effectiveness(self, macro_action: int) -> float:
        """Fraction of scanning attempts disrupted by IP mutation."""
        return 0.70 if macro_action in (MACRO_BOTH, MACRO_HAM) else 0.0

    def _rm_effectiveness(self, macro_action: int) -> float:
        """Fraction of DDoS impact absorbed by route mutation."""
        return 0.70 if macro_action in (MACRO_BOTH, MACRO_RM) else 0.0

    def compute_defense_reward(
        self,
        macro_action:     int,
        predicted_events: np.ndarray,
    ) -> Tuple[float, int, int]:
        """
        R_d — Eq. (1).

        Returns (Rd, n_scan_successes, n_ddos_successes).
        """
        from data_loader import SecurityEvent

        n_infil = int(np.sum(predicted_events == SecurityEvent.INFILTRATION))
        n_ddos  = int(np.sum(predicted_events == SecurityEvent.DOS_DDOS))

        ham_eff = self._ham_effectiveness(macro_action)
        rm_eff  = self._rm_effectiveness(macro_action)

        # Θ_{t,i}: successful scans after HAM   (Eq. 1)
        theta   = max(0, round(n_infil * (1.0 - ham_eff)))
        # Υ_t: compromised switches after RM     (Eq. 1)
        upsilon = max(0, min(round(n_ddos * (1.0 - rm_eff)), self.n_switches))

        self.ep_scan_hits  += theta
        self.ep_ddos_hits  += upsilon
        self.ep_scan_total += n_infil
        self.ep_ddos_total += n_ddos
        self.step_count    += 1

        if theta > 0 or upsilon > 0:
            Rd = -self.alpha1 * theta - self.alpha2 * upsilon
        else:
            Rd = self.C   # successful defense
        return float(Rd), theta, upsilon

    def compute_resource_reward(
        self,
        macro_action: int,
        ham_cost:     float,
        rm_cost:      float,
    ) -> float:
        """
        R_c — Eq. (2).  W_t = −γ1 Σ e^a_i − γ2 Σ e^r_y
        """
        if macro_action == MACRO_STATIC:
            return 0.0
        Wt = -self.gamma1 * ham_cost - self.gamma2 * rm_cost
        return float(Wt)

    def compute_total_reward(
        self,
        macro_action:     int,
        predicted_events: np.ndarray,
        ham_cost:         float,
        rm_cost:          float,
    ) -> Tuple[float, int, int]:
        """
        R_total = R_d + R_c — Eq. (3).
        Also returns scan / DDoS hits for logging.
        """
        Rd, scan_hits, ddos_hits = self.compute_defense_reward(
            macro_action, predicted_events
        )
        Rc = self.compute_resource_reward(macro_action, ham_cost, rm_cost)
        return Rd + Rc, scan_hits, ddos_hits

    # -----------------------------------------------------------------------
    # DSR — Eq. (19)
    # -----------------------------------------------------------------------

    def compute_dsr(self) -> float:
        """
        DSR = (1 − (ΣN^s + ΣN^d) / (ΣL^s + ΣL^d)) × 100 %
        """
        numerator   = self.ep_scan_hits + self.ep_ddos_hits
        denominator = max(self.ep_scan_total + self.ep_ddos_total, 1)
        return max(0.0, min(100.0, (1.0 - numerator / denominator) * 100.0))

    # -----------------------------------------------------------------------
    # Network performance simulation
    # -----------------------------------------------------------------------

    def simulate_network_performance(
        self, macro_action: int
    ) -> Tuple[float, float]:
        """
        Simulated RTT (ms) and PLR (%) when the given macro-action is active.
        Matches Fig. 10 from the paper qualitatively:
          CM-MTD RTT ≈ 2–4 ms  vs  no-mutation ≈ 1 ms.

        TODO — replace with real Mininet measurements (see hooks below).
        """
        base_rtt, base_plr = 1.0, 0.0

        if macro_action == MACRO_STATIC:
            return base_rtt, base_plr

        rtt_delta = 0.0
        plr_delta = 0.0

        if macro_action in (MACRO_BOTH, MACRO_HAM):
            rtt_delta += float(self.rng.uniform(1.0, 3.0))
            plr_delta += float(self.rng.uniform(0.0, 5.0))

        if macro_action in (MACRO_BOTH, MACRO_RM):
            rtt_delta += float(self.rng.uniform(0.5, 2.0))
            plr_delta += float(self.rng.uniform(0.0, 3.0))

        # Occasional spikes (mutation reconfiguration events)
        if self.rng.rand() < 0.05:
            rtt_delta += float(self.rng.uniform(5.0, 15.0))
            plr_delta += float(self.rng.uniform(10.0, 30.0))

        return base_rtt + rtt_delta, base_plr + plr_delta

    # -----------------------------------------------------------------------
    # Mininet-WiFi / Ryu placeholder hooks
    # -----------------------------------------------------------------------

    def mininet_push_flow_rules(self, routes: list):
        """
        TODO: Push mutated routes to Ryu SDN controller via REST API.

        Example:
            import requests
            for flow_id, route in enumerate(routes):
                rule = self._build_openflow_rule(flow_id, route)
                requests.post(
                    'http://ryu_controller:8080/stats/flowentry/add',
                    json=rule
                )
        """
        pass

    def mininet_update_ip_addresses(self, ip_assignment: np.ndarray):
        """
        TODO: Update ARP tables and flow rules for new IP spaces via Ryu.

        Example:
            for node_id, ip_space in enumerate(ip_assignment):
                new_ip = self._ip_space_to_cidr(ip_space, node_id)
                self.ryu_client.update_host_ip(node_id, new_ip)
        """
        pass

    def mininet_measure_rtt(self, src: int, dst: int) -> float:
        """
        TODO: Run Mininet ping to measure RTT.

        Example:
            result = self.net.ping(
                [self.net.get(f'h{src}'), self.net.get(f'h{dst}')],
                timeout=1
            )
            return result  # RTT in ms
        """
        return 0.0

    def mininet_measure_plr(self, src: int, dst: int) -> float:
        """
        TODO: Run iPerf to measure packet loss ratio.

        Example:
            srv = self.net.get(f'h{dst}')
            cli = self.net.get(f'h{src}')
            srv.cmd('iperf3 -s -D')
            out = cli.cmd(f'iperf3 -c {srv.IP()} -t 2 --json')
            data = json.loads(out)
            lost = data['end']['sum_received']['lost_packets']
            sent = data['end']['sum_sent']['packets']
            return lost / max(sent, 1) * 100
        """
        return 0.0


# ---------------------------------------------------------------------------
# DQN upper-layer agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    ε-greedy DQN for macro-action selection (upper layer, Algorithm 1 lines 12–18).
    """

    def __init__(
        self,
        state_dim:         int,
        n_macro_actions:   int   = N_MACRO_ACTIONS,
        hidden:            List[int] = None,
        lr:                float = 1e-3,
        gamma:             float = 0.99,
        epsilon:           float = 1.0,
        epsilon_min:       float = 0.05,
        epsilon_decay:     float = 0.9995,
        buffer_capacity:   int   = 20_000,
        batch_size:        int   = 64,
        target_update_freq: int  = 200,
        device:            str   = "cpu",
    ):
        self.gamma             = gamma
        self.epsilon           = epsilon
        self.epsilon_min       = epsilon_min
        self.epsilon_decay     = epsilon_decay
        self.batch_size        = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(
            device if torch.cuda.is_available() and device != "cpu" else "cpu"
        )

        self.q_net     = DQNNetwork(state_dim, n_macro_actions, hidden).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)
        self._update_count = 0

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy macro-action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(N_MACRO_ACTIONS)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(s).argmax(dim=-1).item())

    def push(self, t: UpperTransition):
        self.buffer.push(t)

    def update(self) -> Optional[float]:
        """One gradient-descent step — Algorithm 1 lines 35–36."""
        if len(self.buffer) < self.batch_size:
            return None

        batch   = self.buffer.sample(self.batch_size)
        states  = torch.FloatTensor(np.array([t.state        for t in batch])).to(self.device)
        actions = torch.LongTensor ([t.macro_action           for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t.reward                 for t in batch]).to(self.device)
        nstates = torch.FloatTensor(np.array([t.next_state   for t in batch])).to(self.device)
        dones   = torch.FloatTensor([float(t.done)            for t in batch]).to(self.device)

        # Current Q — Eq. (14)
        curr_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(nstates).max(dim=1)[0]
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.mse_loss(curr_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())


# ---------------------------------------------------------------------------
# PPO lower-layer agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    Clipped PPO for mutation-action selection (lower layer, Algorithm 1 lines 19–28).
    """

    def __init__(
        self,
        state_dim:   int,
        n_actions:   int,
        hidden:      List[int] = None,
        lr:          float     = 1e-4,
        gamma:       float     = 0.99,
        gae_lambda:  float     = 0.95,   # ξ in the paper
        clip_eps:    float     = 0.20,   # ε in the paper
        ppo_epochs:  int       = 4,
        minibatch:   int       = 32,
        device:      str       = "cpu",
    ):
        self.gamma      = gamma
        self.lam        = gae_lambda
        self.clip_eps   = clip_eps
        self.ppo_epochs = ppo_epochs
        self.minibatch  = minibatch
        self.device = torch.device(
            device if torch.cuda.is_available() and device != "cpu" else "cpu"
        )

        self.actor  = PPOActor (state_dim, n_actions, hidden).to(self.device)
        self.critic = PPOCritic(state_dim, hidden          ).to(self.device)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.rollout: List[LowerTransition] = []

    # -----------------------------------------------------------------------

    def _macro_oh(self, macro: int) -> torch.Tensor:
        oh = torch.zeros(N_MACRO_ACTIONS, device=self.device)
        oh[macro] = 1.0
        return oh

    def select_action(
        self, state: np.ndarray, macro_action: int
    ) -> Tuple[int, float]:
        """Sample from π_θ(A|S,O). Returns (action_idx, log_prob)."""
        self.actor.eval()
        s  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mo = self._macro_oh(macro_action).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(s, mo).squeeze(0)
        dist     = torch.distributions.Categorical(probs)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def get_value(self, state: np.ndarray, macro_action: int) -> float:
        self.critic.eval()
        s  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mo = self._macro_oh(macro_action).unsqueeze(0)
        with torch.no_grad():
            return float(self.critic(s, mo).item())

    def push(self, t: LowerTransition):
        self.rollout.append(t)

    def _compute_gae(self) -> Tuple[List[float], List[float]]:
        """GAE — Algorithm 1 lines 25–27."""
        adv, vtgt = [], []
        gae = 0.0
        for t in reversed(range(len(self.rollout))):
            tr = self.rollout[t]
            if t == len(self.rollout) - 1:
                next_val = 0.0
            else:
                ntr      = self.rollout[t + 1]
                next_val = self.get_value(ntr.state, ntr.macro_action)
            val   = self.get_value(tr.state, tr.macro_action)
            delta = tr.reward + self.gamma * next_val - val
            gae   = delta + self.gamma * self.lam * gae
            adv.insert(0, gae)
            vtgt.insert(0, gae + val)
        return adv, vtgt

    def update(self) -> Tuple[float, float]:
        """PPO update — Algorithm 1 lines 37–40."""
        if not self.rollout:
            return 0.0, 0.0

        adv_list, vtgt_list = self._compute_gae()

        # Normalise advantages
        adv_arr  = np.array(adv_list,  dtype=np.float32)
        vtgt_arr = np.array(vtgt_list, dtype=np.float32)
        adv_arr  = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        # Stash old log-probs (π_old)
        old_lps = torch.FloatTensor(
            [tr.log_prob for tr in self.rollout]
        ).to(self.device)

        states  = torch.FloatTensor(
            np.array([tr.state for tr in self.rollout])
        ).to(self.device)
        macros  = torch.stack([
            self._macro_oh(tr.macro_action) for tr in self.rollout
        ]).to(self.device)
        actions = torch.LongTensor(
            [tr.action for tr in self.rollout]
        ).to(self.device)
        advantages = torch.FloatTensor(adv_arr ).to(self.device)
        val_targets = torch.FloatTensor(vtgt_arr).to(self.device)

        total_al = total_cl = n_updates = 0.0

        for _ in range(self.ppo_epochs):
            idx = np.random.permutation(len(self.rollout))
            for start in range(0, len(self.rollout), self.minibatch):
                mb = torch.from_numpy(idx[start:start + self.minibatch]).long()
                if len(mb) == 0:
                    continue

                # Actor loss — Eqs. (16), (17)
                self.actor.train()
                probs   = self.actor(states[mb], macros[mb])
                dist    = torch.distributions.Categorical(probs)
                new_lps = dist.log_prob(actions[mb])
                ratio   = torch.exp(new_lps - old_lps[mb])
                adv_mb  = advantages[mb]
                al = -torch.min(
                    ratio * adv_mb,
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_mb,
                ).mean()

                self.actor_opt.zero_grad()
                al.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()

                # Critic loss
                self.critic.train()
                vals = self.critic(states[mb], macros[mb])
                cl   = F.mse_loss(vals, val_targets[mb])
                self.critic_opt.zero_grad()
                cl.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_opt.step()

                total_al += al.item()
                total_cl += cl.item()
                n_updates += 1

        self.rollout.clear()
        denom = max(n_updates, 1)
        return total_al / denom, total_cl / denom


# ---------------------------------------------------------------------------
# HDRL coordinator
# ---------------------------------------------------------------------------

class HDRLAgent:
    """
    Combines DQNAgent (upper) and PPOAgent (lower) into the full HDRL system
    described in Algorithm 1.

    State encoding:  one-hot expansion of per-node predicted events
                     → flat vector of length  n_nodes × n_event_types.
    """

    def __init__(
        self,
        n_nodes:          int  = 12,
        n_event_types:    int  = 3,
        n_feasible_actions: int = 20,
        dqn_hidden:       List[int] = None,
        ppo_hidden:       List[int] = None,
        dqn_lr:           float = 1e-3,
        ppo_lr:           float = 1e-4,
        gamma:            float = 0.99,
        gae_lambda:       float = 0.95,
        clip_eps:         float = 0.20,
        epsilon:          float = 1.0,
        device:           str   = "cpu",
    ):
        self.n_nodes            = n_nodes
        self.n_event_types      = n_event_types
        self.n_feasible_actions = n_feasible_actions
        self.state_dim          = n_nodes * n_event_types

        self.dqn = DQNAgent(
            state_dim       = self.state_dim,
            n_macro_actions = N_MACRO_ACTIONS,
            hidden          = dqn_hidden or [256],
            lr              = dqn_lr,
            gamma           = gamma,
            epsilon         = epsilon,
            device          = device,
        )
        self.ppo = PPOAgent(
            state_dim  = self.state_dim,
            n_actions  = n_feasible_actions,
            hidden     = ppo_hidden or [256, 256],
            lr         = ppo_lr,
            gamma      = gamma,
            gae_lambda = gae_lambda,
            clip_eps   = clip_eps,
            device     = device,
        )

    # -----------------------------------------------------------------------
    # State encoding
    # -----------------------------------------------------------------------

    def encode_state(self, predicted_events: np.ndarray) -> np.ndarray:
        """
        Convert per-node event predictions to a flat one-hot state vector.

        predicted_events : (n_nodes,) int array
        Returns          : (n_nodes × n_event_types,) float32 array
        """
        state = np.zeros(self.state_dim, dtype=np.float32)
        for i, ev in enumerate(predicted_events):
            if 0 <= int(ev) < self.n_event_types:
                state[i * self.n_event_types + int(ev)] = 1.0
        return state

    # -----------------------------------------------------------------------
    # Action selection
    # -----------------------------------------------------------------------

    def select_macro_action(self, state: np.ndarray) -> int:
        return self.dqn.select_action(state)

    def select_action(
        self, state: np.ndarray, macro_action: int
    ) -> Tuple[int, float]:
        return self.ppo.select_action(state, macro_action)

    # -----------------------------------------------------------------------
    # Buffer management
    # -----------------------------------------------------------------------

    def push_upper(self, t: UpperTransition):
        self.dqn.push(t)

    def push_lower(self, t: LowerTransition):
        self.ppo.push(t)

    # -----------------------------------------------------------------------
    # Network updates
    # -----------------------------------------------------------------------

    def update_upper(self) -> Optional[float]:
        return self.dqn.update()

    def update_lower(self) -> Tuple[float, float]:
        return self.ppo.update()

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, prefix: str):
        torch.save(self.dqn.q_net.state_dict(),    f"{prefix}_dqn.pt")
        torch.save(self.ppo.actor.state_dict(),     f"{prefix}_ppo_actor.pt")
        torch.save(self.ppo.critic.state_dict(),    f"{prefix}_ppo_critic.pt")

    def load(self, prefix: str):
        dev = next(self.dqn.q_net.parameters()).device
        self.dqn.q_net.load_state_dict(
            torch.load(f"{prefix}_dqn.pt",        map_location=dev, weights_only=True))
        self.ppo.actor.load_state_dict(
            torch.load(f"{prefix}_ppo_actor.pt",  map_location=dev, weights_only=True))
        self.ppo.critic.load_state_dict(
            torch.load(f"{prefix}_ppo_critic.pt", map_location=dev, weights_only=True))


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = HDRLAgent(n_nodes=12, n_event_types=3, n_feasible_actions=10)
    env   = NetworkEnvironment(n_nodes=12, n_switches=12)
    dummy_events = np.array([0, 1, 2, 0, 1, 0, 2, 1, 0, 0, 2, 1])
    state = agent.encode_state(dummy_events)
    macro = agent.select_macro_action(state)
    action_idx, lp = agent.select_action(state, macro)
    reward, sh, dh = env.compute_total_reward(macro, dummy_events, 0.3, 0.4)
    print(f"macro={MACRO_NAMES[macro]}  action={action_idx}  "
          f"reward={reward:.2f}  scan_hits={sh}  ddos_hits={dh}")
