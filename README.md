# CM-MTD: Collaborative Mutation-based Moving Target Defense

> **Reproducible Research Implementation**  
> IEEE JSAC 2023 — *"When Moving Target Defense Meets Attack Prediction in Digital Twins: A Convolutional and Hierarchical Reinforcement Learning Approach"*  
> Zhang et al., 2023

---

## Overview

This repository provides a complete, PhD-grade research implementation of the CM-MTD framework, featuring:

- **LSTM attack prediction** on CICIDS-2017 (or synthetic data)
- **Hierarchical Deep RL**: Upper-layer DQN (macro-action) + Lower-layer PPO (mutation actions)
- **All 5 paper baselines**: STATIC, HAM-only, RM-only, RRT+FRVM, DQN-RM+FRVM
- **Full experimental protocol**: 5 seeds, mean ± std, 95% CI
- **Statistical analysis**: paired t-test, Wilcoxon, Cohen's d
- **All paper figures reproduced**: Figs 7–11 as PNG / PDF / SVG

---

## Quick Start

### 1. Installation

```bash
# Python 3.12 recommended
git clone <this-repo>
cd project/

python -m venv venv
source venv/bin/activate          # Linux/Mac
# or: venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. Smoke Test (no data download needed)

```bash
# Quick end-to-end test with synthetic data (~5 minutes)
python main.py --synthetic --n-episodes 200 --n-eval 50
```

### 3. Full Pipeline with Synthetic Data

```bash
python main.py --synthetic --all-seeds --n-episodes 10000 --n-eval 1000
```

### 4. Full Pipeline with Real CICIDS-2017

```bash
# 1. Download dataset from: https://www.unb.ca/cic/datasets/ids-2017.html
# 2. Place all 8 CSV files in data/cicids2017/
# 3. Run:
python main.py --all-seeds
```

---

## Dataset: CICIDS-2017

| Day       | Attack Types |
|-----------|-------------|
| Monday    | BENIGN only |
| Tuesday   | FTP-Patator, SSH-Patator (→ BruteForce) |
| Wednesday | DoS Hulk, GoldenEye, Slowhttptest, Slowloris, Heartbleed (→ DoS) |
| Thursday  | Web Attack – BruteForce/XSS/SQLi (→ WebAttack), Infiltration |
| Friday    | Bot, PortScan, DDoS |

**Download:** https://www.unb.ca/cic/datasets/ids-2017.html  
Place all `.csv` files in `data/cicids2017/`

**Unified labels used:**

| Class | ID | Coverage |
|-------|----|---------|
| BENIGN | 0 | Normal traffic |
| DoS | 1 | DoS Hulk, GoldenEye, Slowhttptest, Slowloris, Heartbleed |
| DDoS | 2 | DDoS |
| PortScan | 3 | PortScan |
| Infiltration | 4 | Infiltration |
| Bot | 5 | Bot |
| BruteForce | 6 | FTP-Patator, SSH-Patator |
| WebAttack | 7 | Brute Force, XSS, SQL Injection |

---

## Project Structure

```
project/
├── config/
│   ├── config.yaml          # All hyperparameters (Table I of paper)
│   └── __init__.py
├── data/
│   ├── cicids2017/          # Place CSV files here
│   ├── processed/           # Preprocessed NumPy arrays (auto-created)
│   └── synthetic/           # Synthetic data cache
├── datasets/
│   ├── cicids2017_loader.py # CSV loading + label mapping
│   ├── preprocessor.py      # Clean, normalize, split
│   └── sequence_builder.py  # Sliding-window event sequences (Eq. 12)
├── models/
│   ├── lstm_predictor.py    # LSTMNet (Eq. 11, Section VI-A)
│   └── networks.py          # DuelingQNetwork, PPOActorCritic (Table I)
├── environment/
│   ├── dtmn_env.py          # Gymnasium DTMN environment (SMDP)
│   ├── network_model.py     # SDN-IoT network simulation (Section III-A)
│   └── reward_functions.py  # R_d, R_c, R_total (Eq. 1–3)
├── agents/
│   ├── upper_layer_dqn.py   # DQN for macro-action (Eq. 13–14)
│   ├── lower_layer_ppo.py   # PPO for mutation actions (Eq. 15–17)
│   ├── hdrl_agent.py        # Full Algorithm 1 implementation
│   ├── baselines.py         # All 5 comparison baselines
│   └── replay_buffer.py     # B₁ (DQN) and B₂ (PPO) buffers
├── visualizations/
│   ├── figure_generator.py  # Figs 7–11 reproduction
│   └── plot_utils.py        # Publication-quality styling
├── utils/
│   ├── logger.py            # Structured logging
│   ├── metrics.py           # DSR (Eq. 19), Fidelity (Eq. 18)
│   ├── seed_utils.py        # Full reproducibility
│   └── statistical_analysis.py  # t-test, Wilcoxon, Cohen's d
├── results/
│   ├── figures/{png,pdf,svg}/   # All paper figures
│   ├── metrics/                  # JSON metrics + DSR arrays
│   └── checkpoints/              # Per-seed training artifacts
├── checkpoints/             # Saved model weights
├── logs/                    # Training logs + TensorBoard events
├── notebooks/               # Jupyter analysis notebooks (add manually)
│
├── train_lstm.py            # Stage 2: LSTM training
├── train_hdrl.py            # Stage 3: HDRL training (Algorithm 1)
├── evaluate.py              # Stage 4: Baseline comparison
├── reproduce_all_figures.py # Stage 5: Figure generation
└── main.py                  # Complete pipeline orchestrator
```

---

## Running Individual Stages

```bash
# Stage 1: Validate data
python main.py --stage preprocess

# Stage 2: Train LSTM only
python train_lstm.py --synthetic --seed 42

# Stage 3: Train HDRL (CM-MTD)
python train_hdrl.py --synthetic --seed 42 --n-episodes 5000

# Stage 4: Evaluate vs baselines
python evaluate.py --synthetic --n-eval 500

# Stage 5: Reproduce all figures
python reproduce_all_figures.py --synthetic-curves
```

---

## Paper Results vs Expected Implementation

| Metric | Paper (Table I / Fig 9) | Expected Range |
|--------|------------------------|----------------|
| DSR — Direct DDoS + scan | 98.2% | 96–99% |
| DSR — Crossfire DDoS + scan | 97% | 95–98% |
| DSR — CICIDS-2017 | 95% | 93–97% |
| LSTM Fidelity — DDoS+scan | 92% | 90–94% |
| LSTM Fidelity — CICIDS-2017 | 83% | 80–86% |
| RTT overhead | +1–3 ms vs baseline | matches |
| Upper layer convergence | ~6000 episodes | ~5000–8000 |
| Lower layer convergence | ~20000 steps | ~15000–25000 |

---

## Key Configuration (config/config.yaml)

All Table I parameters are in `config/config.yaml`:

```yaml
training:
  n_episodes: 10000       # M
  n_steps_per_episode: 25 # T
  K_steps: 5              # K (inner steps per macro-action)

dqn:
  hidden_layers: [256]    # Table I
  learning_rate: 1.0e-3   # Table I
  gamma: 0.99
  epsilon_start: 1.0

ppo:
  hidden_layers: [256, 256]  # Table I
  learning_rate: 1.0e-4      # Table I
  gae_lambda: 0.95           # λ = ξ in paper
  clip_range: 0.2            # ε
```

---

## Extending the Framework

### Custom Attack Sequences

```python
from environment.dtmn_env import DTMNEnvironment
import numpy as np

# Your attack sequence: integer array of attack class IDs
my_sequence = np.array([0, 0, 3, 3, 2, 0, 1, ...])  # shape [T]
env = DTMNEnvironment(..., attack_sequence=my_sequence)
```

### Adding a New Baseline

```python
from agents.baselines import BaselineAgent

class MyBaseline(BaselineAgent):
    name = "MY_METHOD"
    def select_action(self, state: np.ndarray) -> int:
        return 3  # always HAM+RM
```

### Custom Reward Coefficients

Edit `config/config.yaml`:
```yaml
reward:
  alpha1: 2.0   # higher scanning penalty
  alpha2: 1.5   # higher DDoS switch penalty
  C_defense: 15.0
  gamma1: 0.3
  gamma2: 0.7
```

---

## Citation

```bibtex
@article{zhang2023cm_mtd,
  title={When Moving Target Defense Meets Attack Prediction in Digital Twins:
         A Convolutional and Hierarchical Reinforcement Learning Approach},
  author={Zhang, Tao and Xu, Changqiao and Lian, Yibo and Tian, Haijiang
          and Kang, Jiawen and Kuang, Xiaohui and Niyato, Dusit},
  journal={IEEE Journal on Selected Areas in Communications},
  volume={41},
  number={10},
  pages={3293--3305},
  year={2023},
  doi={10.1109/JSAC.2023.3310072}
}
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: No CICIDS2017 CSV files` | Add `--synthetic` flag or download dataset |
| CUDA out of memory | Add `--device cpu` |
| TensorFlow import error | Run `pip install tensorflow>=2.15` |
| Slow training | Reduce `--n-episodes 500` for testing |
| Figures look wrong | Run `python reproduce_all_figures.py --synthetic-curves` |

---

*Tested on Ubuntu 24, Python 3.12, NVIDIA GPU with CUDA 12.x*
