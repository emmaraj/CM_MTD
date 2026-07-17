# CM-MTD: Collaborative Mutation-Based Moving Target Defense

Research-grade, dataset-agnostic reimplementation of the hierarchical
DRL framework from:

> Zhang et al., "When Moving Target Defense Meets Attack Prediction in
> Digital Twins: A Convolutional and Hierarchical Reinforcement Learning
> Approach," *IEEE JSAC*, vol. 41, no. 10, Oct. 2023.

LSTM attack prediction → hierarchical DRL (upper-layer DQN over
macro-actions, lower-layer PPO over micro-actions) → a simulated
Digital Twin Mobile Network that steps strictly through empirical
CICIDS-2017 traces (no synthetic/random network states).

See `docs/SETUP.md` for the Linux Mint install guide (Python/TensorFlow
now; Mininet-WiFi/Ryu for later). This document covers the project
itself: structure, config, how to run it, and — importantly — where the
implementation had to make a judgment call because the paper doesn't
fully specify something.

---

## Project structure

```
cm_mtd/
├── README.md                  <- this file
├── requirements.txt
├── config/
│   └── config.yaml            <- every hyperparameter, path, and TODO(paper-clarify) lives here
├── docs/
│   └── SETUP.md                <- Linux Mint setup guide
├── data/                       <- put your .npy files here (see below)
├── checkpoints/                <- saved LSTM/DQN/PPO models (created at runtime)
├── logs/                       <- training logs (created at runtime)
├── results/                     <- saved metrics + figures (created at runtime, see below)
│   └── figures/                 <- fig7/8/9/11 PNGs + Table II JSON from generate_figures.py
├── scripts/
│   ├── generate_dummy_data.py  <- smoke-test data generator (NOT for real results)
│   └── generate_figures.py     <- renders paper-comparable figures from saved training results
└── src/
    ├── __init__.py
    ├── config_parser.py        <- YAML loading, dataset loading, dim inference, class weights
    ├── environment.py          <- Gymnasium env: Waxman topology + reward Eq. 1-3 + DSR Eq. 19
    ├── models.py                <- LSTM predictor, DQN agent, PPO agent (all TF/Keras)
    └── main.py                  <- CLI entrypoint / training loop (Algorithm 1)
```

Nothing in `src/` hardcodes CICIDS-2017's shape. `Dataset.input_dim` and
`Dataset.num_classes` (in `config_parser.py`) are inferred from whatever
`.npy` arrays `config.yaml` points at, so swapping in a different
dataset later means editing `config.yaml`'s `data:` block — not touching
any Python.

---

## Getting your data in place

Run `cicids2017.ipynb` (already in your workspace) through to its final
cell, which writes:

```
X_train_env_state.npy   X_test_env_state.npy
y_train_env_state.npy   y_test_env_state.npy
```

Point `config/config.yaml`'s `data:` block at wherever those land, or
copy them into `cm_mtd/data/` (the default path). The notebook maps raw
CICIDS-2017 labels down to the paper's three event types — `Benign`,
`DoS/DDoS`, `Infiltration` — via feature-selection (zero-variance +
>0.90-correlation pruning) and a stratified 80/20 split, which is what
`config.yaml`'s `class_names` assumes.

For a fast sanity check without the real dataset:

```bash
python3 scripts/generate_dummy_data.py --config config/config.yaml
```

This writes small dummy arrays with CICIDS-2017's characteristic severe
class imbalance (Infiltration ≈ 0.1% of rows) so the class-weight-capping
logic actually gets exercised. It is a development aid — it uses
`np.random`, unlike the actual RL environment — never use it to draw
conclusions about CM-MTD's real performance.

---

## Running it

```bash
# 1. Pretrain the LSTM attack predictor (Section VI-A)
python3 -m src.main --config config/config.yaml --mode train_lstm

# 2. Train the hierarchical DQN/PPO agents (Algorithm 1)
python3 -m src.main --config config/config.yaml --mode train_rl

# or both in sequence:
python3 -m src.main --config config/config.yaml --mode all

# Evaluate a previously trained LSTM's fidelity (Eq. 18) without retraining:
python3 -m src.main --config config/config.yaml --mode evaluate
```

All three stages were run end-to-end against dummy data as part of
building this (LSTM trains, class weights get capped as designed, DQN/PPO
train jointly, DSR is computed per window, checkpoints save/reload
correctly). `config/config.yaml`'s defaults (`num_episodes: 10000`,
`steps_per_episode: 25`) match Table I of the paper.

**Expect roughly half a day for a full run on CPU** (~14 hours measured
at real config scale in testing; a GPU or faster CPU will do better).
Every model inference/training call in the hot RL loop is compiled via
`tf.function` specifically to make this tractable — plain eager
`model()` calls or `.predict()`/`.fit()` cost ~30ms of pure dispatch
overhead *per call regardless of input size*, which is irrelevant for
normal training but fatal when called millions of times across a full
run (T×K×M ≈ 6.25M environment steps at the defaults). If you profile a
change and see per-episode time creep back up, check whether it
introduced a varying-shape input to any of the `_infer_fn`/`_train_step_fn`
compiled functions in `models.py` — a changing shape forces TensorFlow to
retrace the graph, which reintroduces the same overhead per new shape
encountered.

---

## Generating figures to compare against the paper

`--mode train_lstm` and `--mode train_rl` now save their raw metrics to
`results/` (`lstm_history.json`, `lstm_confusion.npz`,
`rl_training_curves.npz`) instead of only printing them. Once you've run
training, render the figures:

```bash
python3 scripts/generate_figures.py --config config/config.yaml
```

This writes to `results/figures/`:

| File | Paper reference | What it shows |
|---|---|---|
| `fig7_lstm_fidelity_loss.png` | Fig. 7 | Prediction accuracy fidelity + loss vs. training epoch |
| `fig8_confusion_matrix.png` | Fig. 8 | Confusion matrix on the held-out test set |
| `table2_classification_report.json` | Table II | Per-class precision/recall/F1 |
| `fig9_dsr_over_training.png` | Fig. 9 | Defense Success Ratio (Eq. 19) over training episodes |
| `fig11_convergence.png` | Fig. 11 | Upper-layer (DQN) and lower-layer (PPO) reward convergence |

Two honesty notes, both called out directly in the figures themselves so
they don't get lost if you screenshot one in isolation:

- **Fig. 9** in the paper compares CM-MTD against two baselines
  (RRT+FRVM, DQ-RM+FRVM), averaged over 5 runs with error shadows. Neither
  baseline is implemented here, so `fig9_dsr_over_training.png` is a
  single CM-MTD run with no comparison curve. If you want the baselines
  built out for an apples-to-apples comparison, that's a well-scoped
  follow-up — say the word.
- **Fig. 11** is similarly a single run (raw + rolling-mean smoothed),
  not a 5-run average with error shadow.
- **Fig. 10** (RTT / packet loss) is **not generated at all**. The paper
  measures this with iPerf against a live Mininet-WiFi topology;
  `environment.py` is a probabilistic reward simulation with no
  packet-level signal to plot. `generate_figures.py` prints exactly this
  explanation rather than fabricating plausible-looking numbers. This
  becomes possible once the real Mininet-WiFi/os-ken integration
  (`docs/SETUP.md` Part 2) is wired in.

---

## How the pieces map to the paper

| Component | File | Paper reference |
|---|---|---|
| LSTM attack predictor | `models.py::LSTMAttackPredictor` | Section VI-A, Eq. 11-12, Fig. 4 |
| Prediction fidelity | `models.py::compute_fidelity` | Eq. 18 |
| Waxman topology | `environment.py::build_waxman_topology` | Section III-A, Table I |
| SMDP state / reward | `environment.py::DigitalTwinNetworkEnv` | Section III-C, Eq. 1-3 |
| LSTM state wrapper | `environment.py::LSTMStatePredictionWrapper` | Fig. 5 (Env → LSTM → state) |
| Defense Success Ratio | `environment.py::EpisodeStats.dsr` | Eq. 19 |
| Upper-layer DQN | `models.py::DQNAgent` | Section VI-B, Eq. 13-14 |
| Lower-layer PPO | `models.py::PPOAgent` | Section VI-B, Eq. 15-17 |
| Training loop | `main.py::train_rl` | Algorithm 1 |

---

## Where the paper doesn't fully specify something

Three places are flagged `TODO(paper-clarify)` directly in
`config.yaml`, with the reasoning for the placeholder value inline as a
comment:

1. **Reward coefficients** `α1, α2, C` (Eq. 1) and `γ1, γ2` (Eq. 2) —
   never given numeric values anywhere in the paper or Table I. Current
   placeholders make defense reward dominate the resource penalty by
   roughly 10-20x, a common ratio in MTD-RL literature, but they are
   *not* the authors' values (which don't appear to be published).
2. **Resource consumption model** `e^a_i`, `e^r_y` (Eq. 2) — the paper
   defines `W_t` in terms of these per-node/per-flow costs but never
   states their units or how they scale with pool size / route length.
   The current model (`resource_model: linear_per_mutation`) charges a
   flat cost per node re-addressed / flow rerouted *this step relative to
   last step*, which rewards temporal stability the way the paper's
   narrative ("reduce deployment time of HAM and RM") implies, without
   inventing numbers the paper doesn't provide.
3. **Fidelity aggregation window** (Eq. 18) — the formula itself is
   unambiguous, but the paper doesn't say whether `|N|` is computed
   per-timestep, per-episode, or cumulatively across the whole evaluation
   run. This affects how closely Fig. 7's curves are reproducible.
   Currently computed per full evaluation pass over the test set;
   `config.yaml`'s `fidelity.aggregation` documents the choice.

If you have the authors' code or supplementary material with concrete
values for any of these, updating `config.yaml` is all that's needed —
nothing in `src/` treats these as hardcoded.

## Deliberate simplifications (not paper ambiguities — engineering choices)

- **No SMT constraint solver.** Section V formalizes IP-pool/route
  feasibility (Eq. 4-9) as a satisfiability problem. Implementing a full
  SMT solver was out of scope for this pass; instead, feasibility is
  guaranteed *by construction* — routes are drawn from precomputed valid
  simple paths between each flow's fixed endpoints, and IP pool choices
  are bounded categorical picks. If/when you want the real constraint
  solving (e.g. flow-table capacity, Eq. 9), that plugs in as an action
  mask inside `PPOAgent.select_action`.
- **PPO actor architecture.** The paper's own complexity analysis
  describes a single-scalar Gaussian actor ("parameters of a normal
  distribution with 2 neurons"), which can't directly parameterize our
  genuinely multi-dimensional, discrete micro-action (one IP-pool index
  per node, one route index per flow). `PPOAgent` uses a multi-head
  categorical policy instead, trained with the same clipped-surrogate
  objective (Eq. 16). Documented in `models.py`'s `PPOAgent` docstring.
- **Attacker targeting is deterministic, not random.** Per the "no
  synthetic data" requirement, `environment.py` never calls `np.random`
  inside `step()`. Which IP pool / switch an attacker is "aiming at" is
  derived via a hash of that row's real feature vector, so outcomes are
  fully determined by the empirical trace plus the agent's action — see
  `environment.py`'s module docstring for the full reasoning.
- **CICIDS-2017 event → node mapping.** The dataset has one label per
  traffic sample, not one label per network node. Samples are assigned to
  the `n` simulated nodes deterministically round-robin over time
  (`_row_indices_for_timestep`) rather than by any per-node semantic
  meaning in the original CSVs (CICIDS-2017 doesn't carry a node-ID
  field). This is the standard way prior MTD-RL work adapts a flow-level
  IDS dataset into a per-node SMDP state.

---

## Known failure modes this implementation guards against

- **LSTM training collapse on Infiltration.** CICIDS-2017's Infiltration
  class is a tiny fraction of rows. Naive `class_weight="balanced"`
  produces a weight large enough to blow up gradients. `class_weight_strategy`
  guards against this (see `config_parser.compute_class_weights`) -- but
  see the class-weight-overcorrection bullet below before assuming
  `balanced_capped` is the right choice for every dataset.
- **OOM/segfault loading the full label set.** `data.max_samples` (default
  200,000) plus a deterministic `stratified_head` truncation keeps the RL
  environment's memory bounded without introducing randomness into what
  gets loaded. This truncation applies ONLY to the environment now -- see
  the next bullet for why.
- **Majority-class collapse from a shared truncation view.** `data.max_samples`
  originally truncated `X_train`/`y_train` once, at dataset-load time, and
  that same truncated array was used both for the RL environment's
  row-cycling *and* the LSTM's sliding-window training. `stratified_head`
  truncation preserves relative row order but reorders/subsamples which
  rows survive, which fragments the genuine row-to-row temporal adjacency
  LSTM sequence windows depend on -- and can produce a model that
  collapses to predicting only the majority class despite a supposedly
  learnable minority class (verified: `class_weight` itself works
  correctly, confirmed via an isolated reproduction). Fixed by decoupling
  the two: `config_parser.load_dataset()` no longer truncates at all, so
  LSTM training always sees the full, contiguous dataset;
  `environment.py`'s `DigitalTwinNetworkEnv` builds its own separately
  truncated row cache via `config_parser.build_env_row_cache()` for its
  memory-bounded row-cycling, where order/continuity doesn't matter the
  same way (rows are already assigned to nodes via an artificial
  round-robin, not genuine per-node temporal meaning).
- **Validation accuracy swinging wildly epoch to epoch.** `keras.Model.fit`'s
  `validation_split` slices a contiguous chunk off the *end* of whatever
  array you pass it -- it does not shuffle. Once LSTM training used the
  full, genuinely time-ordered dataset (previous bullet), that end slice
  became a real contiguous chunk of the original CICIDS-2017 file, which
  can be dominated by whichever attack burst happened to land there.
  Reproduced directly: an artificial dataset with an all-one-class tail
  gave a flat `val_accuracy = 0.0000` for every epoch; after the fix,
  smooth convergence to 0.88. Fixed in `LSTMAttackPredictor.fit()` by
  shuffling at the *window* level (each window's own internal
  `sequence_length` ordering is untouched -- only which windows land in
  train vs. validation is randomized) before calling `model.fit()`.
- **Class-weight overcorrection.** `balanced_capped` (inverse-frequency,
  ratio-clipped) was the original default, calibrated against
  Infiltration's extreme rarity. On real data it overcorrected the
  *other* class: DoS/DDoS is a substantial 14% of rows, not a tiny
  minority, and a ~6x weight boost was enough to flip the model from
  "always predict Benign" to "mostly predict DoS/DDoS" -- net accuracy
  *below* the trivial majority-class baseline. Default changed to
  `balanced_sqrt` (sqrt-dampened inverse-frequency, ~2.4x instead of ~6x
  for DoS/DDoS:Benign on real proportions) -- see
  `config_parser.compute_class_weights` for the full comparison.
  `balanced_capped` is still available if `balanced_sqrt` undercorrects
  on your data.
- **TF/PyTorch GPU contention.** This iteration is pure TensorFlow/Keras
  end-to-end (LSTM, DQN, *and* PPO) specifically to avoid mixing runtimes
  on the same GPU. See `docs/SETUP.md` §1.3 for the history.

---

## Next steps (not yet done)

- Point `config.yaml` at the real `cicids2017.ipynb` output and run a full
  10,000-episode training pass to reproduce Fig. 7-11.
- Replace `DigitalTwinNetworkEnv`'s probabilistic reward model with a real
  Mininet-WiFi + `os-ken`/Ryu emulation once `docs/SETUP.md` Part 2 is
  set up (the `step()` contract is designed to make this a contained
  change).
- If the authors' concrete reward-coefficient values ever surface, update
  `config.yaml` and re-run — no code changes needed.
