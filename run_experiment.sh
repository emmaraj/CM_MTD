#!/usr/bin/env bash
# =============================================================================
# run_experiment.sh
# =============================================================================
# Runs the full CM-MTD pipeline in the correct order:
#   1. Sanity-check each module (import + quick self-test)
#   2. Train & evaluate  — produces CSV / JSON logs
#   3. Visualize results — reads logs and writes PNG figures
#
# Usage (defaults — fast demo run):
#   bash run_experiment.sh
#
# Usage (paper-scale run):
#   bash run_experiment.sh --n_episodes 10000 --n_timesteps 10000 \
#                          --lstm_episodes 40  --device cpu
#
# Optional arguments are forwarded to train_and_eval.py.
# The visualize step reads the log directory produced by training.
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
N_EPISODES=500          # outer RL episodes per dataset mode
N_TIMESTEPS=5000        # simulation events per mode
LSTM_EPISODES=15        # LSTM training epochs per mode
LOG_DIR="./logs"
MODEL_DIR="./models"
FIGURE_DIR="./figures"
DEVICE="cpu"
SEED=42
CICIDS_PATH=""          # leave blank → uses synthetic CICIDS-like data

# ── Parse arguments (key=value or --key value style) ─────────────────────────
for arg in "$@"; do
  case $arg in
    --n_episodes=*)    N_EPISODES="${arg#*=}"  ;;
    --n_timesteps=*)   N_TIMESTEPS="${arg#*=}" ;;
    --lstm_episodes=*) LSTM_EPISODES="${arg#*=}";;
    --log_dir=*)       LOG_DIR="${arg#*=}"     ;;
    --model_dir=*)     MODEL_DIR="${arg#*=}"   ;;
    --figure_dir=*)    FIGURE_DIR="${arg#*=}"  ;;
    --device=*)        DEVICE="${arg#*=}"      ;;
    --seed=*)          SEED="${arg#*=}"        ;;
    --cicids_path=*)   CICIDS_PATH="${arg#*=}" ;;
    *) echo "[WARN] Unknown argument: $arg" ;;
  esac
done

# ── Locate the directory containing this script ───────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Resolve Python ────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
  echo "[ERROR] Python interpreter not found. Set PYTHON= env var or install Python 3."
  exit 1
fi
echo "[INFO] Using Python: $($PYTHON --version)"

# ── Colour helpers ────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
  YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
else
  RED=''; GREEN=''; CYAN=''; YELLOW=''; BOLD=''; NC=''
fi

banner() { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════${NC}"; \
           echo -e "${BOLD}${CYAN}  $1${NC}"; \
           echo -e "${BOLD}${CYAN}══════════════════════════════════════${NC}"; }
ok()     { echo -e "${GREEN}[✓]${NC} $1"; }
warn()   { echo -e "${YELLOW}[!]${NC} $1"; }
fail()   { echo -e "${RED}[✗] $1${NC}"; exit 1; }

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p "$LOG_DIR" "$MODEL_DIR" "$FIGURE_DIR"

# =============================================================================
# STEP 0 — Dependency check
# =============================================================================
banner "Step 0 — Checking dependencies"

REQUIRED="torch numpy pandas scikit-learn z3 matplotlib seaborn scipy"
MISSING=""
for pkg in $REQUIRED; do
  mod="${pkg//-/_}"          # e.g. scikit-learn → scikit_learn
  mod="${mod//z3/z3}"        # keep z3 as-is
  if $PYTHON -c "import $mod" 2>/dev/null; then
    ok "$pkg"
  else
    warn "$pkg not found — attempting pip install …"
    if pip install "$pkg" --break-system-packages -q; then
      ok "$pkg installed"
    else
      MISSING="$MISSING $pkg"
    fi
  fi
done

if [[ -n "$MISSING" ]]; then
  fail "Could not install:$MISSING  Aborting."
fi

# =============================================================================
# STEP 1 — Module sanity checks
# =============================================================================
banner "Step 1 — Module sanity checks"

run_check() {
  local label="$1"; local script="$2"
  if $PYTHON "$script" 2>&1 | tail -5; then
    ok "$label passed"
  else
    fail "$label failed — check output above."
  fi
}

run_check "data_loader"    "data_loader.py"
run_check "lstm_predictor" "lstm_predictor.py"
run_check "smt_constraints" "smt_constraints.py"
run_check "hdrl_agent"     "hdrl_agent.py"

# =============================================================================
# STEP 2 — Training & evaluation
# =============================================================================
banner "Step 2 — Training & evaluation"

echo "Parameters:"
echo "  n_episodes    = $N_EPISODES"
echo "  n_timesteps   = $N_TIMESTEPS"
echo "  lstm_episodes = $LSTM_EPISODES"
echo "  device        = $DEVICE"
echo "  log_dir       = $LOG_DIR"
echo "  model_dir     = $MODEL_DIR"
echo "  seed          = $SEED"
[[ -n "$CICIDS_PATH" ]] && echo "  cicids_path   = $CICIDS_PATH"

TRAIN_ARGS=(
  --n_episodes    "$N_EPISODES"
  --n_timesteps   "$N_TIMESTEPS"
  --lstm_episodes "$LSTM_EPISODES"
  --log_dir       "$LOG_DIR"
  --model_dir     "$MODEL_DIR"
  --device        "$DEVICE"
  --seed          "$SEED"
)
[[ -n "$CICIDS_PATH" ]] && TRAIN_ARGS+=(--cicids_path "$CICIDS_PATH")

echo ""
START_TIME=$SECONDS
$PYTHON train_and_eval.py "${TRAIN_ARGS[@]}" || fail "train_and_eval.py exited with error."
ELAPSED=$(( SECONDS - START_TIME ))
ok "Training complete in ${ELAPSED}s"

# =============================================================================
# STEP 3 — Verify expected log files
# =============================================================================
banner "Step 3 — Verifying log files"

REQUIRED_LOGS=(
  "$LOG_DIR/defense_log.csv"
  "$LOG_DIR/convergence_log.csv"
  "$LOG_DIR/network_perf_log.csv"
  "$LOG_DIR/confusion_matrices.json"
  "$LOG_DIR/final_summary.json"
)

for f in "${REQUIRED_LOGS[@]}"; do
  if [[ -f "$f" ]]; then
    ROWS=$(wc -l < "$f" 2>/dev/null || echo "?")
    ok "$f  (${ROWS} lines)"
  else
    warn "Expected log file not found: $f"
  fi
done

# LSTM training logs (one per mode)
for mode in direct_ddos crossfire_ddos cicids2017; do
  f="$LOG_DIR/lstm_training_log_${mode}.csv"
  if [[ -f "$f" ]]; then
    ok "$f"
  else
    warn "Missing: $f"
  fi
done

# =============================================================================
# STEP 4 — Visualization
# =============================================================================
banner "Step 4 — Generating figures"

$PYTHON visualize_results.py \
  --log_dir    "$LOG_DIR" \
  --output_dir "$FIGURE_DIR" \
  || fail "visualize_results.py exited with error."

ok "Figures written to $FIGURE_DIR/"

# List figures produced
echo ""
echo "Figures:"
find "$FIGURE_DIR" -name "*.png" -printf "  %f\n" 2>/dev/null \
  || ls "$FIGURE_DIR"/*.png 2>/dev/null | xargs -I{} basename {}

# =============================================================================
# DONE
# =============================================================================
banner "Experiment complete"
echo -e "${GREEN}Outputs:${NC}"
echo "  Logs    → $LOG_DIR/"
echo "  Models  → $MODEL_DIR/"
echo "  Figures → $FIGURE_DIR/"
echo ""
echo -e "${GREEN}Quick re-visualize (no retraining):${NC}"
echo "  python visualize_results.py --log_dir $LOG_DIR --output_dir $FIGURE_DIR"
echo ""
echo -e "${GREEN}Paper-scale run:${NC}"
echo "  bash run_experiment.sh --n_episodes 10000 --n_timesteps 10000 --lstm_episodes 40"
