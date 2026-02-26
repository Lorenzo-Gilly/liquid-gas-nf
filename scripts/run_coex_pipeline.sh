#!/bin/bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LOG="$ROOT/experiments/coex_pipeline.log"
mkdir -p experiments/coex_comparison checkpoints/coex data/coex

stamp() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
fail()  { stamp "FAILED: $*"; exit 1; }

stamp "=========================================="
stamp "Coexistence NF pipeline starting  PID=$$"
stamp "  Base T*=0.50  Target T*=0.36  N=128  rho*=0.30"
stamp "=========================================="

stamp "Step 1/3: Generate training data (base T*=0.50 + target T*=0.36)..."
python coex_train.py --generate-data 2>&1 | tee experiments/coex_data_gen.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then fail "Data generation"; fi
stamp "Step 1/3 DONE"

stamp "Step 2/3: Train normalising flow (200 epochs)..."
python coex_train.py 2>&1 | tee experiments/coex_train.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then fail "Training"; fi
stamp "Step 2/3 DONE  params saved to checkpoints/coex/params_final.pkl"

stamp "Step 3/3: Run flow-augmented MCMC comparison..."
python scripts/flow_mcmc_comparison.py 2>&1 | tee experiments/coex_comparison.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then fail "Comparison"; fi
stamp "Step 3/3 DONE"

stamp "=========================================="
stamp "ALL STEPS COMPLETE"
stamp "Key outputs:"
stamp "  checkpoints/coex/params_final.pkl      — trained flow"
stamp "  checkpoints/coex/train_log.txt          — epoch metrics"
stamp "  experiments/coex_comparison/timeseries_no_flow.png"
stamp "  experiments/coex_comparison/timeseries_with_flow.png"
stamp "  experiments/coex_comparison/mixing_time_comparison.png"
stamp "  experiments/coex_comparison/results_summary.txt"
stamp "=========================================="
