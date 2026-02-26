#!/bin/bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LOG="$ROOT/experiments/run_diagnostics.log"
mkdir -p experiments/diag1 experiments/diag2 experiments/diag3 experiments/diag4 experiments/diag5

stamp() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
fail()  { stamp "FAILED: $*"; exit 1; }

stamp "=========================================="
stamp "Diagnostics pipeline starting  PID=$$"
stamp "Root: $ROOT"
stamp "=========================================="

stamp "Starting Diagnostic 1 (convergence test, T*=0.36 and 0.45, 500K moves)..."
python scripts/diag1_long_convergence.py 2>&1 | tee experiments/diag1.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then fail "Diagnostic 1"; fi
stamp "Diagnostic 1 DONE"

stamp "Starting Diagnostic 2 (equilibrium grids)..."
for T in 0.36 0.45; do
    stamp "  diag2 T*=$T"
    python scripts/diag2_equilibrium_grid.py --temperature "$T" 2>&1 | tee -a experiments/diag2.log
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then fail "Diagnostic 2 T*=$T"; fi
done
stamp "Diagnostic 2 DONE"

stamp "Starting Diagnostic 3 (OP histograms)..."
for T in 0.36 0.45; do
    stamp "  diag3 T*=$T"
    python scripts/diag3_op_histogram.py --temperature "$T" 2>&1 | tee -a experiments/diag3.log
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then fail "Diagnostic 3 T*=$T"; fi
done
stamp "Diagnostic 3 DONE"

stamp "Starting Diagnostic 4 (temperature panel, 6 temps x 500K moves)..."
python scripts/diag4_temp_panel.py 2>&1 | tee experiments/diag4.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then fail "Diagnostic 4"; fi
stamp "Diagnostic 4 DONE"

stamp "Starting Diagnostic 5 (mixing time vs temperature)..."
python scripts/diag5_mixing_time.py 2>&1 | tee experiments/diag5.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then fail "Diagnostic 5"; fi
stamp "Diagnostic 5 DONE"

stamp "=========================================="
stamp "ALL DIAGNOSTICS COMPLETE"
stamp "  experiments/diag1/T0.36_timeseries.png"
stamp "  experiments/diag1/T0.45_timeseries.png"
stamp "  experiments/diag1/T0.36_filmstrip.png"
stamp "  experiments/diag1/T0.45_filmstrip.png"
stamp "  experiments/diag2/T0.36_grid.png"
stamp "  experiments/diag2/T0.45_grid.png"
stamp "  experiments/diag3/T0.36_histograms.png"
stamp "  experiments/diag3/T0.45_histograms.png"
stamp "  experiments/diag4/scatter_grid.png"
stamp "  experiments/diag5/mixing_time.png"
stamp "=========================================="
