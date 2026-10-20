#!/usr/bin/env bash
# Rerun every scenario-based experiment with MEASURED outdoor PM2.5 (U.S. Embassy Dhaka, data/outdoor_dhaka_*.csv;
# ScenarioConfig.outdoor = "dhaka" is the default). One worker pool at a time.
# Resumable: every finished step leaves logs/rerun_markers/<step>.done and is skipped on the next start; a failed step
# is retried twice (stages e2, e4, robust2 and mc also resume from their own saved progress).
# Results that do not use scenarios (e6 solver checks and timing, e10 identification, platform calibration) are kept.
cd "$(dirname "$0")"
PY="/c/Users/Asus ROG Strix G16/AppData/Local/Python/bin/python.exe"
export AQ_WORKERS=${AQ_WORKERS:-14}
LOG=../../logs/results_rerun_real_outdoor.log
MK=../../logs/rerun_markers
mkdir -p "$MK"
step() {   # step <name> <script> [stages...]
  local name=$1; shift
  if [ -f "$MK/$name.done" ]; then echo "$(date +%T) skip $name (done)" >> "$LOG"; return; fi
  for attempt in 1 2 3; do
    echo "$(date +%T) >>> $name (attempt $attempt): $*" >> "$LOG"
    if "$PY" -u "$@" >> "$LOG" 2>&1; then touch "$MK/$name.done"; return; fi
    echo "$(date +%T) !!! $name failed (attempt $attempt)" >> "$LOG"
    sleep 30
  done
}
echo "$(date +%T) rerun (resumable) started with AQ_WORKERS=$AQ_WORKERS" >> "$LOG"
for s in tune e1 e1b e3 e2 e4 e5 e7 e8 e9; do step "run_all_$s" run_all.py "$s"; done
for s in tune test refine front; do step "e11_$s" e11_baselines.py "$s"; done
for s in base2 pisd2 budget budgetrcl e7roll robust2 e4b mc; do step "e12_$s" e12_revision.py "$s"; done
for s in rollout invariance; do step "e13_$s" e13_theory_checks.py "$s"; done
step e14 e14_forecast_skill.py
step diag_mech diag_mech.py
for s in sweep main budget lambda lambda1 cert budgetsweep certlp milpcheck; do step "e15_$s" e15_leverage.py "$s"; done
step blocks test_blocks.py
step e16 e16_sampled.py
echo "$(date +%T) ALL STAGES DONE" >> "$LOG"
