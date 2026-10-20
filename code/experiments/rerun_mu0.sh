#!/usr/bin/env bash
# Second pass after the measured-outdoor rerun: the rollout costate law now runs without the soft-bound penalty
# (mu = 0, RolloutCostate(penalty=False) default, the setting of Theorem 1). Reruns every stage that contains CL-r.
# Resumable like rerun_real_outdoor.sh (markers mu0_<step>.done).
cd "$(dirname "$0")"
PY="/c/Users/Asus ROG Strix G16/AppData/Local/Python/bin/python.exe"
export AQ_WORKERS=${AQ_WORKERS:-14}
LOG=../../logs/results_rerun_real_outdoor.log
MK=../../logs/rerun_markers
mkdir -p "$MK"
step() {
  local name=$1; shift
  if [ -f "$MK/$name.done" ]; then echo "$(date +%T) skip $name (done)" >> "$LOG"; return; fi
  for attempt in 1 2 3; do
    echo "$(date +%T) >>> $name (attempt $attempt): $*" >> "$LOG"
    if "$PY" -u "$@" >> "$LOG" 2>&1; then touch "$MK/$name.done"; return; fi
    echo "$(date +%T) !!! $name failed (attempt $attempt)" >> "$LOG"
    sleep 30
  done
}
if [ ! -f "$MK/mu0_cleared.done" ]; then                      # CL-r results with the penalty must not be resumed
  rm -f ../../results/e12/robust2_raw.csv ../../results/e12/e4b_raw.csv ../../results/e12/mc_raw.csv
  touch "$MK/mu0_cleared.done"
fi
echo "$(date +%T) mu0 pass started" >> "$LOG"
for s in base2 budgetrcl robust2 e4b mc; do step "mu0_e12_$s" e12_revision.py "$s"; done
step mu0_e13_rollout e13_theory_checks.py rollout
for s in sweep main cert budgetsweep; do step "mu0_e15_$s" e15_leverage.py "$s"; done
step mu0_export_rollout export_rollout.py
echo "$(date +%T) MU0 PASS DONE" >> "$LOG"
