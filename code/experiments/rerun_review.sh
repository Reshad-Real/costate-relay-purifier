#!/usr/bin/env bash
# Experiments for the revision after the internal review of V1 (2026-10-16):
#   more leverages below the testbed's (sweep, lambda1, cert, e16), the price-told comparator in the budget sweep,
#   the infiltration-rate (a0) sweep with THC re-tuned per room.
# Resumable (markers rv_<step>.done in logs/rerun_markers). Launch from the session, not via WMI.
cd "$(dirname "$0")"
PY="/c/Users/Asus ROG Strix G16/AppData/Local/Python/bin/python.exe"
export AQ_WORKERS=${AQ_WORKERS:-12}
LOG=../../logs/results_review_v1.log
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
echo "$(date +%T) review chain started" >> "$LOG"
for s in sweep lambda1 cert budgetsweep a0sweep; do step "rv_e15_$s" e15_leverage.py "$s"; done
step rv_e16 e16_sampled.py
echo "$(date +%T) REVIEW CHAIN DONE" >> "$LOG"
