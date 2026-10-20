#!/usr/bin/env bash
# Run the ISA-revision experiments in sequence (one worker pool at a time).
# Waits for the E11 test stage that may already be running, then continues.
cd "$(dirname "$0")"
PY="/c/Users/Asus ROG Strix G16/AppData/Local/Python/bin/python.exe"
export AQ_WORKERS=${AQ_WORKERS:-10}
LOG=../../logs/results_revision.log
while ! grep -q "test done" ../../results_e11.log 2>/dev/null; do sleep 20; done
echo "$(date +%T) e11 test finished" >> "$LOG"
"$PY" e11_baselines.py refine front >> "$LOG" 2>&1
"$PY" e12_revision.py base2 budget e7roll robust2 e4b mc >> "$LOG" 2>&1
"$PY" e13_theory_checks.py invariance >> "$LOG" 2>&1
"$PY" e6_milp.py 100 >> "$LOG" 2>&1
echo "$(date +%T) ALL REVISION STAGES DONE" >> "$LOG"
