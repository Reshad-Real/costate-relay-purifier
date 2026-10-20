#!/usr/bin/env bash
# Continue the ISA-revision chain after the budget stages (robustness with the budgeted rollout law added).
cd "$(dirname "$0")"
PY="/c/Users/Asus ROG Strix G16/AppData/Local/Python/bin/python.exe"
export AQ_WORKERS=${AQ_WORKERS:-10}
LOG=../../logs/results_revision.log
echo "$(date +%T) chain 2 started" >> "$LOG"
"$PY" e12_revision.py e7roll >> "$LOG" 2>&1
"$PY" e13_theory_checks.py invariance >> "$LOG" 2>&1
"$PY" e12_revision.py robust2 e4b mc >> "$LOG" 2>&1
"$PY" e6_milp.py 100 >> "$LOG" 2>&1
echo "$(date +%T) ALL REVISION STAGES DONE (chain 2)" >> "$LOG"
