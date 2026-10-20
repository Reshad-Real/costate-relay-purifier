# Costate relay control of a positive bilinear air-purification process

Code, data and simulation results for the manuscript *"Costate relay control of a positive bilinear
air-purification process under weak actuation: Near-optimality and the value of preview"* (submitted to ISA
Transactions). Every number, table and figure of the manuscript and its supplementary material is generated from
the files in this repository.

All controller results are from closed-loop simulation of a three-zone purifier model parameterised from fan
datasheets, purifier design data and the literature. Outdoor PM2.5 is measured (U.S. Embassy Dhaka monitor, see
below); indoor emissions and sensor errors are generated.

## Contents

| Path | Content |
|---|---|
| `code/aqmpc/` | Python package: multizone bilinear plant, scenarios, sensor chain, controllers (baseline multilevel relay, threshold-hysteresis, PI, timer, zone tracking, stationary and rollout costate laws, economic MPC), exact branch-and-bound solver, forecaster, budget allocation, clairvoyant reference, closed-loop simulator |
| `code/experiments/` | Experiment scripts (see below) and the chain scripts that ran them |
| `code/analysis/` | `summarize_isa.py` (with `summarize.py`): statistics, LaTeX tables and number macros; `fallbacks.py` |
| `code/figures/` | `make_figs_isa.py`, `make_figs_2col.py`: all figures, drawn at printed size |
| `code/csolver/` | Portable C99 solver (`mpc_solver.c/.h`), rollout and stationary costate laws (`rollout_law.h`, `costate_law.h`), tests and operation counter |
| `code/data_prep/` | `prep_outdoor.py`: builds the outdoor PM2.5 series from the raw downloads and checks it against the EPA data |
| `code/tools/` | Manuscript build helpers (bibliography, Word version, style checks) |
| `data/` | `outdoor_dhaka_pm25.csv` (hourly), `outdoor_dhaka_days.csv` (working days used); raw downloads in `data/external/` |
| `results/` | Simulation outputs (CSV/JSON) of every experiment; `results/e9/forecast_points.csv.gz` is the compressed raw forecast record |
| `manuscript/generated/` | Generated number macros (`macros.tex`) and tables (`tab_*.tex`) used by the manuscript |
| `figures2/` | Generated figures (PDF and PNG) |

## Experiments

| Script (stages) | Manuscript content |
|---|---|
| `run_all.py` (`tune e1 e1b e3 e2 e4 e5 e7 e8 e9`) | Tuning of the economic weight; main comparison; budget; exposure–energy fronts; robustness; exposure weights; budget formulation; horizon and blocking; forecasting |
| `e11_baselines.py` | Tuning and testing of the rule-based and tracking baselines |
| `e12_revision.py` | Further baselines, budget designs, faults, Monte Carlo study |
| `e13_theory_checks.py` | Rollout improvement (Proposition 3) and forecast invariance (Proposition 2) along closed-loop trajectories |
| `e14_forecast_skill.py` | Forecast skill against persistence |
| `e15_leverage.py` | Clairvoyant reference and certificates (box and LP), leverage sweep, infiltration sweep, budget sweep |
| `e16_sampled.py` | Test of Theorem 1 on the sampled model itself |
| `e6_exactness.py`, `e6_milp.py`, `e6_methods.py`, `e6_timing.py` | Exactness of branch and bound, MILP comparison, relax-and-round, solve times |
| `export_costate.py`, `export_rollout.py` | C versions of the costate laws, checked against Python |

The chain scripts (`rerun_real_outdoor.sh`, `rerun_mu0.sh`, `rerun_review.sh`) list the order in which the stages
were run; they are resumable and set the Python interpreter in their `PY` variable, which has to be adapted.

## Reproduce

Requirements: Python 3.11 or later with numpy, scipy (HiGHS via `scipy.optimize`), pandas and matplotlib; gcc for
the C code. The full set of experiments takes several hours on a 16–20-core machine; `AQ_WORKERS` sets the number of
worker processes, and `QUICK=1` runs a smoke test with a few days per experiment (outputs in `results_quick/`).

```bash
cd code/data_prep && python prep_outdoor.py            # outdoor PM2.5 series (already included in data/)
cd ../experiments && bash rerun_real_outdoor.sh        # main experiments (see the chain scripts for the order)
cd ../analysis && python summarize_isa.py              # numbers and tables -> manuscript/generated/
cd ../figures && python make_figs_isa.py && python make_figs_2col.py weights forecast   # figures -> figures2/
```

To check the published numbers without rerunning the simulations, run only the last two steps on the included
`results/`.

## Data sources

- Outdoor PM2.5: hourly readings of the U.S. Embassy Dhaka monitor (AirNow), as relayed on the public ThingSpeak
  channel 1541265, October 2021 to February 2024 (`data/external/airnow_dhaka/`).
- Check of the relay against the official daily means of the same monitor: U.S. EPA, Sarwar et al. (2023), data DOI
  [10.23719/1528454](https://doi.org/10.23719/1528454) (`data/external/epa_dhaka/`). The result of the check is in
  `results/platform/outdoor_check.json`.

The third-party data remain subject to their providers' terms.
