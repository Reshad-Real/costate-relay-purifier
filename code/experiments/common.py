"""Shared configuration for the experiment scripts (seeds, grids, tuning helpers)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from dataclasses import replace
from aqmpc.params import Plant, MPCSettings
from aqmpc.harness import evaluate, train_forecaster

QUICK = os.environ.get("QUICK") == "1"
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "results_quick" if QUICK else "results")
TUNE_SEEDS = list(range(100, 103 if QUICK else 140))
TEST_SEEDS = list(range(1000, 1003 if QUICK else 1200))
SMALL_SEEDS = list(range(1000, 1002 if QUICK else 1040))
FRONT_SEEDS = list(range(1000, 1002 if QUICK else 1060))
RHO_GRID = [0.003, 0.007, 0.012, 0.02, 0.035, 0.06, 0.1, 0.2]
E_FULL = 6.96 * 10.0          # fan energy of always-on 100 % over the 10-h window (Wh)


def out(name):
    p = os.path.join(RES, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def normalise(df):
    """Add captured share of the achievable reduction and energy fraction (per shift, per config)."""
    keys = [k for k in ["config", "seed"] if k in df.columns]
    off = df[df.policy == "Off"].set_index(keys)
    ao = df[df.policy == "AO-100"].set_index(keys)
    idx = [tuple(r) if len(keys) > 1 else r[0] for r in df[keys].values]
    ce_off = np.array([off.CE.loc[i] for i in idx])
    ce_ao = np.array([ao.CE.loc[i] for i in idx])
    df = df.copy()
    df["captured"] = (ce_off - df.CE.values) / (ce_off - ce_ao)
    df["Efrac"] = df.energy_Wh.values / E_FULL
    return df


def load_tuning():
    p = out("tuning.json")
    return json.load(open(p)) if os.path.exists(p) else None
