"""Mechanism diagnostic (30 test shifts): where the relay spends energy and which duty levels each
policy uses while running. Output: results/e1/mechanism.csv."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from aqmpc.params import Plant, MPCSettings
from aqmpc.scenarios import make_shift, ScenarioConfig, SHIFT, DT_SIM
from aqmpc.controllers import Relay, RelayOccupied, CostateFeedback, MPC
from aqmpc.simulate import run_shift
from aqmpc.harness import train_forecaster
from common import out, RES

if __name__ == "__main__":
    T = json.load(open(os.path.join(RES, "tuning.json")))
    pl = Plant(); st = MPCSettings(rho=T["rho_iso"])
    fc = train_forecaster(pl, workers=4)
    rows = []
    for seed in range(1000, 1030):
        sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
        occ = (sh.t >= SHIFT[0]) & (sh.t < SHIFT[1])
        for name, ctrl in [("Relay", Relay()), ("Relay-occ", RelayOccupied()), ("Costate-FB", CostateFeedback(pl, st)),
                           ("MPC-FF", MPC(pl, fc, "learned", st))]:
            m, tr = run_shift(sh, ctrl, pl, sensor_seed=seed + 7919, return_traces=True)
            U = tr["U"]; P = pl.fan.power(U) * DT_SIM
            on = U > 0
            rows.append({"seed": seed, "policy": name, "CE": m["CE"], "E": m["energy_Wh"],
                         "E_unocc_share": P[~occ].sum() / max(P.sum(), 1e-9),
                         "frac80": np.mean(U[on] == 0.8) if on.any() else np.nan,
                         "frac90": np.mean(U[on] == 0.9) if on.any() else np.nan,
                         "frac100": np.mean(U[on] == 1.0) if on.any() else np.nan,
                         "runtime_h": m["runtime_h"]})
    d = pd.DataFrame(rows)
    d.to_csv(out("e1/mechanism.csv"), index=False)
    print(d.groupby("policy")[["CE", "E", "E_unocc_share", "frac80", "frac90", "frac100", "runtime_h"]].mean().round(3))
