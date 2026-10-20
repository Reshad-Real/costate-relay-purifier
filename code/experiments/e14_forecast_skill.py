"""E14: skill of the schedule forecaster against persistence, up to 4-h leads, nominal and with a systematic
plan bias (all planned batches 20 min late; out-of-family for the forecaster). Scored against the true
interval-averaged drive of zone-A PM2.5 and zone-B TVOC on 40 relay-operated shifts."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
import numpy as np
import pandas as pd
from aqmpc.params import Plant, MPCSettings, NP, NZ, iPM25, iTVOC
from aqmpc.scenarios import make_shift, ScenarioConfig, DT_SIM, SHIFT
from aqmpc.controllers import Relay
from aqmpc.simulate import run_shift
from aqmpc.plant import Discretised
from aqmpc.forecast import OnlineDrive
from aqmpc.harness import train_forecaster
from common import out, SMALL_SEEDS

LEADS = [1, 5, 15, 30, 60, 120, 240]
CHANNELS = [("PM2.5 zone A", iPM25, 0), ("TVOC zone B", iTVOC, 1)]


def score(seed, pl, fc, scfg):
    shift = make_shift(np.random.default_rng(seed), scfg)
    _, tr = run_shift(shift, Relay(), pl, sensor_seed=seed + 7919, return_traces=True)
    st = int(round((1 / 60) / DT_SIM))
    K = len(shift.t) // st
    Y = tr["Y"][: K * st].reshape(K, st, NP, NZ).mean(axis=1)
    Ub = tr["U"][: K * st].reshape(K, st).mean(axis=1)
    Dt = tr["drive"][: K * st].reshape(K, st, NP, NZ).mean(axis=1)
    disc = Discretised(pl, 1 / 60, MPCSettings().duties)
    t = shift.t[::st][:K]
    ods = {"schedule": OnlineDrive(disc, fc, shift.plan, mode="learned"),
           "persistence": OnlineDrive(disc, None, shift.plan, mode="persistence")}
    for od in ods.values():
        od.init_steady(Y[0], 0.0)
    recs = []
    for k in range(1, K - 1):
        for od in ods.values():
            od.update(t[k], Y[k], Ub[k - 1])
        if not (SHIFT[0] <= t[k] < SHIFT[1]):
            continue
        for name, od in ods.items():
            F = od.forecast(t[k], max(LEADS) + 1, 1 / 60)
            for L in LEADS:
                if k + L >= K:
                    continue
                for cname, p, z in CHANNELS:
                    recs.append({"seed": seed, "forecaster": name, "lead_min": L, "channel": cname,
                                 "err": F[L, p, z] - Dt[k + L, p, z], "true": Dt[k + L, p, z]})
    return recs


if __name__ == "__main__":
    pl = Plant()
    fc = train_forecaster(pl)
    rows = []
    for cname, scfg in [("nominal", ScenarioConfig()), ("plan bias +20 min", replace(ScenarioConfig(), onset_bias=20 / 60))]:
        for sd in SMALL_SEEDS:
            for r in score(sd, pl, fc, scfg):
                r["scenario"] = cname
                rows.append(r)
    df = pd.DataFrame(rows)
    g = df.groupby(["scenario", "channel", "forecaster", "lead_min"]).agg(
        rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))), sd_true=("true", "std")).reset_index()
    g["nrmse"] = g.rmse / g.sd_true
    piv = g.pivot_table(index=["scenario", "channel", "lead_min"], columns="forecaster", values="rmse").reset_index()
    piv["skill_vs_persistence"] = 1 - piv["schedule"] / piv["persistence"]
    os.makedirs(os.path.dirname(out("e14/skill.csv")), exist_ok=True)
    g.to_csv(out("e14/forecast_rmse.csv"), index=False)
    piv.to_csv(out("e14/skill.csv"), index=False)
    print(piv.round(3).to_string(index=False))
