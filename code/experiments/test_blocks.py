"""Ablation (40 days, scales 0.25 and 1, exact state and drive): does a one-step first move block remove the first-order loss of EMPC with the rollout terminal weight?"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from aqmpc.params import Plant, MPCSettings
from aqmpc.scenarios import make_shift, ScenarioConfig
from aqmpc.simulate import run_shift
from aqmpc.controllers import MPC
from common import load_tuning, SMALL_SEEDS, out

VARIANTS = {
    "L-uniform": dict(terminal_rollout=True),
    "L-first1": dict(terminal_rollout=True, blocks=(1, 2, 3, 6, 8, 10)),
    "L-first1-M0": dict(terminal_rollout=True, blocks=(1, 2, 3, 6, 8, 10), M=0.0),
    "L-uniform-M0": dict(terminal_rollout=True, M=0.0),
}


def job(a):
    s, seed, rho_iso, var = a
    pl = replace(Plant(), q_scale=s)
    rho = s * rho_iso
    st = MPCSettings(rho=rho, **VARIANTS[var])
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    m = run_shift(sh, MPC(pl, None, "oracle", st, name=var), pl, sensor_seed=seed + 7919)
    return dict(scale=s, seed=seed, policy=var, J=m["CE"] + rho * m["energy_Wh"])


if __name__ == "__main__":
    T = load_tuning()
    seeds = SMALL_SEEDS
    jobs = [(s, sd, T["rho_iso"], v) for s in [0.25, 1.0] for sd in seeds for v in VARIANTS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=int(os.environ.get("AQ_WORKERS", 12))) as ex:
        rows = list(ex.map(job, jobs, chunksize=1))
    d = pd.DataFrame(rows)
    d.to_csv(out("e15/blocks_test.csv"), index=False)
    L = pd.read_csv(out("e15/leverage.csv"))
    ref = L[L.policy.isin(["Clairvoyant", "Off", "EMPC-oracle", "CL-r-exact"])].pivot_table(
        index=["scale", "seed"], columns="policy", values="J")
    J = d.pivot_table(index=["scale", "seed"], columns="policy", values="J").join(ref, how="inner")
    for sc in [0.25, 1.0]:
        j = J.loc[sc]
        rel = {c: 100 * ((j[c] - j["Clairvoyant"]) / (j["Off"] - j["Clairvoyant"])).mean()
               for c in list(VARIANTS) + ["EMPC-oracle", "CL-r-exact"]}
        print(sc, {k: round(v, 4) for k, v in rel.items()}, flush=True)
    print(f"{time.time() - t0:.0f} s")
