"""Pilot run: train the forecaster, then compare policies (and an MPC rho sweep) on a few shifts."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from dataclasses import replace
from aqmpc.params import Plant, MPCSettings, POLLUTANTS
from aqmpc.harness import train_forecaster, evaluate

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    pl = Plant(q_scale=scale)
    fc = train_forecaster(pl)
    print("weld PM2.5 amplitude est/true:", np.round(fc.a[0, 0, 0], 1), round(45e3 / pl.vol[0], 1))
    seeds = list(range(1000, 1000 + n))
    specs = [("Off", {}), ("Relay", {}), ("Relay-PM", {}), ("AO", {"u": 0.8}), ("AO", {"u": 1.0}), ("PI", {})]
    Mpen = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    for rho in [0.005, 0.02, 0.05, 0.1, 0.2]:
        st = MPCSettings(rho=rho, e_shift=1e9, M=Mpen)
        specs.append(("CF", {"settings": st, "name": f"CF-{rho}"}))
        for fm in ["persistence", "learned", "oracle"]:
            specs.append(("MPC", {"forecast": fm, "settings": st, "name": f"{fm[:4]}-{rho}"}))
    t0 = time.time()
    rows = evaluate(specs, seeds, truth=pl, forecaster=fc)
    print("evaluated %d runs in %.1fs" % (len(rows), time.time() - t0))
    df = pd.DataFrame(rows)
    base = df[df.policy == "Off"].set_index("seed")
    ao = df[df.policy == "AO-100"].set_index("seed")
    df["captured"] = [(base.CE[s] - c) / (base.CE[s] - ao.CE[s]) for s, c in zip(df.seed, df.CE)]
    df["Efrac"] = [e / ao.energy_Wh[s] for s, e in zip(df.seed, df.energy_Wh)]
    g = df.groupby("policy")[["CE", "captured", "Efrac", "energy_Wh", "TWA_PM25", "runtime_h", "switches",
                              "exceed_L2_min_PM", "exceed_L3_min"]].mean().round(3)
    print(g.sort_values("Efrac"))
    os.makedirs("../results/pilot", exist_ok=True)
    df.to_csv(f"../results/pilot/pilot_scale{scale}_M{Mpen}.csv", index=False)
