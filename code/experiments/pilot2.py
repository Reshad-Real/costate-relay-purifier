"""Pilot 2: regimes where forecasts might pay (long sensing delay; tight shift energy budget)."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from aqmpc.params import Plant, MPCSettings
from aqmpc.sensors import SensorConfig
from aqmpc.harness import train_forecaster, evaluate
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)

def summarise(rows):
    df = pd.DataFrame(rows)
    base = df[df.policy == "Off"].set_index("seed"); ao = df[df.policy == "AO-100"].set_index("seed")
    df["captured"] = [(base.CE[s] - c) / (base.CE[s] - ao.CE[s]) for s, c in zip(df.seed, df.CE)]
    df["Efrac"] = [e / ao.energy_Wh[s] for s, e in zip(df.seed, df.energy_Wh)]
    return df, df.groupby("policy")[["CE", "captured", "Efrac", "energy_Wh", "switches", "exceed_L3_area"]].agg(["mean"]).round(3)

if __name__ == "__main__":
    mode = sys.argv[1]; n = int(sys.argv[2]); scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    pl = Plant(q_scale=scale)
    seeds = list(range(1000, 1000 + n))
    if mode == "delay":
        sc = SensorConfig(delay_steps=30, lag_scale=4.0)
        fc = train_forecaster(pl, sens_cfg=sc)
        specs = [("Off", {}), ("AO", {"u": 1.0}), ("Relay", {})]
        for rho in [0.02, 0.05]:
            st = MPCSettings(rho=rho)
            specs += [("CF", {"settings": st, "name": f"CF-{rho}"})]
            specs += [("MPC", {"forecast": f, "settings": st, "name": f"{f[:4]}-{rho}"}) for f in ["persistence", "learned", "oracle"]]
        rows = evaluate(specs, seeds, truth=pl, sens_cfg=sc, forecaster=fc)
    else:
        budget = float(sys.argv[4])
        fc = train_forecaster(pl)
        specs = [("Off", {}), ("AO", {"u": 1.0}), ("Relay", {})]
        for alloc in ["fixed", "pacing", "rollout"]:
            st = MPCSettings(rho=0.02, e_shift=budget, alloc=alloc)
            specs += [("CF", {"settings": st, "name": f"CF-{alloc}", "use_forecaster": alloc == "rollout"})]
            specs += [("MPC", {"forecast": "learned", "settings": st, "name": f"MPC-FF-{alloc}"})]
            specs += [("MPC", {"forecast": "persistence", "settings": st, "name": f"MPC-FB-{alloc}"})]
            if alloc == "rollout":
                specs += [("MPC", {"forecast": "oracle", "settings": st, "name": f"MPC-OR-{alloc}"})]
        rows = evaluate(specs, seeds, truth=pl, forecaster=fc)
    df, g = summarise(rows)
    print(g.sort_values(("Efrac", "mean")))
    os.makedirs("../results/pilot", exist_ok=True)
    df.to_csv(f"../results/pilot/pilot2_{mode}_{scale}_{sys.argv[4] if len(sys.argv)>4 else ''}.csv", index=False)
