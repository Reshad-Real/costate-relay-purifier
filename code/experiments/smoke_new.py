"""Smoke test and timing of the new baselines on one shift."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aqmpc.params import Plant, MPCSettings
from aqmpc.harness import evaluate, train_forecaster
from common import load_tuning

if __name__ == "__main__":
    T = load_tuning()
    pl = Plant()
    fc = train_forecaster(pl)
    st = MPCSettings(rho=T["rho_iso"])
    specs = [("Relay", {}), ("THC", {"th1": 35, "dth": 20, "hyst": 5, "gate": True}),
             ("MPC", {"forecast": "learned", "settings": st}),
             ("MPC-track", {"settings": st, "q": 0.01, "r25": 25.0}),
             ("MPC-RR", {"settings": st})]
    for s in specs:
        t0 = time.time()
        r = evaluate([s], [1000], truth=pl, forecaster=fc, workers=1)[0]
        print(f"{r['policy']:12s} CE {r['CE']:.3f} E {r['energy_Wh']:.1f} Wh sw {r['switches']:4d} "
              f"L3min {r['exceed_L3_min']:.1f}  [{time.time() - t0:.1f} s]")
