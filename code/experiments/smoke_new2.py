"""Smoke test of the revision controllers on one shift."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aqmpc.params import Plant, MPCSettings
from aqmpc.harness import evaluate
from common import load_tuning

if __name__ == "__main__":
    T = load_tuning()
    pl = Plant()
    st = MPCSettings(rho=T["rho_iso"])
    specs = [("AO", {"u": 0.8}), ("Sched", {"u": 0.8}), ("Sched", {"u": 0.8, "pre": 0.5, "name": "Sched-80-pre"}),
             ("PI-SD", {"r25": 25, "kp": 0.5, "ki": 1.0}), ("THC", {"th1": 25, "dth": 40, "hyst": 5, "avg": 6}),
             ("CF", {"settings": st, "name": "Costate-FB"}), ("RCL", {"settings": st}),
             ("RCL", {"settings": st, "info": "oracle", "name": "Rollout-CL-oracle"})]
    for s in specs:
        t0 = time.time()
        r = evaluate([s], [1000], truth=pl, workers=1)[0]
        print(f"{r['policy']:18s} CE {r['CE']:.3f} E {r['energy_Wh']:5.1f} Wh J {r['CE'] + T['rho_iso'] * r['energy_Wh']:.4f} "
              f"sw {r['switches']:4d}  [{time.time() - t0:.1f} s]")
