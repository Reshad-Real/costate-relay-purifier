"""How does the exact clairvoyant MILP scale with the number of one-minute blocks? (diagnostic)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
import numpy as np
from aqmpc.params import Plant, MPCSettings
from aqmpc.scenarios import make_shift, ScenarioConfig
from aqmpc.clairvoyant import Clairvoyant
from common import load_tuning

T = load_tuning()
pl = Plant(); rho = T["rho_iso"]; st = MPCSettings(rho=rho)
sh = make_shift(np.random.default_rng(1000), ScenarioConfig())
from aqmpc.scenarios import Shift
for nb in [30, 60, 120]:
    n = nb * 6
    sh2 = Shift(sh.t[:n], sh.S[:n], sh.c_out[:n], sh.occ[:n], sh.plan, sh.actual, sh.date)
    cv = Clairvoyant(pl, sh2, st.duties, rho)
    i0 = cv.improve(np.ones(nb, int))[0]
    J1 = cv.forward(i0)[1]
    t0 = time.time(); Jlp = cv.exact_milp(relax=True, time_limit=300)[0]; tl = time.time() - t0
    t0 = time.time(); Jm, im, stt, gap = cv.exact_milp(time_limit=300); tm = time.time() - t0
    print(f"nb={nb}: 1-opt {J1:.8f}  LP {Jlp:.8f} ({tl:.1f}s)  MILP {Jm} status {stt} ({tm:.1f}s)  "
          f"1opt-MILP {(J1 - Jm) / J1 if Jm else float('nan'):.2e}  1opt-LP {(J1 - Jlp) / J1:.2e}", flush=True)
