"""Feasibility check of the exact clairvoyant MILP (convex-hull form) against the local search and the certificate."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
import numpy as np
from aqmpc.params import Plant, MPCSettings
from aqmpc.scenarios import make_shift, ScenarioConfig
from aqmpc.simulate import run_shift
from aqmpc.clairvoyant import Clairvoyant
from aqmpc.extra_controllers import RolloutCostate
from common import load_tuning
from e15_leverage import clairvoyant_best

T = load_tuning()
for s, seed in [(1.0, 1000), (1.0, 1001), (4.0, 1000)]:
    pl = replace(Plant(), q_scale=s); rho = s * T["rho_iso"]; st = MPCSettings(rho=rho)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    m, tr = run_shift(sh, RolloutCostate(pl, st, info="oracle"), pl, sensor_seed=seed + 7919, return_traces=True)
    cv = Clairvoyant(pl, sh, st.duties, rho)
    (idx, Jc, src), rec = clairvoyant_best(cv, sh, {"CL-r-exact": tr["U"]}, np.random.default_rng(seed), n_random=2)
    LB = cv.lower_bound(idx)[0]
    t0 = time.time(); Jlp, _, st_lp, _ = cv.exact_milp(relax=True); tlp = time.time() - t0
    t0 = time.time(); Jm, im, st_m, gap = cv.exact_milp(time_limit=900); tm = time.time() - t0
    Jm_check = cv.forward(im)[1] if im is not None else np.nan
    print(f"s={s} seed={seed} date={sh.date}: Jc={Jc:.7f} box-LB={LB:.7f} LP={Jlp:.7f} ({tlp:.1f}s) "
          f"MILP={Jm:.7f} status={st_m} gap={gap} ({tm:.1f}s) forward(MILP)={Jm_check:.7f} "
          f"Jc-MILP={100*(Jc-Jm)/Jc:.2e}%  Jc-LP={100*(Jc-Jlp)/Jc:.2e}%", flush=True)
