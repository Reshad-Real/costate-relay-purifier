"""Quick check of the optimality certificate (lower bound on J*) on a few days and purifier scales."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
import numpy as np
from aqmpc.params import Plant, MPCSettings
from aqmpc.scenarios import make_shift, ScenarioConfig
from aqmpc.simulate import run_shift
from aqmpc.clairvoyant import Clairvoyant
from aqmpc.extra_controllers import RolloutCostate
from common import load_tuning, TEST_SEEDS
from e15_leverage import clairvoyant_best

T = load_tuning()
rho_iso = T["rho_iso"] if "rho_iso" in T else MPCSettings().rho
print("rho_iso", rho_iso, flush=True)
for s in [0.25, 1.0, 4.0]:
    for seed in TEST_SEEDS[:3]:
        t0 = time.time()
        pl = replace(Plant(), q_scale=s)
        rho = s * rho_iso
        st = MPCSettings(rho=rho)
        sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
        m, tr = run_shift(sh, RolloutCostate(pl, st, info="oracle"), pl, sensor_seed=seed + 7919, return_traces=True)
        cv = Clairvoyant(pl, sh, st.duties, rho)
        (idx, Jc, src), rec = clairvoyant_best(cv, sh, {"CL-r-exact": tr["U"]}, np.random.default_rng(seed), n_random=2)
        LB, J, contrib = cv.lower_bound(idx)
        off = cv.forward(np.zeros(cv.nb, dtype=int))[1]
        Jr = cv.forward(cv.to_idx(tr["U"]))[1]
        print(f"s={s} seed={seed}: Jc={Jc:.5f} LB={LB:.5f} certgap(Jc)={100*(Jc-LB)/Jc:.4f}% "
              f"CLrex gap={100*(Jr-Jc)/Jc:.4f}% range={100*(off-Jc)/Jc:.2f}% "
              f"nonzero={np.sum(contrib<0)}/{cv.nb} {time.time()-t0:.1f}s", flush=True)
