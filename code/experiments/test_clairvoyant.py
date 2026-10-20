"""Consistency and speed test of the clairvoyant benchmark on one day (not part of the paper's pipeline)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from aqmpc.params import Plant, MPCSettings
from aqmpc.scenarios import make_shift, ScenarioConfig
from aqmpc.simulate import run_shift
from aqmpc.clairvoyant import Clairvoyant, Playback
from aqmpc.extra_controllers import RolloutCostate
from common import load_tuning

T = load_tuning(); rho = T["rho_iso"]
pl = Plant(); st = MPCSettings(rho=rho)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
t0 = time.time()
cv = Clairvoyant(pl, sh, st.duties, rho)
print("setup s", round(time.time() - t0, 2), flush=True)
J_of = lambda m: m["CE"] + rho * m["energy_Wh"]
m_r, tr = run_shift(sh, RolloutCostate(pl, st, info="oracle"), pl, sensor_seed=seed + 7919, return_traces=True)
idx_r = cv.to_idx(tr["U"])
t0 = time.time(); _, J_int = cv.forward(idx_r)
print("CL-r oracle: sim J", round(J_of(m_r), 6), "internal J", round(J_int, 6), "forward s", round(time.time() - t0, 3), flush=True)
t0 = time.time(); mu = cv.costate(idx_r); print("costate s", round(time.time() - t0, 3), flush=True)
Cs, _ = cv.forward(idx_r)
t0 = time.time(); D = cv.deltas(idx_r, Cs, mu); print("deltas s", round(time.time() - t0, 3), "min", D.min(), flush=True)
# exactness of a single-block change
m0 = int(np.argmin(D.min(axis=1))); j0 = int(D[m0].argmin())
alt = idx_r.copy(); alt[m0] = j0
print("single change predicted", D[m0, j0], "actual", cv.forward(alt)[1] - J_int, flush=True)
t0 = time.time(); idx_c, J_c, it = cv.improve(idx_r)
print("improve from CL-r:", round(J_c, 6), "iters", it, "s", round(time.time() - t0, 1), flush=True)
m_c = run_shift(sh, Playback(cv.schedule(idx_c)), pl, sensor_seed=seed + 7919)
print("playback sim J", round(J_of(m_c), 6), flush=True)
base = np.where(sh.occ[::6][:cv.nb] > 0, 1, 0)
t0 = time.time(); _, J_b, it_b = cv.improve(base); print("from base 0.8:", round(J_b, 6), it_b, round(time.time() - t0, 1), flush=True)
