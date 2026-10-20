"""E10: identification of the purifier's removal contribution from decay tests read by the low-cost
sensors (supports the protocol of Appendix A; reviewer 2, comment 3).

Procedure simulated: a 10-min pulse release in zone A (e.g. incense), then 60 min of decay with the
purifier off; the same with the purifier at full duty. A first-order decay with background,
y(t) = b + a exp(-k t), is fitted to the zone-A PM2.5 reading from 5 min after the release stops. The
purifier contribution is dk = k_on - k_off, from which the full-duty flow is rescaled as
Q_hat = Q_nom * dk / dk_model(Q_nom). Repeated over sensor noise, gain errors and truth variations.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from dataclasses import replace
from scipy.optimize import curve_fit
from aqmpc.params import Plant, NP, NZ, iPM25
from aqmpc.plant import TruthSim
from aqmpc.sensors import SensorChain, SensorConfig
from common import out

DT = 10 / 3600


def decay_test(truth, u, rng, c_out_pm=20.0, src=600e3, sens_cfg=SensorConfig()):
    sim = TruthSim(truth, DT)
    c_out = np.array([c_out_pm, 1.8 * c_out_pm, 80, 420, 0.5, 20])
    d0 = truth.lam0 * c_out[:, None] * np.ones((1, NZ))
    c = sim.steady_state(u, d0)
    sens = SensorChain(rng, DT, sens_cfg); sens.reset(c)
    S = np.zeros((NP, NZ)); S[iPM25, 0] = src
    ys, ts = [], []
    n_rel, n_dec = int(10 / 60 / DT), int(65 / 60 / DT)
    for i in range(n_rel + n_dec):
        d = d0 + (S / sim.vol[None, :] if i < n_rel else 0.0)
        y = sens.step(c)
        c = sim.step(c, u, d)
        if i >= n_rel + int(5 / 60 / DT):
            ys.append(y[iPM25, 0]); ts.append((i - n_rel) * DT)
    t = np.array(ts) - ts[0]; y = np.array(ys)
    f = lambda t, b, a, k: b + a * np.exp(-k * t)
    p, _ = curve_fit(f, t, y, p0=[y[-1], y[0] - y[-1], 1.5], maxfev=20000)
    return p[2]


def model_dk(pl):
    """Same fit applied to noise-free model output (maps a true Q to the apparent dk)."""
    rng = np.random.default_rng(0)
    clean = SensorConfig(gain_sd=0.0, noise_scale=1e-6, lag_scale=1e-6, delay_steps=0, eco2_xsens=0.0)
    return decay_test(pl, 1.0, rng, sens_cfg=clean) - decay_test(pl, 0.0, rng, sens_cfg=clean)


if __name__ == "__main__":
    nom = Plant()
    dk_nom = model_dk(nom)
    rows = []
    rng = np.random.default_rng(10)
    for case, truth in [("nominal", nom), ("Q x0.6", replace(nom, q_scale=0.6)), ("Q x1.25", replace(nom, q_scale=1.25)),
                        ("lam0 x2", replace(nom, lam0=2.0)), ("interzonal x0.25", replace(nom, f_ab=75., f_bc=75., f_ac=37.5))]:
        q_true = truth.q_op
        for rep in range(50):
            for n_pairs in [1, 3]:
                ks = [(decay_test(truth, 1.0, rng), decay_test(truth, 0.0, rng)) for _ in range(n_pairs)]
                dk = np.mean([a - b for a, b in ks])
                q_hat = nom.q_op * dk / dk_nom
                rows.append({"case": case, "rep": rep, "pairs": n_pairs, "dk": dk, "q_true": q_true, "q_hat": q_hat,
                             "rel_err": q_hat / q_true - 1})
    df = pd.DataFrame(rows)
    df.to_csv(out("e10/identification.csv"), index=False)
    s = df.groupby(["case", "pairs"]).rel_err.agg(["mean", "std", lambda e: np.mean(np.abs(e))]).round(3)
    print("apparent dk at nominal Q (noise-free):", round(dk_nom, 3))
    print(s)
