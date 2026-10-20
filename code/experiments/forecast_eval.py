"""E9: forecast accuracy and the circularity of model-inverted sources (reviewer 2, comment 5).

In the plant the emission drive is unobservable, so the forecaster is trained and scored on the
drive obtained by inverting the controller's own model. In simulation the true drive is known, so
we can (i) score forecasts against the true drive as well as the inverted one, (ii) train on the
true drive instead, and (iii) see how model mismatch is absorbed into the inverted signal.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *                                     # noqa
from dataclasses import replace
from aqmpc.params import Plant, MPCSettings, NP, NZ, iPM25, iTVOC
from aqmpc.scenarios import make_shift, ScenarioConfig, DT_SIM, SHIFT
from aqmpc.controllers import Relay
from aqmpc.simulate import run_shift
from aqmpc.plant import Discretised
from aqmpc.forecast import OnlineDrive, invert_drive
from aqmpc.harness import train_forecaster

LEADS = [1, 5, 15, 30]
CHANNELS = [("PM2.5 zone A", iPM25, 0), ("TVOC zone B", iTVOC, 1)]


def score_shift(seed, truth, model, fc_list):
    shift = make_shift(np.random.default_rng(seed), ScenarioConfig())
    _, tr = run_shift(shift, Relay(), truth, sensor_seed=seed + 7919, return_traces=True)
    st = int(round((1 / 60) / DT_SIM))
    K = len(shift.t) // st
    Y = tr["Y"][: K * st].reshape(K, st, NP, NZ).mean(axis=1)
    Ub = tr["U"][: K * st].reshape(K, st).mean(axis=1)
    Dtrue = tr["drive"][: K * st].reshape(K, st, NP, NZ).mean(axis=1)
    disc = Discretised(model, 1 / 60, MPCSettings().duties)
    Dinv = invert_drive(disc, Y, Ub)
    t = shift.t[::st][:K]
    recs = []
    for fname, fc in fc_list:
        od = OnlineDrive(disc, fc, shift.plan, mode="learned")
        od.init_steady(Y[0], 0.0)
        for k in range(1, K - max(LEADS) - 1):
            od.update(t[k], Y[k], Ub[k - 1])
            if not (SHIFT[0] <= t[k] < SHIFT[1] - max(LEADS) / 60):
                continue
            F = od.forecast(t[k], max(LEADS) + 1, 1 / 60)
            for L in LEADS:
                for cname, p, z in CHANNELS:
                    recs.append({"seed": seed, "forecaster": fname, "lead_min": L, "channel": cname,
                                 "f": F[L, p, z], "true": Dtrue[k + L, p, z], "inv": Dinv[k + L, p, z]})
    return recs


def run():
    nom = Plant()
    configs = [("nominal model", nom), ("mismatch: Q x0.6, lam0 x1.5", replace(nom, q_scale=0.6, lam0=1.5))]
    allrec = []
    closed = []
    for cname, truth in configs:
        fc_inv = train_forecaster(truth=truth, model=nom)
        fc_true = train_forecaster(truth=truth, model=nom, use_true=True)
        for sd in SMALL_SEEDS:
            for r in score_shift(sd, truth, nom, [("trained on inverted drive", fc_inv),
                                                  ("trained on true drive", fc_true)]):
                r["config"] = cname
                allrec.append(r)
        # closed loop where forecasts matter: budget allocation by rollout
        T = load_tuning()
        rB = T["rho_B"]["0.25"]; eb = 0.25 * E_FULL
        st = MPCSettings(rho=rB, e_shift=eb, alloc="rollout")
        for fname, fc in [("trained on inverted drive", fc_inv), ("trained on true drive", fc_true)]:
            rows = evaluate([("Off", {}), ("AO", {"u": 1.0}),
                             ("MPC", {"forecast": "learned", "settings": st, "name": "MPC-FF/rollout"})],
                            SMALL_SEEDS, truth=truth, model=nom, forecaster=fc)
            d = pd.DataFrame(rows); d["config"] = cname; d["forecaster"] = fname
            closed.append(d)
    df = pd.DataFrame(allrec)
    df.to_csv(out("e9/forecast_points.csv"), index=False)
    g = df.assign(err_true=df.f - df.true, err_inv=df.f - df.inv, inv_minus_true=df.inv - df.true)
    summ = g.groupby(["config", "forecaster", "channel", "lead_min"]).agg(
        rmse_vs_true=("err_true", lambda e: float(np.sqrt(np.mean(e ** 2)))),
        rmse_vs_inv=("err_inv", lambda e: float(np.sqrt(np.mean(e ** 2)))),
        bias_vs_true=("err_true", "mean"),
        inv_bias=("inv_minus_true", "mean"),
        true_mean=("true", "mean"), true_sd=("true", "std")).reset_index()
    summ.to_csv(out("e9/forecast_summary.csv"), index=False)
    cl = pd.concat(closed)
    cl["config_seed"] = cl.config + "|" + cl.forecaster
    out_rows = []
    for (cfg, fn), d in cl.groupby(["config", "forecaster"]):
        d = d.copy(); d["config"] = cfg + "|" + fn
        out_rows.append(normalise(d))
    pd.concat(out_rows).to_csv(out("e9/closed_loop.csv"), index=False)


if __name__ == "__main__":
    run()
