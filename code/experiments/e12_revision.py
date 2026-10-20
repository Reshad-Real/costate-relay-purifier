"""E12: experiments added in response to the ISA Transactions mock review.

Stages (run in order; each writes results/e12/*.csv and logs to stdout):
  base2    tune Sched (constant duty during occupancy, optional pre-purge), PI-SD (gated PI with sigma-delta
           output) and THC with 60-s averaged input on TUNE_SEEDS (J = CE + rho_iso*E); test the best of
           each plus the rollout costate law (RCL, estimated and oracle information) on TEST_SEEDS
  budget   25 % budget: tune the fixed price of MPC-FB/fixed and MPC-FF/fixed (maximise captured share
           subject to the hard budget) and the pacing gain of MPC-FB/pacing on TUNE_SEEDS; test on TEST_SEEDS
  e7roll   budget-state formulation with rollout allocation (rho = 0.002 start) for the E7 budgets
  robust2  structural / operational variants (purifier location, fan failure, sensor drift, packet loss,
           stuck node, plan bias) on SMALL_SEEDS with the controllers designed for the nominal model
  e4b      THC and the tuned MPC-FB/fixed for the 30 E4 configurations (budget comparator consistency)
  mc       joint Monte Carlo: 100 random draws of all assumed parameters, 10 shifts each
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
import numpy as np
import pandas as pd
from aqmpc.params import Plant, MPCSettings, FanModel
from aqmpc.harness import evaluate, train_forecaster
from aqmpc.scenarios import ScenarioConfig
from aqmpc.sensors import SensorConfig
from common import out, load_tuning, normalise, TUNE_SEEDS, TEST_SEEDS, SMALL_SEEDS, E_FULL
from run_all import robustness_configs


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def T2():
    return json.load(open(out("e11/tuning2.json")))


def norm_e1(rows):
    """Normalise rows on TEST_SEEDS with E1's Off and AO-100 runs."""
    main = pd.read_csv(out("e1/main.csv"))
    off = main[main.policy == "Off"].set_index("seed").CE
    ao = main[main.policy == "AO-100"].set_index("seed").CE
    rows = rows.copy()
    rows["captured"] = [(off[s] - c) / (off[s] - ao[s]) for s, c in zip(rows.seed, rows.CE)]
    rows["Efrac"] = rows.energy_Wh / E_FULL
    return rows


def tune(specs, seeds, pl, fc, rho, tag):
    df = pd.DataFrame(evaluate(specs, seeds, truth=pl, forecaster=fc))
    df["key"] = [json.dumps({k: v for k, v in s[1].items() if k not in ("settings", "name")}, default=float)
                 for s in specs for _ in seeds]
    df["J"] = df.CE + rho * df.energy_Wh
    g = df.groupby("key").agg(J=("J", "mean"), E=("energy_Wh", "mean"), CE=("CE", "mean"))
    g.to_csv(out(f"e12/grid_{tag}.csv"))
    best = json.loads(g.J.idxmin())
    log(f"{tag}: best {best} J {g.J.min():.4f}")
    return best


def stage_base2():
    T = load_tuning()
    rho = T["rho_iso"]
    pl = Plant()
    fc = train_forecaster(pl)
    st = MPCSettings(rho=rho)
    res = {}
    specs = [("Sched", {"u": u, "pre": p}) for u in [0.8, 0.9, 1.0] for p in [0.0, 0.25, 0.5, 1.0]]
    res["sched"] = tune(specs, TUNE_SEEDS, pl, fc, rho, "sched")
    specs = [("PI-SD", {"r25": r, "kp": kp, "ki": ki}) for r in [10, 15, 25, 35]
             for kp in [0.25, 0.5, 1.0, 2.0] for ki in [0.0, 0.5, 2.0, 6.0]]
    res["pisd"] = tune(specs, TUNE_SEEDS, pl, fc, rho, "pisd")
    specs = [("THC", {"th1": a, "dth": b, "hyst": h, "gate": True, "avg": 6}) for a in [15, 20, 25, 30, 35]
             for b in [20, 30, 40, 60, 80] for h in [0.0, 2.5, 5.0]]
    res["thcavg"] = tune(specs, TUNE_SEEDS, pl, fc, rho, "thcavg")
    json.dump(res, open(out("e12/tuning3.json"), "w"), indent=2)
    specs = [("Sched", dict(res["sched"], name="Sched")), ("PI-SD", dict(res["pisd"], name="PI-SD")),
             ("THC", dict(res["thcavg"], name="THC-avg")),
             ("RCL", {"settings": st, "name": "Rollout-CL"}),
             ("RCL", {"settings": st, "info": "oracle", "name": "Rollout-CL-oracle"}),
             ("RCL", {"settings": st, "penalty": True, "name": "Rollout-CL-pen"}),
             ("AO", {"u": 0.8})]
    rows = norm_e1(pd.DataFrame(evaluate(specs, TEST_SEEDS, truth=pl, forecaster=fc)))
    rows.loc[rows.policy == "AO-80", "policy"] = "AO-80-check"
    rows.to_csv(out("e12/test_base2.csv"), index=False)
    log(f"base2 test: {rows.groupby('policy')[['captured', 'Efrac', 'switches']].mean().round(3).to_dict()}")


def stage_budget():
    T = load_tuning()
    pl = Plant()
    fc = train_forecaster(pl)
    rB = T["rho_B"]["0.25"]
    eb = 0.25 * E_FULL
    facs = [0.25, 0.4, 0.6, 0.8, 1.0, 1.3]
    specs = []
    for f in facs:
        st = MPCSettings(rho=rB * f, e_shift=eb, alloc="fixed")
        specs.append(("MPC", {"forecast": "persistence", "settings": st, "name": f"FBfix|{f}"}))
        specs.append(("MPC", {"forecast": "learned", "settings": st, "name": f"FFfix|{f}"}))
    for g in [0.5, 1.5, 3.0, 6.0]:
        st = MPCSettings(rho=rB, e_shift=eb, alloc="pacing", alloc_gain=g)
        specs.append(("MPC", {"forecast": "persistence", "settings": st, "name": f"FBpace|{g}"}))
    # strongest rule-based budget baseline: threshold + hysteresis + occupancy gating with an energy cut-off,
    # thresholds re-tuned for the budget
    for a in [15, 25, 35, 50, 70, 100]:
        for b in [20, 40, 80]:
            specs.append(("THC", {"th1": a, "dth": b, "hyst": 5.0, "gate": True, "e_shift": eb,
                                  "name": f"THCcut|{a}_{b}"}))
    df = pd.DataFrame(evaluate(specs, TUNE_SEEDS, truth=pl, forecaster=fc))
    df.to_csv(out("e12/budget_tune_raw.csv"), index=False)
    g = df.groupby("policy").agg(CE=("CE", "mean"), E=("energy_Wh", "mean"), Emax=("energy_Wh", "max"))
    g.to_csv(out("e12/budget_tune.csv"))
    best = {}
    for fam in ["FBfix", "FFfix", "FBpace"]:
        sub = g[g.index.str.startswith(fam + "|")]
        k = sub.CE.idxmin()                         # lowest exposure; the budget state keeps E <= E_B
        best[fam] = float(k.split("|")[1])
    sub = g[g.index.str.startswith("THCcut|")]
    th1, dth = sub.CE.idxmin().split("|")[1].split("_")
    best["THCcut"] = {"th1": float(th1), "dth": float(dth)}
    log(f"budget tuning: {best}")
    json.dump({"rB": rB, "best": best}, open(out("e12/budget_tuning.json"), "w"), indent=2)
    specs = [("MPC", {"forecast": "persistence", "settings": MPCSettings(rho=rB * best["FBfix"], e_shift=eb, alloc="fixed"),
                      "name": "MPC-FB/fixed*"}),
             ("MPC", {"forecast": "learned", "settings": MPCSettings(rho=rB * best["FFfix"], e_shift=eb, alloc="fixed"),
                      "name": "MPC-FF/fixed*"}),
             ("MPC", {"forecast": "persistence", "settings": MPCSettings(rho=rB, e_shift=eb, alloc="pacing",
                                                                          alloc_gain=best["FBpace"]),
                      "name": "MPC-FB/pacing*"}),
             ("THC", {**best["THCcut"], "hyst": 5.0, "gate": True, "e_shift": eb, "name": "THC/cutoff*"})]
    rows = norm_e1(pd.DataFrame(evaluate(specs, TEST_SEEDS, truth=pl, forecaster=fc)))
    rows["budget_frac"] = 0.25
    rows.to_csv(out("e12/budget_test.csv"), index=False)
    log(f"budget test: {rows.groupby('policy')[['captured', 'Efrac']].mean().round(3).to_dict()}")


def stage_pisd2():
    """Refine the PI grid until the optimum is interior in every dimension or at a physical bound (ki = 0), then
    re-test and replace the PI-SD rows of e12/test_base2.csv (the first version is kept as test_base2_pisd_v1.csv)."""
    T = load_tuning()
    rho = T["rho_iso"]
    pl = Plant()
    fc = train_forecaster(pl)
    axes = {"r25": [4.0, 6.0, 8.0, 10.0, 12.0], "kp": [0.06, 0.1, 0.15, 0.25, 0.35], "ki": [0.0, 0.05, 0.15]}
    best = None
    for it in range(5):
        specs = [("PI-SD", {"r25": r, "kp": kp, "ki": ki}) for r in axes["r25"] for kp in axes["kp"] for ki in axes["ki"]]
        best = tune(specs, TUNE_SEEDS, pl, fc, rho, f"pisd2_{it}")
        moved = False
        for k, v in axes.items():
            vals = sorted(v)
            i = vals.index(best[k])
            if i == len(vals) - 1:
                axes[k] = vals[1:] + [vals[-1] * 1.5]; moved = True
            elif i == 0 and not (k == "ki" and vals[0] == 0.0):
                axes[k] = [vals[0] / 1.5] + vals[:-1]; moved = True
        if not moved:
            break
    log(f"pisd2: final {best} after {it + 1} grids, axes {axes}")
    t3 = json.load(open(out("e12/tuning3.json")))
    t3["pisd_v1"] = t3.get("pisd"); t3["pisd"] = best; t3["pisd_axes"] = axes
    json.dump(t3, open(out("e12/tuning3.json"), "w"), indent=2)
    rows = norm_e1(pd.DataFrame(evaluate([("PI-SD", dict(best, name="PI-SD"))], TEST_SEEDS, truth=pl, forecaster=fc)))
    b2 = pd.read_csv(out("e12/test_base2.csv"))
    if not os.path.exists(out("e12/test_base2_pisd_v1.csv")):
        b2.to_csv(out("e12/test_base2_pisd_v1.csv"), index=False)
    b2 = pd.concat([b2[b2.policy != "PI-SD"], rows], ignore_index=True)
    b2.to_csv(out("e12/test_base2.csv"), index=False)
    log(f"pisd2 test: {rows[['captured', 'Efrac', 'switches']].mean().round(3).to_dict()}")


def stage_budgetrcl():
    """Budgeted rollout costate law: fixed energy price (tuned on the tuning seeds) + budget filter."""
    T = load_tuning()
    pl = Plant()
    fc = train_forecaster(pl)
    rB = T["rho_B"]["0.25"]
    eb = 0.25 * E_FULL
    facs = [0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.8, 2.5]
    specs = [("RCL", {"settings": MPCSettings(rho=rB * f, e_shift=eb), "name": f"RCLfix|{f}"}) for f in facs]
    df = pd.DataFrame(evaluate(specs, TUNE_SEEDS, truth=pl, forecaster=fc))
    g = df.groupby("policy").agg(CE=("CE", "mean"), E=("energy_Wh", "mean"), Emax=("energy_Wh", "max"))
    g.to_csv(out("e12/budget_rcl_tune.csv"))
    f = float(g.CE.idxmin().split("|")[1])
    log(f"budget RCL tuning: fac {f}")
    json.dump({"rB": rB, "RCLfix": f}, open(out("e12/budget_rcl_tuning.json"), "w"), indent=2)
    specs = [("RCL", {"settings": MPCSettings(rho=rB * f, e_shift=eb), "name": "RCL/fixed*"})]
    rows = norm_e1(pd.DataFrame(evaluate(specs, TEST_SEEDS, truth=pl, forecaster=fc)))
    rows["budget_frac"] = 0.25
    rows.to_csv(out("e12/budget_rcl_test.csv"), index=False)
    log(f"budget RCL test: {rows[['captured', 'Efrac']].mean().round(3).to_dict()}")


def stage_e7roll():
    pl = Plant()
    fc = train_forecaster(pl)
    e7 = pd.read_csv(out("e7/budget_formulation.csv"))
    budgets = sorted(e7.budget_Wh.unique())
    specs = [("MPC", {"forecast": "learned", "settings": MPCSettings(rho=0.002, e_shift=b, alloc="rollout"),
                      "name": f"budget-state + rollout|{b:.4f}"}) for b in budgets]
    seeds = sorted(e7.seed.unique())
    df = pd.DataFrame(evaluate(specs, seeds, truth=pl, forecaster=fc))
    df["budget_Wh"] = [float(p.split("|")[1]) for p in df.policy]
    df.to_csv(out("e12/e7_rollout.csv"), index=False)
    log("e7roll done")


def structural_configs():
    nom = Plant()
    return [("purifier in zone A", {"w_in": np.array([1.0, 0.0, 0.0])}, {}, {}),
            ("purifier in zone C", {"w_in": np.array([0.0, 0.0, 1.0])}, {}, {}),
            ("one fan failed", {"q_scale": 0.5, "fan": FanModel(p_fan=nom.fan.p_fan / 2)}, {}, {}),
            ("sensor drift +2 %/h", {}, {}, {"drift_per_h": 0.02}),
            ("packet loss 20 % + 30 s", {}, {}, {"drop_p": 0.2, "delay_steps": 5}),
            ("zone-C node frozen", {}, {}, {"stuck_zone": 2}),
            ("plan bias +20 min", {}, {"onset_bias": 20 / 60}, {}),
            ("plan bias -20 min", {}, {"onset_bias": -20 / 60}, {})]


def robust_specs(T, Tb, T2_):
    st = MPCSettings(rho=T["rho_iso"])
    rB = T["rho_B"]["0.25"]; eb = 0.25 * E_FULL
    return [("Off", {}), ("AO", {"u": 1.0}), ("Relay", {}), ("THC", dict(T2_["thc"], name="THC")),
            ("RCL", {"settings": st, "name": "Rollout-CL"}),
            ("CF", {"settings": st, "name": "Costate-FB"}),
            ("MPC", {"forecast": "persistence", "settings": st}),
            ("MPC", {"forecast": "learned", "settings": st}),
            ("MPC", {"forecast": "persistence", "settings": MPCSettings(rho=rB * Tb["best"]["FBfix"], e_shift=eb, alloc="fixed"),
                     "name": "MPC-FB/fixed*"}),
            ("MPC", {"forecast": "learned", "settings": MPCSettings(rho=rB, e_shift=eb, alloc="rollout"),
                     "name": "MPC-FF/rollout"}),
            ("THC", {**Tb["best"]["THCcut"], "hyst": 5.0, "gate": True, "e_shift": eb, "name": "THC/cutoff*"}),
            ("RCL", {"settings": MPCSettings(rho=rB * rcl_fac(), e_shift=eb), "name": "RCL/fixed*"})]


def rcl_fac():
    return json.load(open(out("e12/budget_rcl_tuning.json")))["RCLfix"]


def run_configs(configs, specs, raw_path, seeds):
    nom = Plant()
    dfs = [pd.read_csv(raw_path)] if os.path.exists(raw_path) else []
    done = set(dfs[0].config) if dfs else set()
    for name, tp, sc, se in configs:
        if name in done:
            continue
        truth = replace(nom, **tp)
        scfg = replace(ScenarioConfig(), **sc)
        scfg_s = replace(SensorConfig(), **se)
        fc = train_forecaster(truth=truth, model=nom, scfg=scfg, sens_cfg=scfg_s)
        d = pd.DataFrame(evaluate(specs, seeds, scfg=scfg, truth=truth, model=nom, sens_cfg=scfg_s, forecaster=fc))
        d["config"] = name
        dfs.append(d)
        pd.concat(dfs).to_csv(raw_path, index=False)
        log(f"{os.path.basename(raw_path)}: {name} done")
    return normalise(pd.concat(dfs))


def stage_robust2():
    T, Tb = load_tuning(), json.load(open(out("e12/budget_tuning.json")))
    df = run_configs(structural_configs(), robust_specs(T, Tb, T2()), out("e12/robust2_raw.csv"), SMALL_SEEDS)
    df.to_csv(out("e12/robust2.csv"), index=False)


def stage_e4b():
    T, Tb = load_tuning(), json.load(open(out("e12/budget_tuning.json")))
    rB = T["rho_B"]["0.25"]; eb = 0.25 * E_FULL
    specs = [("Off", {}), ("AO", {"u": 1.0}), ("THC", dict(T2()["thc"], name="THC")),
             ("RCL", {"settings": MPCSettings(rho=T["rho_iso"]), "name": "Rollout-CL"}),
             ("MPC", {"forecast": "persistence", "settings": MPCSettings(rho=rB * Tb["best"]["FBfix"], e_shift=eb, alloc="fixed"),
                      "name": "MPC-FB/fixed*"}),
             ("THC", {**Tb["best"]["THCcut"], "hyst": 5.0, "gate": True, "e_shift": eb, "name": "THC/cutoff*"}),
             ("RCL", {"settings": MPCSettings(rho=rB * rcl_fac(), e_shift=eb), "name": "RCL/fixed*"})]
    cfgs = [(n, tp, sc, se) for n, tp, sc, se, _ in robustness_configs()]
    df = run_configs(cfgs, specs, out("e12/e4b_raw.csv"), SMALL_SEEDS)
    df.to_csv(out("e12/e4b.csv"), index=False)


def mc_configs(n=100, seed=31):
    rng = np.random.default_rng(seed)
    nom = Plant()
    C = []
    for i in range(n):
        u = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
        w = rng.dirichlet([4.0, 3.5, 2.5])
        tp = {"q_scale": u(0.5, 1.25), "lam0": nom.lam0 * u(0.5, 2.0), "k_dep": nom.k_dep * u(0.5, 2.0),
              "eta": np.minimum(nom.eta * np.array([u(0.8, 1.0)] * 2 + [1, 1, 1, 1]), 0.99),
              "f_ab": nom.f_ab * u(0.25, 4.0), "f_bc": nom.f_bc * u(0.25, 4.0), "f_ac": nom.f_ac * u(0.25, 4.0),
              "w_in": w, "height": float(rng.uniform(2.5, 4.0)), "fan": FanModel(p_exp=float(rng.uniform(2.5, 3.0)))}
        lo = u(5.0, 60.0)
        sc = {"onset_sd": float(rng.uniform(0, 15)) / 60, "amp_sd": float(rng.uniform(0.15, 0.6)),
              "unsched_rate": float(rng.uniform(0, 6)), "pm_out_scale": u(0.25, 2.0)}
        se = {"gain_sd": float(rng.uniform(0.02, 0.2)), "noise_scale": float(rng.uniform(0.5, 3.0))}
        C.append((f"mc{i:03d}", tp, sc, se))
    return C


def stage_mc():
    T = load_tuning()
    st = MPCSettings(rho=T["rho_iso"])
    specs = [("Off", {}), ("AO", {"u": 1.0}), ("Relay", {}), ("THC", dict(T2()["thc"], name="THC")),
             ("RCL", {"settings": st, "name": "Rollout-CL"}),
             ("MPC", {"forecast": "learned", "settings": st})]
    nom = Plant()
    raw = out("e12/mc_raw.csv")
    dfs = [pd.read_csv(raw)] if os.path.exists(raw) else []
    done = set(dfs[0].config) if dfs else set()
    for i, (name, tp, sc, se) in enumerate(mc_configs()):
        if name in done:
            continue
        truth = replace(nom, **tp)
        scfg = replace(ScenarioConfig(), **sc)
        scfg_s = replace(SensorConfig(), **se)
        fc = train_forecaster(truth=truth, model=nom, scfg=scfg, sens_cfg=scfg_s, seeds=list(range(5)))
        seeds = list(range(5000 + 10 * i, 5010 + 10 * i))
        d = pd.DataFrame(evaluate(specs, seeds, scfg=scfg, truth=truth, model=nom, sens_cfg=scfg_s, forecaster=fc))
        d["config"] = name
        dfs.append(d)
        if (i + 1) % 10 == 0:
            pd.concat(dfs).to_csv(raw, index=False)
            log(f"mc {i + 1} draws done")
    df = pd.concat(dfs)
    df.to_csv(raw, index=False)
    normalise(df).to_csv(out("e12/mc.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(out("e12"), exist_ok=True)
    stages = sys.argv[1:]
    fns = {"base2": stage_base2, "budget": stage_budget, "e7roll": stage_e7roll, "robust2": stage_robust2,
           "e4b": stage_e4b, "mc": stage_mc, "budgetrcl": stage_budgetrcl, "pisd2": stage_pisd2}
    for s in stages:
        log(f"stage {s}")
        fns[s]()
