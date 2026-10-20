"""E11: stronger baselines for the ISA Transactions version.

1. Tuning on TUNE_SEEDS (same cost J = CE + rho_iso*E as PI and the timer in stage_tune):
   - THC: tuned multilevel threshold controller with hysteresis and optional occupancy gating
   - MPC-track: conventional zone-tracking MPC (setpoint r25, weight q)
   - PI and timer re-tuned on wider grids (the earlier optima were on grid edges)
2. Test on TEST_SEEDS (paired with E1; normalised with E1's Off and AO-100 runs):
   THC*, MPC-track*, MPC-RR (relax-and-round economic MPC), PI*, Timer*
3. Fronts on FRONT_SEEDS: THC threshold sweep and MPC-track weight sweep (as-built purifier).
Usage: python e11_baselines.py [tune|test|front|all]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from aqmpc.params import Plant, MPCSettings
from aqmpc.harness import evaluate, train_forecaster
from common import out, load_tuning, TUNE_SEEDS, TEST_SEEDS, FRONT_SEEDS, E_FULL


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def grid_best(specs, seeds, pl, fc, rho, tag):
    df = pd.DataFrame(evaluate(specs, seeds, truth=pl, forecaster=fc))
    df["key"] = [json.dumps(s[1], default=float) for s in specs for _ in seeds]
    df["J"] = df.CE + rho * df.energy_Wh
    g = df.groupby("key").agg(J=("J", "mean"), E=("energy_Wh", "mean"), CE=("CE", "mean"))
    g.to_csv(out(f"e11/grid_{tag}.csv"))
    best = json.loads(g.J.idxmin())
    log(f"{tag}: best {best}  J {g.J.min():.4f}")
    return best, g


def edge_note(best, grids):
    return {k: ("edge" if best[k] in (min(v), max(v)) and len(v) > 1 else "interior") for k, v in grids.items()}


def stage_tune():
    T = load_tuning()
    pl = Plant()
    fc = train_forecaster(pl)
    rho = T["rho_iso"]
    st = MPCSettings(rho=rho)
    res = {}
    # THC
    gt = {"th1": [10, 15, 25, 35, 45, 60], "dth": [5, 10, 20, 40], "hyst": [0, 5, 10, 20], "gate": [True, False]}
    specs = [("THC", {"th1": a, "dth": b, "hyst": h, "gate": g})
             for a in gt["th1"] for b in gt["dth"] for h in gt["hyst"] for g in gt["gate"]]
    res["thc"], _ = grid_best(specs, TUNE_SEEDS, pl, fc, rho, "thc")
    res["thc_edges"] = edge_note(res["thc"], gt)
    # PI (wider integral-gain grid)
    gp = {"r25": [25, 35, 50], "kp": [0.5, 1.0, 2.0], "ki": [0.1, 0.25, 0.5, 1.0, 2.0]}
    specs = [("PI", {"r25": r, "r10": 2.4 * r, "kp": kp, "ki": ki})
             for r in gp["r25"] for kp in gp["kp"] for ki in gp["ki"]]
    res["pi"], _ = grid_best(specs, TUNE_SEEDS, pl, fc, rho, "pi")
    res["pi_edges"] = edge_note(res["pi"], gp)
    # timer (longer lead and tail)
    gm = {"lead": [20 / 60, 30 / 60, 45 / 60, 60 / 60, 90 / 60], "tail": [20 / 60, 40 / 60, 60 / 60, 90 / 60, 120 / 60],
          "u_ff": [0.8, 0.9]}
    specs = [("Timer", {"lead": l, "tail": t, "u_ff": u}) for l in gm["lead"] for t in gm["tail"] for u in gm["u_ff"]]
    res["timer"], _ = grid_best(specs, TUNE_SEEDS, pl, fc, rho, "timer")
    res["timer_edges"] = edge_note(res["timer"], gm)
    # tracking MPC
    gk = {"r25": [10, 15, 25, 35], "q": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}
    specs = [("MPC-track", {"settings": st, "q": q, "r25": r}) for r in gk["r25"] for q in gk["q"]]
    df = pd.DataFrame(evaluate(specs, TUNE_SEEDS, truth=pl, forecaster=fc))
    df["key"] = [json.dumps({"q": s[1]["q"], "r25": s[1]["r25"]}) for s in specs for _ in TUNE_SEEDS]
    df["J"] = df.CE + rho * df.energy_Wh
    g = df.groupby("key").agg(J=("J", "mean"), E=("energy_Wh", "mean"), CE=("CE", "mean"))
    g.to_csv(out("e11/grid_track.csv"))
    res["track"] = json.loads(g.J.idxmin())
    res["track_edges"] = edge_note(res["track"], gk)
    log(f"track: best {res['track']}  J {g.J.min():.4f}")
    json.dump(res, open(out("e11/tuning2.json"), "w"), indent=2)
    log(f"tuning done: {json.dumps(res)}")


def stage_test():
    T = load_tuning()
    T2 = json.load(open(out("e11/tuning2.json")))
    pl = Plant()
    fc = train_forecaster(pl)
    st = MPCSettings(rho=T["rho_iso"])
    specs = [("THC", T2["thc"]), ("PI", T2["pi"]), ("Timer", T2["timer"]),
             ("MPC-track", {"settings": st, **T2["track"]}), ("MPC-RR", {"settings": st})]
    rows = pd.DataFrame(evaluate(specs, TEST_SEEDS, truth=pl, forecaster=fc))
    main = pd.read_csv(out("e1/main.csv"))
    off = main[main.policy == "Off"].set_index("seed").CE
    ao = main[main.policy == "AO-100"].set_index("seed").CE
    rows["captured"] = [(off[s] - c) / (off[s] - ao[s]) for s, c in zip(rows.seed, rows.CE)]
    rows["Efrac"] = rows.energy_Wh / E_FULL
    rows.loc[rows.policy == "PI", "policy"] = "PI*"
    rows.loc[rows.policy == "Timer+Relay-PM", "policy"] = "Timer*"
    rows.to_csv(out("e11/test.csv"), index=False)
    log(f"test done: {rows.groupby('policy')[['captured', 'Efrac', 'switches']].mean().round(3).to_dict()}")


def stage_refine():
    """Second, finer grid around the THC and PI optima (both first-grid optima lay on a grid edge);
    re-runs the test shifts for the refined settings and replaces their rows in e11/test.csv."""
    T = load_tuning()
    T2 = json.load(open(out("e11/tuning2.json")))
    pl = Plant()
    fc = train_forecaster(pl)
    rho = T["rho_iso"]
    gt = {"th1": [20, 25, 30], "dth": [30, 40, 60, 80, 120], "hyst": [2.5, 5, 7.5], "gate": [True]}
    specs = [("THC", {"th1": a, "dth": b, "hyst": h, "gate": g})
             for a in gt["th1"] for b in gt["dth"] for h in gt["hyst"] for g in gt["gate"]]
    thc, g_thc = grid_best(specs, TUNE_SEEDS, pl, fc, rho, "thc_refine")
    gm = {"lead": [35 / 60, 45 / 60, 55 / 60], "tail": [0.0, 10 / 60, 20 / 60, 30 / 60], "u_ff": [0.8]}
    specs = [("Timer", {"lead": l, "tail": t, "u_ff": u}) for l in gm["lead"] for t in gm["tail"] for u in gm["u_ff"]]
    tm, g_tm = grid_best(specs, TUNE_SEEDS, pl, fc, rho, "timer_refine")
    # tracking MPC: the first-grid optimum (q = 0.03, r25 = 10) was on the lower edge of both axes
    st = MPCSettings(rho=rho)
    gk = {"r25": [5.0, 7.5, 10.0, 12.5], "q": [0.003, 0.01, 0.03]}
    specs = [("MPC-track", {"settings": st, "q": q, "r25": r}) for r in gk["r25"] for q in gk["q"]]
    df = pd.DataFrame(evaluate(specs, TUNE_SEEDS, truth=pl, forecaster=fc))
    df["key"] = [json.dumps({"q": s[1]["q"], "r25": s[1]["r25"]}) for s in specs for _ in TUNE_SEEDS]
    df["J"] = df.CE + rho * df.energy_Wh
    g_trk = df.groupby("key").agg(J=("J", "mean"), E=("energy_Wh", "mean"), CE=("CE", "mean"))
    g_trk.to_csv(out("e11/grid_track_refine.csv"))
    trk = json.loads(g_trk.J.idxmin())
    old = (pd.read_csv(out("e11/grid_thc.csv")).set_index("key").J.min(),
           pd.read_csv(out("e11/grid_track.csv")).set_index("key").J.min(),
           pd.read_csv(out("e11/grid_timer.csv")).set_index("key").J.min())
    changed = []
    if g_tm.J.min() < old[2] - 1e-9:
        T2["timer"] = tm; T2["timer_edges"] = edge_note(tm, gm); changed.append("Timer")
    if g_thc.J.min() < old[0] - 1e-9:
        T2["thc"] = thc; T2["thc_edges"] = edge_note(thc, gt); changed.append("THC")
    if g_trk.J.min() < old[1] - 1e-9:
        T2["track"] = trk; T2["track_edges"] = edge_note(trk, gk); changed.append("MPC-track")
    json.dump(T2, open(out("e11/tuning2.json"), "w"), indent=2)
    log(f"refine: changed {changed}; THC {T2['thc']} track {T2['track']}")
    if not changed:
        return
    specs = [("THC", T2["thc"])] if "THC" in changed else []
    specs += [("MPC-track", {"settings": st, **T2["track"]})] if "MPC-track" in changed else []
    specs += [("Timer", T2["timer"])] if "Timer" in changed else []
    rows = pd.DataFrame(evaluate(specs, TEST_SEEDS, truth=pl, forecaster=fc))
    main = pd.read_csv(out("e1/main.csv"))
    off = main[main.policy == "Off"].set_index("seed").CE
    ao = main[main.policy == "AO-100"].set_index("seed").CE
    rows["captured"] = [(off[s] - c) / (off[s] - ao[s]) for s, c in zip(rows.seed, rows.CE)]
    rows["Efrac"] = rows.energy_Wh / E_FULL
    rows.loc[rows.policy == "PI", "policy"] = "PI*"
    rows.loc[rows.policy == "Timer+Relay-PM", "policy"] = "Timer*"
    t = pd.read_csv(out("e11/test.csv"))
    t = pd.concat([t[~t.policy.isin(rows.policy.unique())], rows], ignore_index=True)
    t.to_csv(out("e11/test.csv"), index=False)
    log("refined test rows written")


def stage_front():
    T2 = json.load(open(out("e11/tuning2.json")))
    T = load_tuning()
    pl = Plant()
    fc = train_forecaster(pl)
    thc = dict(T2["thc"])
    specs = []
    th_grid = [8, 12, 18, 25, 35, 50, 70, 100]
    for a in th_grid:
        thc_a = dict(thc, th1=a, name=f"THC|{a}")
        specs.append(("THC", thc_a))
    q_grid = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    st = MPCSettings(rho=T["rho_iso"])
    for q in q_grid:
        specs.append(("MPC-track", {"settings": st, "q": q, "r25": T2["track"]["r25"], "name": f"TRK|{q}"}))
    rows = pd.DataFrame(evaluate(specs, FRONT_SEEDS, truth=pl, forecaster=fc))
    main = pd.read_csv(out("e1/main.csv"))
    off = main[main.policy == "Off"].set_index("seed").CE
    ao = main[main.policy == "AO-100"].set_index("seed").CE
    rows["captured"] = [(off[s] - c) / (off[s] - ao[s]) for s, c in zip(rows.seed, rows.CE)]
    rows["Efrac"] = rows.energy_Wh / E_FULL
    rows.to_csv(out("e11/fronts.csv"), index=False)
    log("fronts done")


if __name__ == "__main__":
    stages = sys.argv[1:] or ["all"]
    if "all" in stages:
        stages = ["tune", "test", "refine", "front"]
    for s in stages:
        log(f"stage {s}")
        {"tune": stage_tune, "test": stage_test, "refine": stage_refine, "front": stage_front}[s]()
