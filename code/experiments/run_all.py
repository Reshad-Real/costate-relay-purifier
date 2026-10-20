"""Run the simulation study. Usage: python run_all.py <stage> [<stage> ...]
Stages: tune e1 e2 e3 e4 e5 e7 e8 e9  (e6 exactness/timing has its own scripts)."""
import sys, os, time, json
from common import *                                   # noqa: F401,F403
from aqmpc.params import Plant, MPCSettings, PHI, PSI, FanModel
from aqmpc.scenarios import ScenarioConfig
from aqmpc.sensors import SensorConfig
from aqmpc.harness import evaluate, train_forecaster
from dataclasses import replace


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------ tuning (held-out seeds)
def stage_tune():
    pl = Plant()
    fc = train_forecaster(pl)
    rows = evaluate([("Relay", {})], TUNE_SEEDS, truth=pl)
    e_relay = float(np.mean([r["energy_Wh"] for r in rows]))
    # rho_iso: MPC-FF energy matches the relay on the tuning seeds
    grid = [0.01, 0.015, 0.02, 0.03, 0.045, 0.07]
    specs = [("MPC", {"forecast": "learned", "settings": MPCSettings(rho=r), "name": f"r{r}"}) for r in grid]
    df = pd.DataFrame(evaluate(specs, TUNE_SEEDS, truth=pl, forecaster=fc))
    e = df.groupby("policy").energy_Wh.mean()
    eg = np.array([e[f"r{r}"] for r in grid])
    rho_iso = float(np.exp(np.interp(-e_relay, -eg, np.log(grid))))
    log(f"relay energy {e_relay:.2f} Wh; MPC-FF energies {np.round(eg, 2)} -> rho_iso {rho_iso:.4f}")
    # PI and timer tuned to minimise J = CE + rho_iso * E
    pi_specs = [("PI", {"r25": r, "r10": 2.4 * r, "kp": kp, "ki": ki})
                for r in [15, 25, 35, 50, 70] for kp in [0.25, 0.5, 1.0, 2.0] for ki in [1.0, 2.0, 6.0, 12.0]]
    dpi = pd.DataFrame(evaluate(pi_specs, TUNE_SEEDS, truth=pl))
    dpi["key"] = [json.dumps(s[1]) for s in pi_specs for _ in TUNE_SEEDS]
    dpi["J"] = dpi.CE + rho_iso * dpi.energy_Wh
    pi_best = json.loads(dpi.groupby("key").J.mean().idxmin())
    tm_specs = [("Timer", {"lead": l / 60, "tail": t / 60, "u_ff": u})
                for l in [0, 5, 10, 20, 30] for t in [0, 10, 20, 40] for u in [0.8, 1.0]]
    dtm = pd.DataFrame(evaluate(tm_specs, TUNE_SEEDS, truth=pl))
    dtm["key"] = [json.dumps(s[1]) for s in tm_specs for _ in TUNE_SEEDS]
    dtm["J"] = dtm.CE + rho_iso * dtm.energy_Wh
    tm_best = json.loads(dtm.groupby("key").J.mean().idxmin())
    # budget rho_B for CF so that mean energy ~= budget, for each budget fraction
    rho_B = {}
    cf_grid = list(np.exp(np.linspace(np.log(0.004), np.log(0.4), 14)))
    cf_specs = [("CF", {"settings": MPCSettings(rho=float(r)), "name": f"c{r:.5f}"}) for r in cf_grid]
    dcf = pd.DataFrame(evaluate(cf_specs, TUNE_SEEDS, truth=pl))
    ecf = dcf.groupby("policy").energy_Wh.mean()
    ec = np.array([ecf[f"c{r:.5f}"] for r in cf_grid])
    for frac in [0.15, 0.25, 0.35, 0.45]:
        rho_B[str(frac)] = float(np.exp(np.interp(-frac * E_FULL, -ec, np.log(cf_grid))))
    tun = {"e_relay_tune": e_relay, "rho_iso": rho_iso, "pi": pi_best, "timer": tm_best, "rho_B": rho_B,
           "pi_grid_J": dpi.groupby("key").J.mean().to_dict(), "timer_grid_J": dtm.groupby("key").J.mean().to_dict()}
    json.dump(tun, open(out("tuning.json"), "w"), indent=2)
    log(f"tuning: {json.dumps({k: v for k, v in tun.items() if 'grid' not in k})}")


# ------------------------------------------------------------------ E1 main comparison
def policies_main(T):
    st = MPCSettings(rho=T["rho_iso"])
    return [("Off", {}), ("Relay", {}), ("Relay-PM", {}), ("AO", {"u": 0.8}), ("AO", {"u": 1.0}),
            ("PI", T["pi"]), ("Timer", T["timer"]),
            ("CF", {"settings": st, "name": "Costate-FB"}),
            ("MPC", {"forecast": "persistence", "settings": st}),
            ("MPC", {"forecast": "learned", "settings": st}),
            ("MPC", {"forecast": "oracle", "settings": st})]


def stage_e1(scale=1.0, tag="e1"):
    T = load_tuning()
    pl = Plant(q_scale=scale)
    fc = train_forecaster(pl)
    rows = evaluate(policies_main(T), TEST_SEEDS, truth=pl, forecaster=fc)
    df = normalise(pd.DataFrame(rows))
    df["rho"] = T["rho_iso"]
    df.to_csv(out(f"{tag}/main.csv"), index=False)
    log(f"{tag} done: {len(df)} rows")


def stage_e1b():
    """Occupancy-gated relay on the E1 test shifts (normalised with E1's Off and AO-100 runs)."""
    pl = Plant()
    rows = pd.DataFrame(evaluate([("Relay-occ", {})], TEST_SEEDS, truth=pl))
    main = pd.read_csv(out("e1/main.csv"))
    off = main[main.policy == "Off"].set_index("seed").CE
    ao = main[main.policy == "AO-100"].set_index("seed").CE
    rows["captured"] = [(off[s] - c) / (off[s] - ao[s]) for s, c in zip(rows.seed, rows.CE)]
    rows["Efrac"] = rows.energy_Wh / E_FULL
    rows.to_csv(out("e1/relay_occ.csv"), index=False)
    log("e1b done")


# ------------------------------------------------------------------ E2 exposure-energy fronts
def stage_e2():
    T = load_tuning()
    for scale in [1.0, 4.0]:
        if os.path.exists(out(f"e2/fronts_scale{scale}.csv")):
            log(f"e2 scale {scale} exists, skipped")
            continue
        pl = Plant(q_scale=scale)
        fc = train_forecaster(pl)
        specs = [("Off", {}), ("Relay", {}), ("Relay-PM", {}), ("AO", {"u": 0.8}), ("AO", {"u": 0.9}),
                 ("AO", {"u": 1.0}), ("Timer", T["timer"])]
        for r in [10, 15, 20, 30, 45, 70]:
            p = dict(T["pi"]); p["r25"] = r; p["r10"] = 2.4 * r
            specs.append(("PI", p))
        for rho in RHO_GRID:
            st = MPCSettings(rho=rho)
            specs.append(("CF", {"settings": st, "name": f"Costate-FB|{rho}"}))
            for f, nm in [("persistence", "MPC-FB"), ("learned", "MPC-FF"), ("oracle", "MPC-oracle")]:
                specs.append(("MPC", {"forecast": f, "settings": st, "name": f"{nm}|{rho}"}))
        rows = evaluate(specs, FRONT_SEEDS, truth=pl, forecaster=fc)
        df = pd.DataFrame(rows)
        # PI rows share a name; label them by setpoint
        pi_r = [s[1]["r25"] for s in specs if s[0] == "PI"]
        m = df.policy == "PI"
        df.loc[m, "policy"] = [f"PI|{r}" for r in pi_r for _ in FRONT_SEEDS]
        df = normalise(df)
        df.to_csv(out(f"e2/fronts_scale{scale}.csv"), index=False)
        log(f"e2 scale {scale} done")


# ------------------------------------------------------------------ E3 energy-budget allocation
def budget_policies(T, frac):
    rB = T["rho_B"][str(frac)]
    eb = frac * E_FULL
    sp = [("Off", {}), ("AO", {"u": 1.0}), ("Relay", {}), ("RelayBudget", {"e_shift": eb})]
    for alloc in ["fixed", "pacing", "rollout"]:
        st = MPCSettings(rho=rB, e_shift=eb, alloc=alloc)
        sp.append(("CF", {"settings": st, "name": f"Costate-FB/{alloc}", "use_forecaster": alloc == "rollout"}))
        sp.append(("MPC", {"forecast": "persistence", "settings": st, "name": f"MPC-FB/{alloc}"}))
        sp.append(("MPC", {"forecast": "learned", "settings": st, "name": f"MPC-FF/{alloc}"}))
    sp.append(("MPC", {"forecast": "oracle", "settings": MPCSettings(rho=rB, e_shift=eb, alloc="rollout"),
                       "name": "MPC-oracle/rollout"}))
    return sp


def stage_e3():
    T = load_tuning()
    pl = Plant()
    fc = train_forecaster(pl)
    rows = evaluate(budget_policies(T, 0.25), TEST_SEEDS, truth=pl, forecaster=fc)
    df = normalise(pd.DataFrame(rows)); df["budget_frac"] = 0.25
    df.to_csv(out("e3/budget_main.csv"), index=False)
    log("e3 main done")
    dfs = []
    for frac in [0.15, 0.35, 0.45]:
        rows = evaluate(budget_policies(T, frac), FRONT_SEEDS, truth=pl, forecaster=fc)
        d = pd.DataFrame(rows); d["budget_frac"] = frac
        dfs.append(normalise(d))
        log(f"e3 budget {frac} done")
    pd.concat(dfs).to_csv(out("e3/budget_sweep.csv"), index=False)


# ------------------------------------------------------------------ E4 robustness
def robustness_configs():
    nom = Plant()
    C = [("nominal", {}, {}, {}, {})]
    for f in [0.5, 0.75, 1.25]:
        C.append((f"Q x{f}", {"q_scale": f}, {}, {}, {}))
    for f in [0.5, 2.0]:
        C.append((f"lam0 x{f}", {"lam0": nom.lam0 * f}, {}, {}, {}))
    for f in [0.5, 2.0]:
        C.append((f"k_dep x{f}", {"k_dep": nom.k_dep * f}, {}, {}, {}))
    C.append(("eta_PM x0.8", {"eta": nom.eta * np.array([0.8, 0.8, 1, 1, 1, 1])}, {}, {}, {}))
    for f in [0.25, 4.0]:
        C.append((f"interzonal x{f}", {"f_ab": nom.f_ab * f, "f_bc": nom.f_bc * f, "f_ac": nom.f_ac * f}, {}, {}, {}))
    C.append(("intake (0.6,0.25,0.15)", {"w_in": np.array([0.6, 0.25, 0.15])}, {}, {}, {}))
    C.append(("height 2.5 m", {"height": 2.5}, {}, {}, {}))
    C.append(("height 4.0 m", {"height": 4.0}, {}, {}, {}))
    C.append(("fan exponent 2.5", {"fan": FanModel(p_exp=2.5)}, {}, {}, {}))
    rng = np.random.default_rng(77)
    for j in range(3):
        f = np.exp(rng.uniform(np.log(0.7), np.log(1.3), 6))
        C.append((f"joint +-30% #{j+1}", {"q_scale": f[0], "lam0": nom.lam0 * f[1], "k_dep": nom.k_dep * f[2],
                                           "eta": np.minimum(nom.eta * f[3], 0.99), "f_ab": nom.f_ab * f[4],
                                           "f_bc": nom.f_bc * f[4], "f_ac": nom.f_ac * f[4], "height": 3.0 * f[5]},
                  {}, {}, {}))
    C.append(("onset sd 0", {}, {"onset_sd": 0.0}, {}, {}))
    C.append(("onset sd 15 min", {}, {"onset_sd": 15 / 60}, {}, {}))
    C.append(("amplitude sd 0.6", {}, {"amp_sd": 0.6}, {}, {}))
    C.append(("cancel p 0.2", {}, {"cancel_p": 0.2}, {}, {}))
    C.append(("unscheduled 3/shift", {}, {"unsched_rate": 3.0}, {}, {}))
    C.append(("unscheduled 6/shift", {}, {"unsched_rate": 6.0}, {}, {}))
    C.append(("sensor gain sd 0.2", {}, {}, {"gain_sd": 0.2}, {}))
    C.append(("sensor noise x3", {}, {}, {"noise_scale": 3.0}, {}))
    C.append(("delay 5 min + lag x4", {}, {}, {"delay_steps": 30, "lag_scale": 4.0}, {}))
    C.append(("eCO2 cross-sens 0", {}, {}, {"eco2_xsens": 0.0}, {}))
    C.append(("outdoor x0.25", {}, {"pm_out_scale": 0.25}, {}, {}))
    C.append(("outdoor x2", {}, {"pm_out_scale": 2.0}, {}, {}))
    return C


def stage_e4():
    T = load_tuning()
    nom = Plant()
    st = MPCSettings(rho=T["rho_iso"])
    rB = T["rho_B"]["0.25"]; eb = 0.25 * E_FULL
    specs = [("Off", {}), ("AO", {"u": 1.0}), ("Relay", {}),
             ("CF", {"settings": st, "name": "Costate-FB"}),
             ("MPC", {"forecast": "persistence", "settings": st}),
             ("MPC", {"forecast": "learned", "settings": st}),
             ("CF", {"settings": MPCSettings(rho=rB, e_shift=eb, alloc="fixed"), "name": "Costate-FB/fixed"}),
             ("MPC", {"forecast": "persistence", "settings": MPCSettings(rho=rB, e_shift=eb, alloc="rollout"),
                      "name": "MPC-FB/rollout"}),
             ("MPC", {"forecast": "learned", "settings": MPCSettings(rho=rB, e_shift=eb, alloc="rollout"),
                      "name": "MPC-FF/rollout"})]
    raw = out("e4/robustness_raw.csv")
    dfs = [pd.read_csv(raw)] if os.path.exists(raw) else []
    done = set(dfs[0].config) if dfs else set()
    for name, tp, sc, se, _ in robustness_configs():
        if name in done:
            continue
        truth = replace(nom, **tp)
        scfg = replace(ScenarioConfig(), **sc)
        scfg_s = replace(SensorConfig(), **se)
        fc = train_forecaster(truth=truth, model=nom, scfg=scfg, sens_cfg=scfg_s)
        rows = evaluate(specs, SMALL_SEEDS, scfg=scfg, truth=truth, model=nom, sens_cfg=scfg_s, forecaster=fc)
        d = pd.DataFrame(rows); d["config"] = name
        dfs.append(d)
        log(f"e4 {name} done")
        pd.concat(dfs).to_csv(raw, index=False)
        pd.concat(dfs).pipe(normalise).to_csv(out("e4/robustness.csv"), index=False)


# ------------------------------------------------------------------ E5 weight sensitivity
def weight_sets(n=30, conc=(20.0, 2.0), seed=11):
    rng = np.random.default_rng(seed)
    S = [("nominal", PHI, PSI), ("PM only", np.array([0.6, 0.4, 0, 0, 0, 0]), PSI),
         ("equal pollutants", np.ones(6) / 6, PSI), ("equal zones", PHI, np.ones(3) / 3),
         ("gases emphasised", np.array([0.1, 0.05, 0.3, 0.05, 0.2, 0.3]), PSI),
         ("zone C emphasised", PHI, np.array([0.2, 0.2, 0.6]))]
    for a in conc:
        for j in range(n // len(conc)):
            S.append((f"Dir(a={a})#{j}", rng.dirichlet(a * PHI), rng.dirichlet(a * PSI)))
    return S


def stage_e5():
    T = load_tuning()
    pl = Plant()
    fc = train_forecaster(pl)
    S = weight_sets()
    alt = [(ph, ps) for _, ph, ps in S]
    st = MPCSettings(rho=T["rho_iso"])
    specs = [("Off", {}), ("AO", {"u": 1.0}), ("Relay", {}), ("AO", {"u": 0.8}),
             ("CF", {"settings": st, "name": "Costate-FB"}),
             ("MPC", {"forecast": "learned", "settings": st})]
    rows = evaluate(specs, SMALL_SEEDS, truth=pl, forecaster=fc, metric_kw={"alt_weights": alt})
    recs = []
    for r in rows:
        for (nm, _, _), ce in zip(S, r["CE_alt"]):
            recs.append({"seed": r["seed"], "policy": r["policy"], "weights": nm, "CE": ce,
                         "energy_Wh": r["energy_Wh"], "design": "nominal"})
    # re-designed controllers: MPC optimised for the evaluation weights
    from aqmpc.exposure import weight_tensor
    for nm, ph, ps in S[:6]:
        Wd = weight_tensor(ph, ps)
        sp = [("MPC", {"forecast": "learned", "settings": st, "W": Wd, "name": "MPC-FF(redesigned)"})]
        rr = evaluate(sp, SMALL_SEEDS, truth=pl, forecaster=fc, metric_kw={"alt_weights": [(ph, ps)]})
        for r in rr:
            recs.append({"seed": r["seed"], "policy": r["policy"], "weights": nm, "CE": r["CE_alt"][0],
                         "energy_Wh": r["energy_Wh"], "design": "matched"})
        log(f"e5 redesign {nm} done")
    df = pd.DataFrame(recs)
    df.to_csv(out("e5/weights.csv"), index=False)
    json.dump([(nm, list(map(float, ph)), list(map(float, ps))) for nm, ph, ps in S],
              open(out("e5/weight_sets.json"), "w"), indent=1)
    log("e5 done")


# ------------------------------------------------------------------ E7 energy-budget formulation
def stage_e7():
    pl = Plant()
    fc = train_forecaster(pl)
    specs = []
    for frac in [0.15, 0.25, 0.35]:
        for ub in [True, False]:
            st = MPCSettings(rho=0.002, e_shift=frac * E_FULL, use_budget_state=ub)
            specs.append(("MPC", {"forecast": "learned", "settings": st,
                                   "name": f"{'budget-state' if ub else 'per-horizon cap'}|{frac}"}))
    df = pd.DataFrame(evaluate(specs, SMALL_SEEDS, truth=pl, forecaster=fc))
    df["budget_Wh"] = [float(p.split("|")[1]) * E_FULL for p in df.policy]
    df.to_csv(out("e7/budget_formulation.csv"), index=False)
    log("e7 done")


# ------------------------------------------------------------------ E8 horizon and blocking
def stage_e8():
    T = load_tuning()
    pl = Plant()
    fc = train_forecaster(pl)
    specs = [("Off", {}), ("AO", {"u": 1.0})]
    for N in [10, 20, 30, 45, 60]:
        for term in [True, False]:
            st = MPCSettings(rho=T["rho_iso"], N=N, Nu=min(6, N), terminal=term)
            specs.append(("MPC", {"forecast": "learned", "settings": st, "name": f"N={N}|Nu=6|term={int(term)}"}))
    for Nu in [3, 4, 5, 7]:
        st = MPCSettings(rho=T["rho_iso"], N=30, Nu=Nu)
        specs.append(("MPC", {"forecast": "learned", "settings": st, "name": f"N=30|Nu={Nu}|term=1"}))
    df = normalise(pd.DataFrame(evaluate(specs, SMALL_SEEDS, truth=pl, forecaster=fc)))
    df.to_csv(out("e8/horizon.csv"), index=False)
    log("e8 done")


# ------------------------------------------------------------------ E9 forecast accuracy / circularity
def stage_e9():
    from forecast_eval import run as fe_run
    fe_run()
    log("e9 done")


if __name__ == "__main__":
    for st in sys.argv[1:]:
        t0 = time.time()
        log(f"=== stage {st}")
        globals()[f"stage_{st}"]()
        log(f"=== stage {st} finished in {time.time() - t0:.0f}s")
