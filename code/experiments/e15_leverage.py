"""E15: clairvoyant reference optima and the value of preview as a function of actuator leverage.

Stages
  sweep   purifier flow scaled by s in {0.25, 0.5, 1, 2, 4} (the controllers are designed for the scaled purifier,
          rho_s = s * rho_iso so that the leading-order trade-off keeps its shape); 40 test days per scale; the relay,
          scheduled duty, CL-s, CL-r (estimated and exact information), EMPC-FB, EMPC-oracle (30-min preview) and the
          clairvoyant schedule (whole-day preview)                                  -> results/e15/leverage.csv
  main    clairvoyant schedule on all test days of the main comparison (as-built purifier, rho_iso)
                                                                                     -> results/e15/clair_main.csv
  budget  clairvoyant schedule under the 25 % energy budget (Lagrangian price found by bisection; the reported
          schedule is feasible)                                                      -> results/e15/clair_budget.csv
The clairvoyant schedule is the best of an exact 1-opt local search started from several schedules (constant 80 %
during occupancy, the CL-r and EMPC-oracle schedules where available, and random schedules); it is evaluated with the
closed-loop simulator by playback, exactly like the controllers."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from aqmpc.params import Plant, MPCSettings
from aqmpc.scenarios import make_shift, ScenarioConfig
from aqmpc.simulate import run_shift
from aqmpc.clairvoyant import Clairvoyant, Playback
from aqmpc.controllers import Relay, MPC, CostateFeedback
from aqmpc.extra_controllers import RolloutCostate, ScheduledDuty
from common import out, load_tuning, TEST_SEEDS, SMALL_SEEDS, E_FULL

SCALES = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 2.0, 4.0]      # six leverages up to the testbed's (review of V1)
A0S = [0.25, 0.5, 1.0, 1.5, 2.0]                              # infiltration rates a0 (1/h) for the a0 sweep
WORKERS = int(os.environ.get("AQ_WORKERS", max(1, (os.cpu_count() or 2) - 2)))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def leverage(plant):
    """Relative increase of the slowest PM2.5 decay rate from purifier off to full duty."""
    from aqmpc.params import iPM25
    M0, M1 = plant.rate_matrices()
    a0 = -np.max(np.linalg.eigvals(M0[iPM25]).real)
    a1 = -np.max(np.linalg.eigvals(M0[iPM25] + M1[iPM25]).real)
    return float((a1 - a0) / a0)


def clairvoyant_best(cv, shift, starts_U, rng, n_random):
    base = np.where(shift.occ[:: cv.block][: cv.nb] > 0, 1, 0)
    starts = [("base", base)] + [(k, cv.to_idx(U)) for k, U in starts_U.items()] + \
             [(f"rand{i}", rng.integers(0, cv.nu, cv.nb)) for i in range(n_random)]
    best, rec = None, {}
    for name, s in starts:
        idx, J, it = cv.improve(s)
        rec[name] = J
        if best is None or J < best[1]:
            best = (idx, J, name)
    return best, rec


def sweep_job(a):
    s, seed, rho_iso, sched = a
    pl = replace(Plant(), q_scale=s)
    rho = s * rho_iso
    st = MPCSettings(rho=rho)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    ss = seed + 7919
    rows, traces = [], {}
    ctrls = [("Relay", Relay()), ("Sched", ScheduledDuty(**sched)), ("CL-s", CostateFeedback(pl, st)),
             ("CL-r", RolloutCostate(pl, st)), ("CL-r-exact", RolloutCostate(pl, st, info="oracle", name="CL-r-exact")),
             ("CL-r-exact-pen", RolloutCostate(pl, st, info="oracle", penalty=True, name="CL-r-exact-pen")),
             ("EMPC-FB", MPC(pl, None, "persistence", st)), ("EMPC-oracle", MPC(pl, None, "oracle", st))]
    for name, c in ctrls:
        m, tr = run_shift(sh, c, pl, sensor_seed=ss, return_traces=True)
        traces[name] = tr["U"]
        rows.append(dict(m, policy=name))
    m = run_shift(sh, Playback(np.zeros(len(sh.t)), "Off"), pl, sensor_seed=ss)
    rows.append(dict(m, policy="Off"))
    m = run_shift(sh, Playback(np.ones(len(sh.t)), "AO-100"), pl, sensor_seed=ss)
    rows.append(dict(m, policy="AO-100"))
    cv = Clairvoyant(pl, sh, st.duties, rho)
    (idx, Jc, src), rec = clairvoyant_best(cv, sh, {k: traces[k] for k in ("CL-r-exact", "EMPC-oracle")},
                                           np.random.default_rng(seed), n_random=4)
    m = run_shift(sh, Playback(cv.schedule(idx)), pl, sensor_seed=ss)
    rows.append(dict(m, policy="Clairvoyant", clair_start=src, clair_internal=Jc,
                     clair_spread=max(rec.values()) - min(rec.values())))
    for r in rows:
        r.update(seed=seed, scale=s, rho=rho, J=r["CE"] + rho * r["energy_Wh"])
    return rows


def lambda_job(a):
    """EMPC with the rollout costate as terminal cost (feedback only, and with perfect information)."""
    s, seed, rho_iso, oracle_too = a
    pl = replace(Plant(), q_scale=s)
    rho = s * rho_iso
    st = MPCSettings(rho=rho, terminal_rollout=True)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    rows = []
    variants = [("EMPC-FB-L", "persistence")] + ([("EMPC-oracle-L", "oracle")] if oracle_too else [])
    for name, mode in variants:
        m = run_shift(sh, MPC(pl, None, mode, st, name=name), pl, sensor_seed=seed + 7919)
        rows.append(dict(m, policy=name, seed=seed, scale=s, rho=rho, J=m["CE"] + rho * m["energy_Wh"]))
    return rows


FIRST1 = dict(terminal_rollout=True, blocks=(1, 2, 3, 6, 8, 10), M=0.0)   # EMPC-Lambda as in Theorem 1(b)


def lambda1_job(a):
    """EMPC-Lambda: rollout terminal weight, a first move block of one control interval and no soft-bound penalty."""
    s, seed, rho_iso, oracle_too = a
    pl = replace(Plant(), q_scale=s)
    rho = s * rho_iso
    st = MPCSettings(rho=rho, **FIRST1)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    rows = []
    variants = [("EMPC-FB-L", "persistence")] + ([("EMPC-oracle-L", "oracle")] if oracle_too else [])
    for name, mode in variants:
        m = run_shift(sh, MPC(pl, None, mode, st, name=name), pl, sensor_seed=seed + 7919)
        rows.append(dict(m, policy=name, seed=seed, scale=s, rho=rho, J=m["CE"] + rho * m["energy_Wh"]))
    return rows


def cert_job(a):
    """Optimality certificate: certified lower bound on J* from the clairvoyant schedule, and the certificate that
    CL-r with exact information computes from its own record."""
    s, seed, rho_iso = a
    pl = replace(Plant(), q_scale=s)
    rho = s * rho_iso
    st = MPCSettings(rho=rho)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    m, tr = run_shift(sh, RolloutCostate(pl, st, info="oracle"), pl, sensor_seed=seed + 7919, return_traces=True)
    cv = Clairvoyant(pl, sh, st.duties, rho)
    (idx, Jc, src), rec = clairvoyant_best(cv, sh, {"CL-r-exact": tr["U"]}, np.random.default_rng(seed), n_random=4)
    box = cv.reach_box()
    LB, _, contrib = cv.lower_bound(idx, box)
    ir = cv.to_idx(tr["U"])
    LBr, Jr, _ = cv.lower_bound(ir, box)
    Joff = cv.forward(np.zeros(cv.nb, dtype=int))[1]
    return dict(scale=s, seed=seed, Jc=Jc, LB=LB, Jrex=Jr, Jrex_sim=m["CE"] + rho * m["energy_Wh"],
                beta_rex=Jr - LBr, Joff=Joff, n_active=int(np.sum(contrib < 0)), date=sh.date)


LP_WORKERS = int(os.environ.get("AQ_WORKERS_LP", 3))      # one LP needs ~2 GB at its peak


def certlp_job(a):
    """LP relaxation of the exact clairvoyant MILP (convex-hull form): a certified lower bound on J*."""
    seed, rho_iso = a
    pl = Plant()
    st = MPCSettings(rho=rho_iso)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    cv = Clairvoyant(pl, sh, st.duties, rho_iso)
    return dict(scale=1.0, seed=seed, LB_lp=cv.exact_milp(relax=True, time_limit=1800)[0])


def certlp_long_job(a):
    """The same LP on the days where the default HiGHS call stopped without a solution (status 4, numerical
    difficulty): solved by HiGHS interior point with crossover, or dual simplex if that fails as well."""
    seed, rho_iso = a
    pl = Plant()
    st = MPCSettings(rho=rho_iso)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    cv = Clairvoyant(pl, sh, st.duties, rho_iso)
    for meth in ["highs-ipm", "highs-ds"]:
        J, _, status, _ = cv.exact_milp(relax=True, time_limit=10800, lp_method=meth)
        if J is not None:
            return dict(scale=1.0, seed=seed, LB_lp=J, lp_method=meth)
    return dict(scale=1.0, seed=seed, LB_lp=None, lp_method="failed")


def milpcheck_job(a):
    """Exact MILP optimum on a two-hour window (07:00-09:00) against the batched 1-opt local search."""
    import time as _t
    from aqmpc.scenarios import Shift
    seed, rho_iso = a
    pl = Plant(); rho = rho_iso
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    n = 120 * 6
    sh2 = Shift(sh.t[:n], sh.S[:n], sh.c_out[:n], sh.occ[:n], sh.plan, sh.actual, sh.date)
    cv = Clairvoyant(pl, sh2, MPCSettings().duties, rho)
    (idx, Jc, src), rec = clairvoyant_best(cv, sh2, {}, np.random.default_rng(seed), n_random=4)
    t0 = _t.time()
    Jm, im, status, gap = cv.exact_milp(time_limit=1800)
    return dict(seed=seed, date=sh.date, Jc=Jc, Jmilp=Jm, status=status, t_milp=_t.time() - t0)


def main_job(a):
    seed, rho = a
    pl = Plant()
    st = MPCSettings(rho=rho)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    ss = seed + 7919
    m, tr = run_shift(sh, RolloutCostate(pl, st, info="oracle"), pl, sensor_seed=ss, return_traces=True)
    cv = Clairvoyant(pl, sh, st.duties, rho)
    (idx, Jc, src), rec = clairvoyant_best(cv, sh, {"CL-r-exact": tr["U"]}, np.random.default_rng(seed), n_random=4)
    m = run_shift(sh, Playback(cv.schedule(idx)), pl, sensor_seed=ss)
    return dict(m, seed=seed, policy="Clairvoyant", J=m["CE"] + rho * m["energy_Wh"], clair_internal=Jc,
                clair_start=src, clair_spread=max(rec.values()) - min(rec.values()))


def budget_job(a):
    """Clairvoyant schedule under an energy budget: minimise CE + nu E with the smallest nu whose 1-opt schedule
    meets the budget (bisection on log nu); the returned schedule is feasible."""
    seed, e_budget = a
    pl = Plant()
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    ss = seed + 7919
    duties = MPCSettings().duties
    E_of = lambda cv, idx: float(cv.Pblk[idx].sum())
    lo, hi = np.log(1e-4), np.log(10.0)
    best = None
    warm = None
    for _ in range(26):
        nu = float(np.exp(0.5 * (lo + hi)))
        cv = Clairvoyant(pl, sh, duties, nu)
        base = np.where(sh.occ[:: cv.block][: cv.nb] > 0, 1, 0) if warm is None else warm
        idx, _, _ = cv.improve(base)
        E = E_of(cv, idx)
        if E <= e_budget:
            hi = np.log(nu)
            ce = cv.forward(idx)[1] - nu * E
            if best is None or ce < best[1]:
                best = (idx.copy(), ce, nu)
        else:
            lo = np.log(nu)
        warm = idx
    idx, ce, nu = best
    m = run_shift(sh, Playback(np.repeat(np.asarray(duties)[idx], 6)), pl, sensor_seed=ss)
    return dict(m, seed=seed, policy="Clairvoyant/budget", nu=nu, budget_Wh=e_budget)


BUDGET_SCALES = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0]
BUDGET_FACS = [0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.8, 2.5]


def rcl_budget_job(a):
    """Causal comparator under the budget: CL-r with exact state and current-interval drive and a fixed price."""
    s, seed, price, eb, tag = a
    pl = replace(Plant(), q_scale=s)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    m = run_shift(sh, RolloutCostate(pl, MPCSettings(rho=price, e_shift=eb), info="oracle", name=tag), pl,
                  sensor_seed=seed + 7919)
    return dict(m, seed=seed, scale=s, policy=tag, price=price)


def budget_scaled_job(a):
    seed, eb, s = a
    pl = replace(Plant(), q_scale=s)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    duties = MPCSettings().duties
    lo, hi = np.log(1e-4), np.log(10.0)
    best, warm = None, None
    for _ in range(26):
        nu = float(np.exp(0.5 * (lo + hi)))
        cv = Clairvoyant(pl, sh, duties, nu)
        base = np.where(sh.occ[:: cv.block][: cv.nb] > 0, 1, 0) if warm is None else warm
        idx, _, _ = cv.improve(base)
        E = float(cv.Pblk[idx].sum())
        if E <= eb:
            hi = np.log(nu)
            ce = cv.forward(idx)[1] - nu * E
            if best is None or ce < best[1]:
                best = (idx.copy(), ce, nu)
        else:
            lo = np.log(nu)
        warm = idx
    idx, ce, nu = best
    m = run_shift(sh, Playback(np.repeat(np.asarray(duties)[idx], 6)), pl, sensor_seed=seed + 7919)
    return dict(m, seed=seed, scale=s, policy="Clairvoyant/budget", nu=nu)


def ends_job(a):
    """Fan-off and full-duty exposure of a day at purifier scale s (for the captured share)."""
    s, seed = a
    pl = replace(Plant(), q_scale=s)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    rows = []
    for name, u in [("Off", 0.0), ("AO-100", 1.0)]:
        m = run_shift(sh, Playback(np.full(len(sh.t), u), name), pl, sensor_seed=seed + 7919)
        rows.append(dict(m, seed=seed, scale=s, policy=name))
    return rows


THC_GRID = [{"th1": a, "dth": b, "hyst": h, "gate": True, "avg": 6} for a in [15, 20, 25, 30, 35]
            for b in [20, 30, 40, 60, 80] for h in [0.0, 2.5, 5.0]]          # the grid of the main tuning (e12)


def thc_tune_job(a):
    """Threshold--hysteresis controller on one tuning day of the room with infiltration a0 (grid point k)."""
    from aqmpc.extra_controllers import HysteresisThreshold
    a0, seed, rho, k = a
    pl = replace(Plant(), lam0=a0)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    m = run_shift(sh, HysteresisThreshold(**THC_GRID[k]), pl, sensor_seed=seed + 7919)
    return dict(a0=a0, seed=seed, k=k, J=m["CE"] + rho * m["energy_Wh"])


def a0_job(a):
    """One day of the room with infiltration rate a0: the controllers designed for that room (THC re-tuned on the
    tuning days), the clairvoyant schedule and the certificate of Theorem 1(a)."""
    from aqmpc.extra_controllers import HysteresisThreshold
    a0, seed, rho, thc = a
    pl = replace(Plant(), lam0=a0)
    st = MPCSettings(rho=rho)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    ss = seed + 7919
    rows, traces = [], {}
    ctrls = [("Relay", Relay()), ("THC", HysteresisThreshold(**thc)), ("CL-r", RolloutCostate(pl, st)),
             ("CL-r-exact", RolloutCostate(pl, st, info="oracle", name="CL-r-exact")),
             ("EMPC-FB-L", MPC(pl, None, "persistence", MPCSettings(rho=rho, **FIRST1), name="EMPC-FB-L"))]
    for name, c in ctrls:
        m, tr = run_shift(sh, c, pl, sensor_seed=ss, return_traces=True)
        traces[name] = tr["U"]
        rows.append(dict(m, policy=name))
    for name, u in [("Off", 0.0), ("AO-100", 1.0)]:
        rows.append(dict(run_shift(sh, Playback(np.full(len(sh.t), u), name), pl, sensor_seed=ss), policy=name))
    cv = Clairvoyant(pl, sh, st.duties, rho)
    (idx, Jc, src), rec = clairvoyant_best(cv, sh, {"CL-r-exact": traces["CL-r-exact"]}, np.random.default_rng(seed),
                                           n_random=4)
    m = run_shift(sh, Playback(cv.schedule(idx)), pl, sensor_seed=ss)
    box = cv.reach_box()
    LB, _, _ = cv.lower_bound(idx, box)
    LBr, Jr, _ = cv.lower_bound(cv.to_idx(traces["CL-r-exact"]), box)
    rows.append(dict(m, policy="Clairvoyant", clair_internal=Jc, LB=max(LB, LBr), beta_rex=Jr - LBr,
                     Jrex_model=Jr))
    for r in rows:
        r.update(seed=seed, a0=a0, rho=rho, eps=leverage(pl), J=r["CE"] + rho * r["energy_Wh"])
    return rows


def thc_params_job(a):
    """THC with explicit parameters on one day of the room with infiltration a0 (grid refinement and evaluation)."""
    from aqmpc.extra_controllers import HysteresisThreshold
    a0, seed, rho, prm = a
    pl = replace(Plant(), lam0=a0)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    m = run_shift(sh, HysteresisThreshold(**prm), pl, sensor_seed=seed + 7919)
    return dict(m, policy="THC", seed=seed, a0=a0, rho=rho, eps=leverage(pl), J=m["CE"] + rho * m["energy_Wh"],
                key=json.dumps(prm, sort_keys=True))


def refine_grid(best):
    """Grid around an optimum on the edge of the THC grid (thresholds in steps of 5, at least 2.5 ug/m3)."""
    th = sorted({max(2.5, best["th1"] + d) for d in (-10, -5, 0, 5)})
    dt = sorted({max(5.0, best["dth"] + d) for d in (-10, -5, 0, 5)})
    return [{"th1": a, "dth": b, "hyst": h, "gate": True, "avg": 6} for a in th for b in dt for h in (0.0, 2.5, 5.0)]


def rcl_price_job(a):
    """CL-r with exact information, told one number about the day: the smallest price at which it meets the budget
    on that day (found by bisection on log price, like the clairvoyant schedule; no budget filter is needed). It
    separates the value of knowing the day's shadow price from the value of the rest of the whole-day preview."""
    s, seed, eb = a
    pl = replace(Plant(), q_scale=s)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    lo, hi = np.log(1e-4), np.log(10.0)
    best = None
    for _ in range(20):
        nu = float(np.exp(0.5 * (lo + hi)))
        m = run_shift(sh, RolloutCostate(pl, MPCSettings(rho=nu), info="oracle", name="CL-r-exact/price*"), pl,
                      sensor_seed=seed + 7919)
        if m["energy_Wh"] <= eb:
            hi = np.log(nu)
            if best is None or m["CE"] < best[0]["CE"]:
                best = (m, nu)
        else:
            lo = np.log(nu)
    m, nu = best
    return dict(m, seed=seed, scale=s, policy="CL-r-exact/price*", price=nu)


def norm(df):
    """Captured share and energy share per day, as in the main comparison."""
    off = df[df.policy == "Off"].set_index(["scale", "seed"]).CE
    full = df[df.policy == "AO-100"].set_index(["scale", "seed"]).CE
    k = list(zip(df.scale, df.seed))
    df["captured"] = [(off[x] - c) / (off[x] - full[x]) for x, c in zip(k, df.CE)]
    return df


if __name__ == "__main__":
    os.makedirs(out("e15"), exist_ok=True)
    T = load_tuning()
    rho_iso = T["rho_iso"]
    t3 = json.load(open(out("e12/tuning3.json")))
    for stage in sys.argv[1:]:
        log(f"stage {stage}")
        if stage == "sweep":
            jobs = [(s, sd, rho_iso, t3["sched"]) for s in SCALES for sd in SMALL_SEEDS]
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                rows = [r for res in ex.map(sweep_job, jobs, chunksize=1) for r in res]
            df = norm(pd.DataFrame(rows))
            df["eps"] = df.scale.map({s: leverage(replace(Plant(), q_scale=s)) for s in SCALES})
            df.to_csv(out("e15/leverage.csv"), index=False)
            log("sweep: " + str(df.groupby(["scale", "policy"]).J.mean().unstack().round(4).to_dict()))
        elif stage == "main":
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                rows = list(ex.map(main_job, [(sd, rho_iso) for sd in TEST_SEEDS], chunksize=1))
            pd.DataFrame(rows).to_csv(out("e15/clair_main.csv"), index=False)
            log(f"main: mean J {pd.DataFrame(rows).J.mean():.4f}")
        elif stage == "lambda":
            jobs = [(s, sd, rho_iso, True) for s in SCALES for sd in SMALL_SEEDS] + \
                   [(1.0, sd, rho_iso, False) for sd in TEST_SEEDS if sd not in SMALL_SEEDS]
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                rows = [r for res in ex.map(lambda_job, jobs, chunksize=1) for r in res]
            df = pd.DataFrame(rows)
            df.to_csv(out("e15/lambda.csv"), index=False)
            log("lambda: " + str(df.groupby(["scale", "policy"]).J.mean().unstack().round(4).to_dict()))
        elif stage == "lambda1":
            jobs = [(s, sd, rho_iso, True) for s in SCALES for sd in SMALL_SEEDS] +                    [(1.0, sd, rho_iso, False) for sd in TEST_SEEDS if sd not in SMALL_SEEDS]
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                rows = [r for res in ex.map(lambda1_job, jobs, chunksize=1) for r in res]
            df = pd.DataFrame(rows)
            df.to_csv(out("e15/lambda1.csv"), index=False)
            log("lambda1: " + str(df.groupby(["scale", "policy"]).J.mean().unstack().round(4).to_dict()))
        elif stage == "cert":
            jobs = [(1.0, sd, rho_iso) for sd in TEST_SEEDS] +                    [(s, sd, rho_iso) for s in SCALES if s != 1.0 for sd in SMALL_SEEDS]
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                rows = list(ex.map(cert_job, jobs, chunksize=1))
            df = pd.DataFrame(rows)
            df.to_csv(out("e15/cert.csv"), index=False)
            log("cert: " + str(df.assign(g=100 * (df.Jc - df.LB) / df.Jc).groupby("scale").g.agg(["mean", "max"])))
        elif stage == "certlp":
            with ProcessPoolExecutor(max_workers=LP_WORKERS) as ex:
                rows = list(ex.map(certlp_job, [(sd, rho_iso) for sd in TEST_SEEDS[:30]], chunksize=1))
            pd.DataFrame(rows).to_csv(out("e15/cert_lp.csv"), index=False)
            log(f"certlp: {len(rows)} days")
        elif stage == "milpcheck":
            with ProcessPoolExecutor(max_workers=max(2, LP_WORKERS)) as ex:
                rows = list(ex.map(milpcheck_job, [(sd, rho_iso) for sd in TEST_SEEDS[:20]], chunksize=1))
            df = pd.DataFrame(rows)
            df.to_csv(out("e15/milpcheck.csv"), index=False)
            log("milpcheck: max rel diff " + str(float(((df.Jc - df.Jmilp) / df.Jmilp).abs().max())))
        elif stage == "budgetsweep":
            from common import TUNE_SEEDS
            eb = 0.25 * E_FULL
            rB = T["rho_B"]["0.25"]
            rows = []
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                for s in BUDGET_SCALES:
                    jobs = [(s, sd, s * rB * f, eb, f"RCLex/fixed|{f}") for f in BUDGET_FACS for sd in TUNE_SEEDS]
                    tune = pd.DataFrame(list(ex.map(rcl_budget_job, jobs, chunksize=1)))
                    f_best = tune.groupby("policy").CE.mean().idxmin()
                    price = float(tune[tune.policy == f_best].price.iloc[0])
                    log(f"budgetsweep s={s}: tuned {f_best}")
                    rows += list(ex.map(rcl_budget_job, [(s, sd, price, eb, "CL-r-exact/fixed*") for sd in SMALL_SEEDS],
                                        chunksize=1))
                    clair = list(ex.map(budget_scaled_job, [(sd, eb, s) for sd in SMALL_SEEDS], chunksize=1))
                    rows += clair
                    rows += list(ex.map(rcl_price_job, [(s, sd, eb) for sd in SMALL_SEEDS], chunksize=1))
                    rows += [r for res in ex.map(ends_job, [(s, sd) for sd in SMALL_SEEDS], chunksize=1) for r in res]
            df = norm(pd.DataFrame(rows))
            df.to_csv(out("e15/budget_sweep.csv"), index=False)
            log("budgetsweep: " + str(df.groupby(["scale", "policy"]).captured.mean().unstack().round(4).to_dict()))
        elif stage == "a0sweep":                  # infiltration rate a0: re-tune THC per room, then evaluate
            from common import TUNE_SEEDS
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                tune = pd.DataFrame(list(ex.map(thc_tune_job, [(a0, sd, rho_iso, k) for a0 in A0S for sd in TUNE_SEEDS
                                                               for k in range(len(THC_GRID))], chunksize=4)))
                best = tune.groupby(["a0", "k"]).J.mean().groupby("a0").idxmin()
                thc = {a0: THC_GRID[best[a0][1]] for a0 in A0S}
                log("a0sweep THC: " + str(thc))
                json.dump({str(k): v for k, v in thc.items()}, open(out("e15/a0_thc.json"), "w"), indent=1)
                rows = [r for res in ex.map(a0_job, [(a0, sd, rho_iso, thc[a0]) for a0 in A0S for sd in SMALL_SEEDS],
                                            chunksize=1) for r in res]
            df = pd.DataFrame(rows)
            df.to_csv(out("e15/a0_sweep.csv"), index=False)
            log("a0sweep: " + str(df.groupby(["a0", "policy"]).J.mean().unstack().round(4).to_dict()))
        elif stage == "a0refine":                 # refine the THC grid where the a0 optimum lies on its edge
            from common import TUNE_SEEDS
            thc = {float(k): v for k, v in json.load(open(out("e15/a0_thc.json"))).items()}
            th1s = sorted({g["th1"] for g in THC_GRID}); dths = sorted({g["dth"] for g in THC_GRID})
            df = pd.read_csv(out("e15/a0_sweep.csv"))
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                for a0 in A0S:
                    best = thc[a0]
                    for _ in range(4):
                        if th1s[0] < best["th1"] < th1s[-1] and dths[0] < best["dth"] < dths[-1]:
                            break
                        grid = refine_grid(best)
                        res = pd.DataFrame(list(ex.map(thc_params_job, [(a0, sd, rho_iso, g) for g in grid
                                                                         for sd in TUNE_SEEDS], chunksize=4)))
                        new = json.loads(res.groupby("key").J.mean().idxmin())
                        th1s = sorted(set(th1s) | {g["th1"] for g in grid}); dths = sorted(set(dths) | {g["dth"] for g in grid})
                        if new == best:
                            break
                        best = new
                    if best != thc[a0]:
                        log(f"a0refine a0={a0}: {thc[a0]} -> {best}")
                        thc[a0] = best
                        rows = list(ex.map(thc_params_job, [(a0, sd, rho_iso, best) for sd in SMALL_SEEDS], chunksize=1))
                        new_rows = pd.DataFrame(rows).drop(columns=["key"])
                        df = pd.concat([df[~((df.a0 == a0) & (df.policy == "THC"))], new_rows], ignore_index=True)
                    th1s = sorted({g["th1"] for g in THC_GRID}); dths = sorted({g["dth"] for g in THC_GRID})
            json.dump({str(k): v for k, v in thc.items()}, open(out("e15/a0_thc.json"), "w"), indent=1)
            df.to_csv(out("e15/a0_sweep.csv"), index=False)
            log("a0refine: " + str(thc))
        elif stage == "certlp_all":               # LP certificate (interior point) on all test days
            with ProcessPoolExecutor(max_workers=LP_WORKERS) as ex:
                rows = list(ex.map(certlp_long_job, [(sd, rho_iso) for sd in TEST_SEEDS], chunksize=1))
            df = pd.DataFrame(rows)
            old = pd.read_csv(out("e15/cert_lp.csv"))
            chk = df.merge(old, on=["scale", "seed"], suffixes=("", "_old")).dropna()
            log(f"certlp_all: {int(df.LB_lp.notna().sum())} of {len(df)} days; max rel diff to earlier LP on "
                f"{len(chk)} days: {float(((chk.LB_lp - chk.LB_lp_old) / chk.LB_lp_old).abs().max()):.2e}")
            df.to_csv(out("e15/cert_lp_all.csv"), index=False)
        elif stage == "certlp_retry":             # LP certificate on the days whose LP hit the 30-min limit
            C = pd.read_csv(out("e15/cert_lp.csv"))
            miss = C[C.LB_lp.isna()].seed.tolist()
            log(f"certlp_retry: {miss}")
            with ProcessPoolExecutor(max_workers=LP_WORKERS) as ex:
                got = list(ex.map(certlp_long_job, [(sd, rho_iso) for sd in miss], chunksize=1))
            for r in got:
                C.loc[C.seed == r["seed"], "LB_lp"] = r["LB_lp"]
            C.to_csv(out("e15/cert_lp.csv"), index=False)
            log(f"certlp_retry: {int(C.LB_lp.notna().sum())} of {len(C)} days now solved")
        elif stage == "budget":
            eb = 0.25 * E_FULL
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                rows = list(ex.map(budget_job, [(sd, eb) for sd in TEST_SEEDS], chunksize=1))
            pd.DataFrame(rows).to_csv(out("e15/clair_budget.csv"), index=False)
            log(f"budget: mean energy {pd.DataFrame(rows).energy_Wh.mean():.3f} Wh of {eb:.3f}")
