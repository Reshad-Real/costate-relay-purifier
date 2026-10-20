"""E13: numerical checks of the two structural results added for the ISA version.

rollout     Theorem (rollout improvement): in the discrete-time model (exact ZOH, interval-averaged true
            drive, exact state, M = 0, no budget) the rollout costate law never costs more than its base
            policy u = u_base, on every shift. Checked on the 200 test shifts for u_base in {0.8, 1.0}.
invariance  Proposition (forecast invariance): with costate bounds from the comparison matrices
            M_lo = M0 + min(0, M1), M_hi = M0 + max(0, M1), the switching function at state c lies in
            [B_min(c,t), B_max(c,t)] for every future duty signal and every drive; when both ends select the
            same duty, the optimal duty at t does not depend on any forecast. Coverage is evaluated along
            closed-loop MPC-FF trajectories on 30 test shifts (interval-averaged readings as state).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.linalg import expm
from aqmpc.params import Plant, MPCSettings, NP, NZ, DUTY_LEVELS
from aqmpc.plant import Discretised
from aqmpc.exposure import weight_tensor
from aqmpc.scenarios import make_shift, ScenarioConfig, SHIFT, T_END, DT_SIM
from aqmpc.simulate import run_shift
from aqmpc.harness import build_controller, train_forecaster
from common import out, load_tuning, TEST_SEEDS


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def interval_drive(pl, shift, dt):
    drive = pl.lam0 * shift.c_out[:, :, None] + shift.S / pl.vol[None, None, :]
    m = int(round(dt / DT_SIM))
    K = len(shift.t) // m
    return drive[: K * m].reshape(K, m, NP, NZ).mean(axis=1), drive[0]


def rollout_check(u_base, rho, seeds):
    pl = Plant()
    st = MPCSettings(rho=rho)
    disc = Discretised(pl, st.dt, st.duties)
    W = weight_tensor()
    dt = st.dt
    K = int(round(T_END / dt))
    tk = dt * np.arange(K + 1)
    o = ((tk >= SHIFT[0]) & (tk < SHIFT[1])).astype(float)
    Ab, Gb = disc.mats(u_base)
    Lam = np.zeros((K + 1, NP, NZ))
    for j in range(K - 1, -1, -1):
        Lam[j] = np.einsum("pji,pj->pi", Ab, o[j + 1] * dt * W + Lam[j + 1])
    pcost = pl.fan.p_fan * st.duties ** pl.fan.p_exp
    pb = pl.fan.p_fan * u_base ** pl.fan.p_exp
    recs = []
    for sd in seeds:
        shift = make_shift(np.random.default_rng(sd), ScenarioConfig())
        D, d0 = interval_drive(pl, shift, dt)
        c0 = np.stack([np.linalg.solve(-disc.M0[p], d0[p]) for p in range(NP)])
        # base policy
        c = c0.copy(); Jb = 0.0
        for k in range(K):
            c = np.einsum("pij,pj->pi", Ab, c) + np.einsum("pij,pj->pi", Gb, D[k])
            Jb += o[k + 1] * dt * float((W * c).sum()) + rho * pb * dt
        # rollout law (exact state and current-interval drive)
        c = c0.copy(); Jr = 0.0
        for k in range(K):
            C1 = np.einsum("upij,pj->upi", disc.A, c) + np.einsum("upij,pj->upi", disc.G, D[k])
            cost = rho * pcost * dt + np.einsum("upz,pz->u", C1, o[k + 1] * dt * W + Lam[k + 1])
            iu = int(np.argmin(cost))
            c = C1[iu]
            Jr += o[k + 1] * dt * float((W * c).sum()) + rho * pcost[iu] * dt
        recs.append({"seed": sd, "u_base": u_base, "J_base": Jb, "J_rollout": Jr})
    return recs


def stage_rollout():
    T = load_tuning()
    recs = []
    for ub in [0.8, 1.0]:
        recs += rollout_check(ub, T["rho_iso"], TEST_SEEDS)
    df = pd.DataFrame(recs)
    df["rel_improvement"] = (df.J_base - df.J_rollout) / df.J_base
    df.to_csv(out("e13/rollout_check.csv"), index=False)
    s = {str(ub): {"n": int(len(g)), "n_violations": int((g.J_rollout > g.J_base * (1 + 1e-12)).sum()),
                   "min_rel_improvement": float(g.rel_improvement.min()),
                   "median_rel_improvement": float(g.rel_improvement.median())}
         for ub, g in df.groupby("u_base")}
    json.dump(s, open(out("e13/rollout_check.json"), "w"), indent=2)
    log(f"rollout check: {s}")


def _trace(a):
    seed, rho, fc = a
    pl = Plant()
    shift = make_shift(np.random.default_rng(seed), ScenarioConfig())
    ctrl = build_controller(("MPC", {"forecast": "learned", "settings": MPCSettings(rho=rho)}), pl, fc)
    m, tr = run_shift(shift, ctrl, pl, sensor_seed=seed + 7919, return_traces=True)
    return seed, tr["Y"], tr["U"], shift.t


def stage_invariance():
    T = load_tuning()
    rho = T["rho_iso"]
    pl = Plant()
    st = MPCSettings(rho=rho)
    disc = Discretised(pl, st.dt, st.duties)
    W = weight_tensor()
    M0, M1 = disc.M0, disc.M1
    Mlo, Mhi = M0 + np.minimum(M1, 0.0), M0 + np.maximum(M1, 0.0)
    dt = st.dt
    K = int(round(T_END / dt))
    tk = dt * np.arange(K + 1)
    o = ((tk >= SHIFT[0]) & (tk < SHIFT[1])).astype(float)

    def costate_bound(Mx):
        A = np.stack([expm(Mx[p] * dt) for p in range(NP)])
        Gi = np.stack([np.linalg.solve(Mx[p], A[p] - np.eye(NZ)) for p in range(NP)])   # int_0^dt exp(M s) ds
        lam = np.zeros((K + 1, NP, NZ))
        for j in range(K - 1, -1, -1):          # lambda(t_j) = A^T lambda(t_{j+1}) + Gi^T o_j w
            lam[j] = np.einsum("pji,pj->pi", A, lam[j + 1]) + np.einsum("pji,pj->pi", Gi, o[j] * W)
        return lam

    lam_lo, lam_hi = costate_bound(Mlo), costate_bound(Mhi)
    pc = pl.fan.p_fan * st.duties ** pl.fan.p_exp

    def level(B):
        return int(np.argmin(rho * pc - st.duties * B))

    fc = train_forecaster(pl)
    seeds = TEST_SEEDS[:30]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("AQ_WORKERS", 8))) as ex:
        traces = list(ex.map(_trace, [(s, rho, fc) for s in seeds]))
    recs = []
    m = int(round(dt / DT_SIM))
    for seed, Y, U, t in traces:
        for i in range(0, len(t), m):
            k = i // m
            if not (SHIFT[0] <= t[i] < SHIFT[1]) or i < m:
                continue
            c = Y[i - m + 1: i + 1].mean(axis=0)
            mvec = -np.einsum("pij,pj->pi", M1, c)                  # B = sum lam . mvec
            Bmin = float(np.minimum(lam_lo[k] * mvec, lam_hi[k] * mvec).sum())
            Bmax = float(np.maximum(lam_lo[k] * mvec, lam_hi[k] * mvec).sum())
            lmin, lmax = level(Bmin), level(Bmax)
            pen = bool((c > st.bound[:, None]).any())
            recs.append({"seed": seed, "t": t[i], "invariant": int(lmin == lmax), "penalty_active": int(pen),
                         "level_inv": lmin if lmin == lmax else -1,
                         "u_applied": float(U[i]), "agree": int(lmin == lmax and abs(st.duties[lmin] - U[i]) < 1e-9)})
    df = pd.DataFrame(recs)
    df.to_csv(out("e13/invariance.csv"), index=False)
    inv = df[df.penalty_active == 0]
    s = {"n_steps": int(len(df)), "share_penalty_active": float(df.penalty_active.mean()),
         "coverage": float(inv.invariant.mean()),
         "agreement_when_invariant": float(inv[inv.invariant == 1].agree.mean()),
         "coverage_by_level": inv[inv.invariant == 1].level_inv.value_counts(normalize=True).sort_index().to_dict()}
    json.dump(s, open(out("e13/invariance.json"), "w"), indent=2)
    log(f"invariance: {s}")


if __name__ == "__main__":
    os.makedirs(out("e13"), exist_ok=True)
    for s in sys.argv[1:]:
        {"rollout": stage_rollout, "invariance": stage_invariance}[s]()
