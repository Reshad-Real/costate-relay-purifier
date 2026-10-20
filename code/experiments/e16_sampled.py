"""E16: direct test of Theorem 1 on its own sampled model.

The plant is the 1-min sampled model of the paper (duty and drive held over each minute; the drive is the minute mean
of the scenario's 10-s drive), and the cost is the theorem's J = sum_k (o_{k+1} dt w'c_{k+1} + rho P(u_k) dt). Plant,
controller model and cost coincide, so the gaps below come from the decisions alone (no model-resolution term).
Controllers (all mu = 0, no budget, exact state):
  RL        rollout costate law, exact current-interval drive
  EL-exact  EMPC-Lambda (terminal Lambda_{k+N}, blocks 1,2,3,6,8,10 min), exact drive over the horizon
  EL-pers   EMPC-Lambda with exact current-interval drive and persistence of it over the rest of the horizon
  EM-stat   EMPC with the stationary terminal weight and uniform 5-min blocks (first-order contrast)
Clairvoyant optimum on the same model: exact 1-opt local search from several starts; certified lower bound LB on J*
(Theorem 1(a)) as the best certificate over all schedules evaluated.                 -> results/e16/sampled.csv"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from aqmpc.params import Plant, MPCSettings, NP, NZ
from aqmpc.scenarios import make_shift, ScenarioConfig, SHIFT
from aqmpc.plant import Discretised, TruthSim
from aqmpc.simulate import DT_SIM
from aqmpc.solver import MPCProblem
from aqmpc.exposure import weight_tensor
from common import out, load_tuning, SMALL_SEEDS
from e15_leverage import leverage

SCALES = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 2.0, 4.0]
WORKERS = int(os.environ.get("AQ_WORKERS", max(1, (os.cpu_count() or 2) - 2)))
DT = 1.0 / 60
BLOCKS1 = (1, 2, 3, 6, 8, 10)


class SampledDay:
    def __init__(self, plant, shift, rho, duties):
        self.disc = Discretised(plant, DT, duties)
        self.A, self.G = self.disc.A, self.disc.G                      # (U, P, Z, Z)
        self.duties = np.asarray(duties, float)
        self.nu = len(self.duties)
        sim = TruthSim(plant, DT_SIM)
        d10 = plant.lam0 * shift.c_out[:, :, None] + shift.S / sim.vol[None, None, :]
        m = int(round(DT / DT_SIM))
        self.K = len(shift.t) // m
        self.d = d10[: self.K * m].reshape(self.K, m, NP, NZ).mean(axis=1)
        self.c0 = self.disc.steady_state(0, self.d[0])
        tk = DT * np.arange(self.K + 1)
        self.o = ((tk >= SHIFT[0]) & (tk < SHIFT[1])).astype(float)
        self.W = weight_tensor()
        self.P = plant.fan.power(self.duties)
        self.rho = rho
        self.ow = self.o[:, None, None] * DT * self.W[None]                # (K+1, P, Z): o_k dt w

    def forward(self, idx):
        C = np.empty((self.K + 1, NP, NZ)); C[0] = self.c0
        J = 0.0
        for k in range(self.K):
            j = idx[k]
            C[k + 1] = np.einsum("pij,pj->pi", self.A[j], C[k]) + np.einsum("pij,pj->pi", self.G[j], self.d[k])
            J += float((self.ow[k + 1] * C[k + 1]).sum()) + self.rho * self.P[j] * DT
        return C, J

    def costate(self, idx, A_of=None):
        lam = np.zeros((self.K + 1, NP, NZ))
        for k in range(self.K - 1, -1, -1):
            A = self.A[idx[k]] if A_of is None else A_of
            lam[k] = np.einsum("pji,pj->pi", A, self.ow[k + 1] + lam[k + 1])
        return lam

    def scores(self, k, c, mu_next, d):
        """rho P(j) dt + sum_p mu_next' (A_j c + G_j d) for every duty j."""
        nxt = np.einsum("upij,pj->upi", self.A, c) + np.einsum("upij,pj->upi", self.G, d)
        return self.rho * self.P * DT + np.einsum("upi,pi->u", nxt, mu_next)

    def deltas(self, idx, C, lam):
        D = np.empty((self.K, self.nu))
        for k in range(self.K):
            s = self.scores(k, C[k], self.ow[k + 1] + lam[k + 1], self.d[k])
            D[k] = s - s[idx[k]]
        return D

    def improve(self, idx, max_iter=1000, tol=1e-12):
        idx = np.asarray(idx).copy()
        C, J = self.forward(idx)
        for _ in range(max_iter):
            lam = self.costate(idx)
            D = self.deltas(idx, C, lam)
            bj = D.argmin(axis=1); bd = D[np.arange(self.K), bj]
            cand = np.flatnonzero(bd < -tol)
            if cand.size == 0:
                break
            order = cand[np.argsort(bd[cand])]
            n = order.size
            while True:
                new = idx.copy(); new[order[:n]] = bj[order[:n]]
                C2, J2 = self.forward(new)
                if J2 < J - tol or n == 1:
                    break
                n = max(1, n // 2)
            if J2 >= J - tol:
                break
            idx, C, J = new, C2, J2
        return idx, J

    def box(self):
        lo = np.empty((self.K + 1, NP, NZ)); hi = np.empty_like(lo)
        lo[0] = hi[0] = self.c0
        for k in range(self.K):
            g = np.einsum("upij,pj->upi", self.G, self.d[k])
            lo[k + 1] = (np.einsum("upij,pj->upi", self.A, lo[k]) + g).min(axis=0)
            hi[k + 1] = (np.einsum("upij,pj->upi", self.A, hi[k]) + g).max(axis=0)
        return lo, hi

    def lower_bound(self, idx, bx):
        """Theorem 1(a): J* >= J(v) + sum_k min_{j, c in box_k} [H_k(j, c) - H_k(v_k, c)]."""
        lo, hi = bx
        C, J = self.forward(idx)
        lam = self.costate(idx)
        tot = 0.0
        for k in range(self.K):
            mu = self.ow[k + 1] + lam[k + 1]
            a = np.einsum("upij,pi->upj", self.A, mu)                        # gradient of H_k(j, .) in c
            g = a - a[idx[k]][None]
            D = self.scores(k, C[k], mu, self.d[k]); D = D - D[idx[k]]
            dev = np.where(g > 0, lo[k][None] - C[k][None], hi[k][None] - C[k][None])
            tot += min(0.0, float((D + np.einsum("upj,upj->u", g, dev)).min()))
        return J + tot, J


def rollout(day, ubar_idx=1):
    Lam = day.costate(None, A_of=day.A[ubar_idx])
    c = day.c0.copy(); idx = np.empty(day.K, int)
    for k in range(day.K):
        j = int(np.argmin(day.scores(k, c, day.ow[k + 1] + Lam[k + 1], day.d[k])))
        idx[k] = j
        c = np.einsum("pij,pj->pi", day.A[j], c) + np.einsum("pij,pj->pi", day.G[j], day.d[k])
    return idx, Lam


def empc(day, plant, Lam, mode, N=30):
    st = MPCSettings(rho=day.rho)
    blens = BLOCKS1 if mode != "stat" else None
    prob = MPCProblem(day.disc, day.W, st.bound, 0.0, day.rho, plant.fan.p_fan, N, 6, plant.fan.p_exp, blens=blens)
    Mref = day.disc.M0 + 0.8 * day.disc.M1
    lam_bar = np.stack([np.linalg.solve(-Mref[p].T, day.W[p]) for p in range(NP)])
    c = day.c0.copy(); idx = np.empty(day.K, int)
    for k in range(day.K):
        D = np.array([day.d[min(k + i, day.K - 1)] for i in range(N)])
        if mode == "pers":
            D[1:] = day.d[k]                                                   # persistence of the current drive
        occ = np.array([day.o[min(k + i, day.K)] if k + i <= day.K else 0.0 for i in range(1, N + 1)])
        if mode == "stat":
            Wf = lam_bar * float(day.o[min(k + N, day.K)] > 0)
        else:
            Wf = Lam[min(k + N, day.K)]
        seq, _, _ = prob.solve_exhaustive(c, D, occ, 1e12, Wf)
        j = int(seq[0]); idx[k] = j
        c = np.einsum("pij,pj->pi", day.A[j], c) + np.einsum("pij,pj->pi", day.G[j], day.d[k])
    return idx


def job(a):
    s, seed, rho_iso = a
    pl = replace(Plant(), q_scale=s)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    day = SampledDay(pl, sh, s * rho_iso, MPCSettings().duties)
    rl, Lam = rollout(day)
    el = empc(day, pl, Lam, "exact")
    ep = empc(day, pl, Lam, "pers")
    es = empc(day, pl, Lam, "stat")
    rng = np.random.default_rng(seed)
    base = np.where(day.o[1:] > 0, 1, 0)
    best, finals = None, []
    for st in [base, rl, el, rng.integers(0, day.nu, day.K), rng.integers(0, day.nu, day.K)]:
        i2, J2 = day.improve(st)
        finals.append(i2)
        if best is None or J2 < best[1]:
            best = (i2, J2)
    bx = day.box()
    LBs = {}
    for name, sch in [("clair", best[0]), ("RL", rl), ("EL-exact", el)]:
        LBs[name] = day.lower_bound(sch, bx)[0]
    row = dict(scale=s, seed=seed, Jc=best[1], LB=max(LBs.values()), LB_clair=LBs["clair"],
               beta_RL=day.forward(rl)[1] - LBs["RL"], beta_EL=day.forward(el)[1] - LBs["EL-exact"],
               Joff=day.forward(np.zeros(day.K, int))[1])
    for name, sch in [("RL", rl), ("EL-exact", el), ("EL-pers", ep), ("EM-stat", es)]:
        row["J_" + name] = day.forward(sch)[1]
    return row


if __name__ == "__main__":
    os.makedirs(out("e16"), exist_ok=True)
    T = load_tuning()
    jobs = [(s, sd, T["rho_iso"]) for s in SCALES for sd in SMALL_SEEDS]
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        jobs = [(s, sd, T["rho_iso"]) for s in [0.25, 1.0] for sd in SMALL_SEEDS[:2]]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(job, jobs, chunksize=1))
    df = pd.DataFrame(rows)
    df["eps"] = df.scale.map({s: leverage(replace(Plant(), q_scale=s)) for s in SCALES})
    df.to_csv(out("e16/sampled.csv" if len(jobs) > 10 else "e16/sampled_quick.csv"), index=False)
    for c in ["RL", "EL-exact", "EL-pers", "EM-stat"]:
        df["rel_" + c] = 100 * (df["J_" + c] - df.Jc) / (df.Joff - df.Jc)
    df["cert_rel"] = 100 * (df.Jc - df.LB) / (df.Joff - df.LB)
    print(df.groupby("scale")[[c for c in df.columns if c.startswith("rel_")] + ["cert_rel"]].mean().round(5))
    print(f"{time.time() - t0:.0f} s", flush=True)
