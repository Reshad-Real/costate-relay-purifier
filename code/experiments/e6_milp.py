"""E6d: the blocked discrete economic MPC step as a mixed-integer linear program solved by a generic solver
(HiGHS via scipy.optimize.milp), against exhaustive enumeration (= branch and bound, E6a).

Exact disaggregated (convex-hull) formulation: binary z[b,m] selects the duty of block b; for each step k
the state is split into mode copies x[k,m] with sum_m x[k,m] = c_k and 0 <= x[k,m] <= U_k z[b(k),m], and
    c_{k+1} = sum_m ( A_m x[k,m] + G_m d_k z[b(k),m] ),
which is exact for binary z. U_k is the forward bound of the elementwise-maximal comparison system. The soft
bound penalty uses s_k >= (c_k - cb)/cb, s_k >= 0; energy and the terminal cost are linear.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
from aqmpc.params import Plant, MPCSettings, NP, NZ, C_BOUND
from aqmpc.plant import Discretised
from aqmpc.solver import MPCProblem
from aqmpc.exposure import weight_tensor
from common import out


def solve_milp(prob, c0, D, occ, e_allow, Wf):
    nu, N, Nu = len(prob.duties), prob.N, prob.Nu
    S = NP * NZ
    blk = np.repeat(np.arange(Nu), prob.blens)
    A = prob.A.reshape(nu, NP, NZ, NZ)
    # block-diagonal (over pollutants) S x S matrices per mode
    Am = np.zeros((nu, S, S)); Gm = np.zeros((nu, S, S))
    for m in range(nu):
        for p in range(NP):
            sl = slice(p * NZ, (p + 1) * NZ)
            Am[m, sl, sl] = prob.A[m, p]; Gm[m, sl, sl] = prob.G[m, p]
    d = D.reshape(N, S)
    # upper bounds from the elementwise-maximal comparison system
    Amax, Gmax = Am.max(axis=0), Gm.max(axis=0)
    U = np.zeros((N + 1, S)); U[0] = c0.ravel()
    for k in range(N):
        U[k + 1] = Amax @ U[k] + Gmax @ d[k]
    U = U * 1.0001 + 1e-9
    # variable layout
    n_c = (N + 1) * S
    n_x = N * nu * S
    n_s = N * S
    n_z = Nu * nu
    ic = lambda k: k * S
    ix = lambda k, m: n_c + (k * nu + m) * S
    is_ = lambda k: n_c + n_x + (k - 1) * S          # k = 1..N
    iz = lambda b, m: n_c + n_x + n_s + b * nu + m
    nv = n_c + n_x + n_s + n_z
    cost = np.zeros(nv)
    Wv = prob.W.ravel(); cbv = prob.cb.ravel()
    for k in range(1, N + 1):
        cost[ic(k):ic(k) + S] += occ[k - 1] * Wv * prob.dt
        cost[is_(k):is_(k) + S] += prob.M * prob.dt
    if Wf is not None:
        cost[ic(N):ic(N) + S] += Wf.ravel()
    for b in range(Nu):
        for m in range(nu):
            cost[iz(b, m)] += prob.rho * prob.blens[b] * prob.e_step[m]
    rows, lo, hi = [], [], []
    Aeq = lil_matrix((S + N * S + N * S + Nu, nv)); r = 0
    beq = []
    # initial state
    for i in range(S):
        Aeq[r, ic(0) + i] = 1.0; beq.append(c0.ravel()[i]); r += 1
    # dynamics: c_{k+1} - sum_m (A_m x_{k,m} + G_m d_k z_{b,m}) = 0
    for k in range(N):
        b = blk[k]
        for i in range(S):
            Aeq[r, ic(k + 1) + i] = 1.0
            for m in range(nu):
                row = Am[m, i]
                for j in np.nonzero(row)[0]:
                    Aeq[r, ix(k, m) + j] = -row[j]
                Aeq[r, iz(b, m)] = -(Gm[m, i] @ d[k])
            beq.append(0.0); r += 1
    # disaggregation: c_k - sum_m x_{k,m} = 0
    for k in range(N):
        for i in range(S):
            Aeq[r, ic(k) + i] = 1.0
            for m in range(nu):
                Aeq[r, ix(k, m) + i] = -1.0
            beq.append(0.0); r += 1
    # one duty per block
    for b in range(Nu):
        for m in range(nu):
            Aeq[r, iz(b, m)] = 1.0
        beq.append(1.0); r += 1
    beq = np.array(beq)
    cons = [LinearConstraint(csr_matrix(Aeq[:r]), beq, beq)]
    # x_{k,m} - U_k z_{b,m} <= 0
    Ain = lil_matrix((N * nu * S + N * S + 1, nv)); q = 0; ub = []; lb = []
    for k in range(N):
        b = blk[k]
        for m in range(nu):
            for i in range(S):
                Ain[q, ix(k, m) + i] = 1.0; Ain[q, iz(b, m)] = -U[k, i]
                lb.append(-np.inf); ub.append(0.0); q += 1
    # penalty: (c_k - cb)/cb - s_k <= 0
    for k in range(1, N + 1):
        for i in range(S):
            Ain[q, ic(k) + i] = 1.0 / cbv[i]; Ain[q, is_(k) + i] = -1.0
            lb.append(-np.inf); ub.append(1.0); q += 1
    # energy
    for b in range(Nu):
        for m in range(nu):
            Ain[q, iz(b, m)] = prob.blens[b] * prob.e_step[m]
    lb.append(-np.inf); ub.append(min(e_allow, 1e12)); q += 1
    cons.append(LinearConstraint(csr_matrix(Ain[:q]), np.array(lb), np.array(ub)))
    integrality = np.zeros(nv); integrality[n_c + n_x + n_s:] = 1
    bounds = Bounds(np.zeros(nv), np.concatenate([np.full(n_c + n_x + n_s, np.inf), np.ones(n_z)]))
    t0 = time.perf_counter()
    res = milp(cost, constraints=cons, integrality=integrality, bounds=bounds,
               options={"mip_rel_gap": 1e-9, "presolve": True})
    t = time.perf_counter() - t0
    z = res.x[n_c + n_x + n_s:].reshape(Nu, nu)
    seq = z.argmax(axis=1)
    return seq, float(res.fun), t, res.status


if __name__ == "__main__":
    n_inst = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    rng = np.random.default_rng(2028)
    pl = Plant(); st = MPCSettings()
    disc = Discretised(pl, st.dt, st.duties)
    W = weight_tensor()
    Mref = disc.M0 + 0.8 * disc.M1
    Lam = np.stack([np.linalg.solve(-Mref[p].T, W[p]) for p in range(NP)])
    base_c = np.array([40, 80, 300, 600, 1.0, 30.0])[:, None] * np.ones((1, NZ))
    base_d = -np.einsum("pij,pj->pi", Mref, base_c)
    recs = []
    for k in range(n_inst):
        rho = float(np.exp(rng.uniform(np.log(1e-3), np.log(0.5))))
        M = float(rng.choice([0.0, 10.0, 100.0]))
        prob = MPCProblem(disc, W, C_BOUND, M, rho, pl.fan.p_fan, st.N, st.Nu)
        c0 = base_c * np.exp(rng.normal(0, 0.6, (NP, NZ)))
        D = np.broadcast_to(base_d, (st.N, NP, NZ)).copy() * np.exp(rng.normal(0, 0.3, (1, NP, NZ)))
        a = rng.integers(0, st.N); b = min(st.N, a + rng.integers(5, 25))
        D[a:b] *= np.exp(rng.uniform(0, 2.5))
        occ = np.ones(st.N); occ[: rng.integers(0, 10)] = 0.0
        e_allow = float(rng.choice([1e9, rng.uniform(0.1, 1.0) * pl.fan.p_fan * st.N * st.dt]))
        Wf = Lam if rng.random() < 0.5 else None
        t0 = time.perf_counter(); s_ex, J_ex, _ = prob.solve_exhaustive(c0, D, occ, e_allow, Wf); t_ex = time.perf_counter() - t0
        s_m, J_m, t_m, status = solve_milp(prob, c0, D, occ, e_allow, Wf)
        Jm_eval, _ = prob.evaluate(s_m, c0, D, occ, Wf)
        recs.append({"J_ex": J_ex, "J_milp": J_m, "J_milp_eval": Jm_eval, "rel_gap": (Jm_eval - J_ex) / abs(J_ex),
                     "same_first": int(s_m[0] == s_ex[0]), "t_milp": t_m, "t_ex_py": t_ex, "status": status,
                     "binding": e_allow < 1e8})
        if (k + 1) % 20 == 0:
            print(k + 1, flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(out("e6/milp.csv"), index=False)
    s = {"n": n_inst, "scipy": scipy.__version__, "max_rel_gap": float(df.rel_gap.abs().max()),
         "first_move_agreement": float(df.same_first.mean()), "median_t_milp_ms": float(1e3 * df.t_milp.median()),
         "p95_t_milp_ms": float(1e3 * df.t_milp.quantile(0.95)), "max_t_milp_ms": float(1e3 * df.t_milp.max()),
         "n_optimal": int((df.status == 0).sum())}
    json.dump(s, open(out("e6/milp_summary.json"), "w"), indent=2)
    print(json.dumps(s, indent=2))
