"""E6c: solution methods for one MPC step on random instances (same generator as e6_exactness).

exact      exhaustive enumeration over U^Nu (identical optimum to branch and bound, see e6_exactness)
RR         continuous relaxation u in [0,1]^Nu solved by a local NLP (SLSQP; budget as a nonlinear
           inequality when it binds), rounded to the nearest admissible duty; if the rounded sequence
           violates the energy allowance, it is repaired by lowering the highest duty block until feasible
const      best constant duty over the whole horizon
Gap = (J_method - J_exact) / |J_exact| evaluated with the exact discrete model.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from aqmpc.params import Plant, MPCSettings, NP, NZ, C_BOUND
from aqmpc.plant import Discretised
from aqmpc.solver import MPCProblem
from aqmpc.exposure import weight_tensor
from aqmpc.extra_controllers import RelaxedProblem
from common import out


def relax_round(prob, rel, c0, D, occ, e_allow, Wf, p_fan, p_exp, duties):
    nu_b = prob.Nu
    L = np.array(prob.blens, dtype=float)
    f = lambda v: rel.cost(v, c0, D, occ, Wf, p_fan, p_exp)
    cons = []
    if e_allow < 1e8:
        cons = [{"type": "ineq", "fun": lambda v: e_allow - float((L * p_fan * np.clip(v, 0, 1) ** p_exp * prob.dt).sum())}]
    best = None
    for x0 in (np.full(nu_b, 0.8), np.full(nu_b, 0.3)):
        r = minimize(f, x0, method="SLSQP", bounds=[(0.0, 1.0)] * nu_b, constraints=cons,
                     options={"maxiter": 100, "ftol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    v = np.clip(best.x, 0, 1)
    seq = np.array([int(np.argmin(np.abs(duties - vb))) for vb in v])
    J, E = prob.evaluate(seq, c0, D, occ, Wf)
    infeasible = E > e_allow + 1e-12
    repaired = seq.copy()
    while True:
        J2, E2 = prob.evaluate(repaired, c0, D, occ, Wf)
        if E2 <= e_allow + 1e-12 or repaired.max() == 0:
            break
        repaired[int(np.argmax(repaired))] -= 1
    J2, E2 = prob.evaluate(repaired, c0, D, occ, Wf)
    return repaired, J2, bool(infeasible), int(best.nfev)


if __name__ == "__main__":
    n_inst = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(2027)
    pl = Plant()
    st = MPCSettings()
    disc = Discretised(pl, st.dt, st.duties)
    W = weight_tensor()
    Mref = disc.M0 + 0.8 * disc.M1
    Lam = np.stack([np.linalg.solve(-Mref[p].T, W[p]) for p in range(NP)])
    base_c = np.array([40, 80, 300, 600, 1.0, 30.0])[:, None] * np.ones((1, NZ))
    base_d = -np.einsum("pij,pj->pi", Mref, base_c)
    rel = None
    recs = []
    for k in range(n_inst):
        rho = float(np.exp(rng.uniform(np.log(1e-3), np.log(0.5))))
        M = float(rng.choice([0.0, 10.0, 100.0]))
        prob = MPCProblem(disc, W, C_BOUND, M, rho, pl.fan.p_fan, st.N, st.Nu)
        rel = RelaxedProblem(prob, disc) if rel is None else rel
        rel.p = prob
        c0 = base_c * np.exp(rng.normal(0, 0.6, (NP, NZ)))
        D = np.broadcast_to(base_d, (st.N, NP, NZ)).copy() * np.exp(rng.normal(0, 0.3, (1, NP, NZ)))
        a = rng.integers(0, st.N); b = min(st.N, a + rng.integers(5, 25))
        D[a:b] *= np.exp(rng.uniform(0, 2.5))
        occ = np.ones(st.N); occ[: rng.integers(0, 10)] = 0.0
        e_allow = float(rng.choice([1e9, rng.uniform(0.1, 1.0) * pl.fan.p_fan * st.N * st.dt]))
        Wf = Lam if rng.random() < 0.5 else None
        t0 = time.perf_counter(); s_ex, J_ex, _ = prob.solve_exhaustive(c0, D, occ, e_allow, Wf); t_ex = time.perf_counter() - t0
        t0 = time.perf_counter()
        s_rr, J_rr, infeas, nfev = relax_round(prob, rel, c0, D, occ, e_allow, Wf, pl.fan.p_fan, pl.fan.p_exp, st.duties)
        t_rr = time.perf_counter() - t0
        Jc = [prob.evaluate(np.full(st.Nu, iu), c0, D, occ, Wf) for iu in range(len(st.duties))]
        Jc = [j for j, e in Jc if e <= e_allow + 1e-12]
        J_c = min(Jc)
        den = max(1e-9, abs(J_ex))
        recs.append({"binding": e_allow < 1e8, "rho": rho, "M": M, "terminal": Wf is not None,
                     "J_ex": J_ex, "J_rr": J_rr, "J_const": J_c,
                     "gap_rr": (J_rr - J_ex) / den, "gap_const": (J_c - J_ex) / den,
                     "first_rr": int(s_rr[0] == s_ex[0]), "rr_round_infeasible": int(infeas),
                     "nfev_rr": nfev, "t_ex": t_ex, "t_rr": t_rr})
    df = pd.DataFrame(recs)
    df.to_csv(out("e6/methods.csv"), index=False)
    summ = {}
    for m in ["rr", "const"]:
        g = df[f"gap_{m}"]
        summ[m] = {"median_gap": float(g.median()), "p95_gap": float(g.quantile(0.95)), "max_gap": float(g.max()),
                   "share_gap_gt_1pct": float((g > 0.01).mean()), "share_suboptimal": float((g > 1e-9).mean())}
    summ["rr"]["first_move_agreement"] = float(df.first_rr.mean())
    summ["rr"]["round_infeasible_share_binding"] = float(df[df.binding].rr_round_infeasible.mean())
    summ["n"] = n_inst
    summ["n_binding"] = int(df.binding.sum())
    json.dump(summ, open(out("e6/methods_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))
