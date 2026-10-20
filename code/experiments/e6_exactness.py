"""E6a: exactness of branch-and-bound vs exhaustive enumeration; admissibility of the lower bound.

Random instances: states drawn around realistic levels, drive forecasts with random events,
random occupancy windows, random energy allowances (some binding), random rho and penalty.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from aqmpc.params import Plant, MPCSettings, NP, NZ, C_BOUND
from aqmpc.plant import Discretised
from aqmpc.solver import MPCProblem
from aqmpc.exposure import weight_tensor
from common import out

if __name__ == "__main__":
    n_inst = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(2026)
    recs = []
    pl = Plant()
    st = MPCSettings()
    disc = Discretised(pl, st.dt, st.duties)
    W = weight_tensor()
    Mref = disc.M0 + 0.8 * disc.M1
    Lam = np.stack([np.linalg.solve(-Mref[p].T, W[p]) for p in range(NP)])
    base_c = np.array([40, 80, 300, 600, 1.0, 30.0])[:, None] * np.ones((1, NZ))
    base_d = -np.einsum("pij,pj->pi", Mref, base_c)
    max_gap, lb_viol, n_lb = 0.0, 0, 0
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
        t0 = time.perf_counter(); s_ex, J_ex, n_leaf = prob.solve_exhaustive(c0, D, occ, e_allow, Wf); t_ex = time.perf_counter() - t0
        t0 = time.perf_counter(); s_bb, J_bb, stats = prob.solve_bnb(c0, D, occ, e_allow, Wf=Wf); t_bb = time.perf_counter() - t0
        gap = abs(J_ex - J_bb) / max(1e-9, abs(J_ex))
        max_gap = max(max_gap, gap)
        # admissibility check at random prefixes: bound <= best completion cost
        for _ in range(3):
            nb = rng.integers(1, st.Nu)
            prefix = rng.integers(0, 4, nb)
            # cost of best completion via exhaustive over the remaining blocks
            best = np.inf
            import itertools
            for tail in itertools.product(range(4), repeat=st.Nu - nb):
                seq = np.concatenate([prefix, tail]).astype(int)
                Jc, Ec = prob.evaluate(seq, c0, D, occ, Wf)
                best = min(best, Jc)
            # cost of prefix + bound
            c = c0.copy(); Jp = 0.0; kk = 0
            for bi in range(nb):
                L = prob.blens[bi]; iu = prefix[bi]
                for _ in range(L):
                    c = np.einsum("pij,pj->pi", prob.A[iu], c) + np.einsum("pij,pj->pi", prob.G[iu], D[kk])
                    Jp += prob._stage(c, occ[kk]); kk += 1
                Jp += prob.rho * L * prob.e_step[iu]
            lb = Jp + prob.lower_bound(c, kk, D, occ, Wf)
            n_lb += 1
            lb_viol += int(lb > best + 1e-9 * abs(best))
        recs.append({"rho": rho, "M": M, "binding": e_allow < 1e8, "terminal": Wf is not None,
                    "J_ex": J_ex, "J_bb": J_bb, "rel_gap": gap, "same_first_move": int(s_ex[0] == s_bb[0]),
                    "leaves_ex": int(n_leaf), "nodes_bb": stats["nodes"], "leaves_bb": stats["leaves"],
                    "pruned_bound": stats["pruned_bound"], "pruned_energy": stats["pruned_energy"],
                    "t_ex": t_ex, "t_bb": t_bb})
    import pandas as pd
    df = pd.DataFrame(recs)

    df.to_csv(out("e6/exactness.csv"), index=False)
    summ = {"instances": n_inst, "max_rel_gap": max_gap, "lb_checks": n_lb, "lb_violations": lb_viol,
            "first_move_agreement": float(df.same_first_move.mean()),
            "median_nodes_bb": float(df.nodes_bb.median()), "p95_nodes_bb": float(df.nodes_bb.quantile(0.95)),
            "full_tree_nodes": int(sum(4 ** i for i in range(st.Nu + 1))),
            "median_leaves_bb": float(df.leaves_bb.median())}
    json.dump(summ, open(out("e6/exactness_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))
