"""E6b: computational cost of the exact solver (C, single precision) and its scaling.

Part A: instances of the deployed model (3 zones x 6 pollutants, |U|=4, N=30, Nu=6); C results are
        checked against the double-precision Python exhaustive solver.
Part B: synthetic multizone models varying |Z|, |P|, Nu and |U|.
Also writes an ESP32 benchmark header (csolver/esp32_bench/instances.h) for measurement on target.
"""
import os, sys, struct, subprocess, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy.linalg import expm
from aqmpc.params import Plant, MPCSettings, NP, NZ, C_BOUND, PHI, PSI, G_NORM
from aqmpc.plant import Discretised
from aqmpc.solver import MPCProblem, block_lengths
from aqmpc.exposure import weight_tensor
from common import out

HERE = os.path.dirname(os.path.abspath(__file__))
CDIR = os.path.join(HERE, "..", "csolver")
EXE = os.path.join(CDIR, "bench.exe")


def synthetic(nz, np_, duties, rng):
    """Random multizone purifier model with nz zones and the first np_ pollutants."""
    pl = Plant()
    area = rng.uniform(30, 60, nz); V = area * 3.0
    F = np.zeros((nz, nz))
    for i in range(nz - 1):
        F[i, i + 1] = F[i + 1, i] = rng.uniform(150, 400)
    if nz > 2:
        i, j = rng.choice(nz, 2, replace=False); F[i, j] = F[j, i] = rng.uniform(50, 150)
    L = (F - np.diag(F.sum(1))) / V[:, None]
    w = rng.dirichlet(np.ones(nz) * 3)
    Q = pl.q_op * nz / 3
    A = np.zeros((len(duties), np_, nz, nz)); G = np.zeros_like(A)
    M0s, M1s = [], []
    for p in range(np_):
        M0 = L - np.eye(nz) * (pl.lam0 + pl.k_dep[p])
        M1 = (Q / V)[:, None] * (w[:, None] * ((1 - pl.eta[p]) * w[None, :]) - np.diag(w))
        M0s.append(M0); M1s.append(M1)
        for iu, u in enumerate(duties):
            M = M0 + u * M1
            A[iu, p] = expm(M / 60); G[iu, p] = np.linalg.solve(M, A[iu, p] - np.eye(nz))
    psi = rng.dirichlet(np.ones(nz) * 3)
    W = (PHI[:np_] / G_NORM[:np_])[:, None] * psi[None, :]
    Mref = [M0s[p] + 0.8 * M1s[p] for p in range(np_)]
    Wf = np.stack([np.linalg.solve(-Mref[p].T, W[p]) for p in range(np_)])
    base = np.array([40, 80, 300, 600, 1.0, 30.0])[:np_, None] * np.ones((1, nz))
    dbase = -np.stack([Mref[p] @ base[p] for p in range(np_)])
    return A, G, W, Wf, base, dbase, C_BOUND[:np_, None] * np.ones((1, nz))


def random_instance(A, G, W, Wf, base, dbase, cb, duties, N, Nu, rng):
    np_, nz = W.shape
    c0 = base * np.exp(rng.normal(0, 0.6, (np_, nz)))
    D = np.broadcast_to(dbase, (N, np_, nz)).copy() * np.exp(rng.normal(0, 0.3, (1, np_, nz)))
    a = rng.integers(0, N); b = min(N, a + rng.integers(5, 25)); D[a:b] *= np.exp(rng.uniform(0, 2.5))
    occ = np.ones(N); occ[: rng.integers(0, 10)] = 0.0
    dt = 1 / 60; pfan = 6.96
    e_step = pfan * np.asarray(duties) ** 3 * dt
    e_allow = float(rng.choice([1e9, rng.uniform(0.2, 1.0) * pfan * N * dt]))
    return dict(nz=nz, np=np_, nu=len(duties), N=N, Nu=Nu, blens=block_lengths(N, Nu),
                dt=dt, rho=float(np.exp(rng.uniform(np.log(2e-3), np.log(0.2)))),
                M=float(rng.choice([0.0, 100.0])), e_allow=e_allow, e_step=e_step, A=A, G=G, W=W,
                Wf=Wf if rng.random() < 0.5 else None, cb=cb, D=D, occ=occ, c0=c0)


def write_instances(path, insts):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(insts)))
        for d in insts:
            f.write(struct.pack("<5i", d["nz"], d["np"], d["nu"], d["N"], d["Nu"]))
            f.write(struct.pack(f"<{d['Nu']}i", *d["blens"]))
            f.write(struct.pack("<i", int(d["Wf"] is not None)))
            hdr = [d["dt"], d["rho"], d["M"], min(d["e_allow"], 3e38)]
            for arr in [hdr, d["e_step"], d["A"], d["G"], d["W"],
                        d["Wf"] if d["Wf"] is not None else np.zeros_like(d["W"]), d["cb"], d["D"], d["occ"], d["c0"]]:
                a = np.asarray(arr, dtype="<f4").ravel()
                f.write(a.tobytes())


def run_c(path, reps=5):
    res = subprocess.run([EXE, path, str(reps)], capture_output=True, text=True, check=True)
    from io import StringIO
    return pd.read_csv(StringIO(res.stdout))


def py_exhaustive(d):
    class _D:  # minimal stand-in for Discretised
        pass
    disc = _D(); disc.A = d["A"]; disc.G = d["G"]; disc.duties = np.zeros(d["nu"]); disc.dt = d["dt"]
    prob = MPCProblem.__new__(MPCProblem)
    prob.A, prob.G, prob.duties, prob.dt = d["A"], d["G"], disc.duties, d["dt"]
    prob.W, prob.cb, prob.M, prob.rho = d["W"], d["cb"], d["M"], d["rho"]
    prob.N, prob.Nu, prob.blens = d["N"], d["Nu"], d["blens"]
    prob.e_step = np.asarray(d["e_step"])
    seq, J, _ = MPCProblem.solve_exhaustive(prob, d["c0"], d["D"], d["occ"], d["e_allow"], d["Wf"])
    return seq, J


def compile_c():
    src = [os.path.join(CDIR, "mpc_solver.c"), os.path.join(CDIR, "bench.c")]
    subprocess.run(["gcc", "-O2", "-std=c99", "-o", EXE] + src, check=True)


def cf32(v):
    s = f"{float(v):.8g}"
    return (s if any(ch in s for ch in ".en") else s + ".0") + "f"


def esp32_header(insts, path):
    """Emit the first instances as C arrays for the on-target benchmark sketch."""
    lines = ["/* generated by experiments/e6_timing.py: deployed-model instances for ESP32 timing */",
             "#pragma once", f"#define N_INST {len(insts)}"]
    for k, d in enumerate(insts):
        def arr(name, a):
            a = np.asarray(a, dtype=np.float32).ravel()
            # prefix I_: the ESP32 Arduino core defines analog-pin constants A0..A19
            return f"static const float I_{name}{k}[{a.size}] = {{" + ",".join(cf32(v) for v in a) + "};"
        for nm in ["A", "G", "W", "cb", "D", "occ", "c0"]:
            lines.append(arr(nm, d[nm]))
        lines.append(arr("Wf", d["Wf"] if d["Wf"] is not None else np.zeros_like(d["W"])))
        lines.append(arr("es", d["e_step"]))
        lines.append(f"static const float I_hdr{k}[4] = {{{cf32(d['dt'])},{cf32(d['rho'])},{cf32(d['M'])},{cf32(min(d['e_allow'], 3e38))}}};")
        lines.append(f"static const int I_has_wf{k} = {int(d['Wf'] is not None)};")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    compile_c()
    rng = np.random.default_rng(42)
    # ---- Part A: deployed model -------------------------------------------------------------
    pl = Plant(); st = MPCSettings(); duties = st.duties
    disc = Discretised(pl, st.dt, duties)
    W = weight_tensor()
    Mref = disc.M0 + 0.8 * disc.M1
    Wf = np.stack([np.linalg.solve(-Mref[p].T, W[p]) for p in range(NP)])
    base = np.array([40, 80, 300, 600, 1.0, 30.0])[:, None] * np.ones((1, NZ))
    dbase = -np.einsum("pij,pj->pi", Mref, base)
    cb = C_BOUND[:, None] * np.ones((1, NZ))
    instA = [random_instance(disc.A, disc.G, W, Wf, base, dbase, cb, duties, st.N, st.Nu, rng) for _ in range(200)]
    pA = out("e6/instances_deployed.bin"); write_instances(pA, instA)
    dfA = run_c(pA, reps=7)
    py = [py_exhaustive(d) for d in instA]
    dfA["J_py"] = [py[i][1] for i in dfA.id]
    dfA["seq0_py"] = [int(py[i][0][0]) for i in dfA.id]
    dfA["rel_err"] = (dfA.J - dfA.J_py).abs() / dfA.J_py.abs()
    dfA.to_csv(out("e6/timing_deployed.csv"), index=False)
    esp32_header(instA[:5], os.path.join(CDIR, "esp32_bench", "instances.h"))
    # ---- Part B: scaling -------------------------------------------------------------------
    U = {2: [0.0, 1.0], 3: [0.0, 0.8, 1.0], 4: [0.0, 0.8, 0.9, 1.0], 6: [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]}
    settings = ([("nz", v, dict(nz=v, np_=6, nu=4, Nu=6)) for v in [1, 2, 3, 5, 8]] +
                [("np", v, dict(nz=3, np_=v, nu=4, Nu=6)) for v in [1, 2, 4, 6]] +
                [("Nu", v, dict(nz=3, np_=6, nu=4, Nu=v)) for v in [3, 4, 5, 6, 7, 8]] +
                [("nu", v, dict(nz=3, np_=6, nu=v, Nu=6)) for v in [2, 3, 4, 6]])
    rows = []
    for factor, val, cfg in settings:
        duties_s = U[cfg["nu"]]
        insts = []
        for j in range(20):
            mdl = synthetic(cfg["nz"], cfg["np_"], duties_s, rng)
            insts.append(random_instance(*mdl, duties_s, 30, cfg["Nu"], rng))
        p = out(f"e6/inst_{factor}_{val}.bin"); write_instances(p, insts)
        reps = 3 if cfg["Nu"] >= 8 or cfg["nu"] >= 6 else 5
        d = run_c(p, reps=reps); d["factor"] = factor; d["value"] = val
        rows.append(d)
        print(factor, val, d.groupby("mode").time_us.median().round(1).to_dict(), flush=True)
    pd.concat(rows).to_csv(out("e6/timing_scaling.csv"), index=False)
    # memory footprint (bytes) of solver workspace + model data for the deployed dimensions
    nu, N, Nu, PZ, Z, P = 4, 30, 6, 18, 3, 6
    ws = 4 * (nu * N * PZ + N * PZ + P * Z * Z + (N + 1) * PZ + (N + 1) + (Nu + 1) * PZ) + 4 * (3 * Nu + 1)
    model = 4 * (2 * nu * P * Z * Z + 3 * PZ + N * PZ + N + PZ)
    summ = {"deployed_max_rel_err_vs_python": float(dfA.rel_err.max()),
            "deployed_first_move_agreement": float((dfA.seq0 == dfA.seq0_py).mean()),
            "deployed_median_time_us": dfA.groupby("mode").time_us.median().to_dict(),
            "deployed_p95_time_us": dfA.groupby("mode").time_us.quantile(0.95).to_dict(),
            "deployed_median_nodes": dfA.groupby("mode").nodes.median().to_dict(),
            "deployed_p95_nodes": dfA.groupby("mode").nodes.quantile(0.95).to_dict(),
            "workspace_bytes_deployed": ws, "model_bytes_deployed": model}
    json.dump(summ, open(out("e6/timing_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))
