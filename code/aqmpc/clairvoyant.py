"""Clairvoyant benchmark: the best duty schedule found with perfect knowledge of the whole day's drive and the true
initial state, on the controllers' decision grid (one decision per minute, held over the 10-s plant steps).

For a fixed schedule the plant is linear in the state and the cost
    J = sum_i occ_i dt w'c_i + rho dt sum_i P(u_i)            (exactly the J of the closed-loop metrics)
is linear in the states, so the effect of changing the duty of ONE decision block [a, b) is exact:
    dJ = rho dt sum_{i in block} (P_new - P_old) + sum_{i in block} occ_i dt w'(c_i^new - c_i^old)
         + mu_b'(c_b^new - c_b^old),
with mu the backward costate of the current schedule at the block boundaries (cost-to-go is linear in the state).
A batched exact 1-opt local search from several starting schedules returns a schedule that no single-block change
improves; its cost is an upper bound on the clairvoyant optimum J*, so J_controller - J_clairvoyant is a lower bound
on how much any amount of future knowledge could have saved on that day."""
import numpy as np
from .params import NP, NZ
from .plant import TruthSim
from .exposure import weight_tensor
from .scenarios import DT_SIM


class Clairvoyant:
    def __init__(self, plant, shift, duties, rho, block=6):
        sim = TruthSim(plant, DT_SIM)
        self.drive = plant.lam0 * shift.c_out[:, :, None] + shift.S / sim.vol[None, None, :]   # as in run_shift
        self.c0 = sim.steady_state(0.0, self.drive[0])
        self.n = len(shift.t)
        assert self.n % block == 0
        self.block, self.nb = block, self.n // block
        self.duties = np.asarray(duties, dtype=float)
        self.nu = len(self.duties)
        self.rho, self.dt = float(rho), DT_SIM
        self.W = weight_tensor()                                   # (P, Z)
        self.occw = (shift.occ * DT_SIM).reshape(self.nb, block)    # occupancy weight of every 10-s step
        self.Pblk = plant.fan.power(self.duties) * DT_SIM * block   # energy of one block at each duty
        A = np.stack([sim._mats(u)[0] for u in self.duties])        # (U, P, Z, Z) one 10-s step
        G = np.stack([sim._mats(u)[1] for u in self.duties])
        # powers A^r (r = 0..block) and the drive part Z[m, j, r] of the state r steps into block m under duty j
        self.Apow = np.empty((self.nu, block + 1, NP, NZ, NZ))
        self.Apow[:, 0] = np.eye(NZ)[None, None]
        for r in range(block):
            self.Apow[:, r + 1] = np.einsum("jpik,jpkl->jpil", A, self.Apow[:, r])
        d = self.drive.reshape(self.nb, block, NP, NZ)
        self.Z = np.zeros((self.nb, self.nu, block + 1, NP, NZ))
        for r in range(block):
            self.Z[:, :, r + 1] = (np.einsum("jpik,mjpk->mjpi", A, self.Z[:, :, r])
                                   + np.einsum("jpik,mpk->mjpi", G, d[:, r]))
        # occupancy-weighted state-sum operator of a block: S_j = sum_{r<block} occ_r W'A^r (depends on block via occ)
        self.WA = np.einsum("pz,jrpzk->jrpk", self.W, self.Apow[:, :block])       # (U, block, P, Z): w'A^r
        self.WZ = np.einsum("pz,mjrpz->mjr", self.W, self.Z[:, :, :block])          # (nb, U, block)

    # ------------------------------------------------------------------ evaluation
    def forward(self, idx):
        """Block-start states Cs (nb+1, P, Z) and exact cost of the block schedule idx (nb,) of duty indices."""
        Cs = np.empty((self.nb + 1, NP, NZ))
        Cs[0] = self.c0
        J = 0.0
        for m in range(self.nb):
            j = idx[m]
            J += float(np.einsum("r,rpk,pk->", self.occw[m], self.WA[j], Cs[m]) + self.occw[m] @ self.WZ[m, j])
            Cs[m + 1] = np.einsum("pik,pk->pi", self.Apow[j, self.block], Cs[m]) + self.Z[m, j, self.block]
        J += self.rho * self.Pblk[idx].sum()
        return Cs, J

    def costate(self, idx):
        """mu[m] (nb+1, P, Z): coefficient of the block-start state in the cost-to-go (mu[nb] = 0)."""
        mu = np.zeros((self.nb + 1, NP, NZ))
        for m in range(self.nb - 1, -1, -1):
            j = idx[m]
            mu[m] = (np.einsum("r,rpk->pk", self.occw[m], self.WA[j])
                     + np.einsum("pik,pi->pk", self.Apow[j, self.block], mu[m + 1]))
        return mu

    def deltas(self, idx, Cs, mu):
        """Exact cost change D[m, j] of setting block m alone to duty j."""
        # stage cost of block m under every duty, from its current start state
        stage = (np.einsum("mr,jrpk,mpk->mj", self.occw, self.WA, Cs[:-1])
                 + np.einsum("mr,mjr->mj", self.occw, self.WZ))
        end = (np.einsum("jpik,mpk->mjpi", self.Apow[:, self.block], Cs[:-1]) + self.Z[:, :, self.block])
        tail = np.einsum("mpi,mjpi->mj", mu[1:], end)
        tot = stage + tail + self.rho * self.Pblk[None, :]
        return tot - tot[np.arange(self.nb), idx][:, None]

    # ------------------------------------------------------------------ certificate
    def reach_box(self):
        """Elementwise bounds lo[m] <= c_m <= hi[m] on every block-start state reachable with any block schedule."""
        lo = np.empty((self.nb + 1, NP, NZ)); hi = np.empty_like(lo)
        lo[0] = hi[0] = self.c0
        Ab = self.Apow[:, self.block]
        for m in range(self.nb):
            nxt_lo = np.einsum("jpik,pk->jpi", Ab, lo[m]) + self.Z[m, :, self.block]
            nxt_hi = np.einsum("jpik,pk->jpi", Ab, hi[m]) + self.Z[m, :, self.block]
            lo[m + 1], hi[m + 1] = nxt_lo.min(axis=0), nxt_hi.max(axis=0)
        return lo, hi

    def lower_bound(self, idx, box=None):
        """Certified lower bound on the clairvoyant optimum J* (Proposition 'optimality certificate'):
        J* >= J(v) + sum_m min_{j, c in box_m} [Q_m(j, c) - Q_m(v_m, c)], with Q_m affine in c and exact
        (cost-difference identity with the costate of v). Returns (LB, J(v), per-block contributions)."""
        idx = np.asarray(idx)
        Cs, J = self.forward(idx)
        mu = self.costate(idx)
        lo, hi = self.reach_box() if box is None else box
        D = self.deltas(idx, Cs, mu)                                            # Q_m(j, c_m^v) - Q_m(v_m, c_m^v)
        a = (np.einsum("mr,jrpk->mjpk", self.occw, self.WA)                     # coefficient of c_m in Q_m(j, .)
             + np.einsum("jpik,mpi->mjpk", self.Apow[:, self.block], mu[1:]))
        g = a - a[np.arange(self.nb), idx][:, None]                             # (nb, U, P, Z)
        dev = np.where(g > 0, lo[:-1, None] - Cs[:-1, None], hi[:-1, None] - Cs[:-1, None])
        rel = D + np.einsum("mjpk,mjpk->mj", g, dev)
        contrib = np.minimum(rel.min(axis=1), 0.0)
        return J + contrib.sum(), J, contrib

    # ------------------------------------------------------------------ exact optimum
    def exact_milp(self, time_limit=900.0, rel_gap=1e-7, relax=False, box=None, lp_method=None):
        """Exact clairvoyant optimum over one-minute block schedules by a mixed-integer linear programme.

        Convex-hull (disaggregated) form of the switched linear dynamics: for every block m and duty j a copy
        x[m,j] of the block-start state with lo_m z[m,j] <= x[m,j] <= hi_m z[m,j] (reachable box), sum_j z[m,j] = 1,
        sum_j x[0,j] = c0 and sum_j x[m+1,j] = sum_j (A_j^6 x[m,j] + Z[m,j] z[m,j]). With binary z this is exact;
        with relax=True it returns the LP bound. Returns (J, idx or None, status, mip_gap)."""
        import scipy.sparse as sp
        from scipy.optimize import milp, LinearConstraint, Bounds
        nb, nu, n = self.nb, self.nu, NP * NZ
        lo, hi = self.reach_box() if box is None else box
        nx = nb * nu * n
        nz = nb * nu
        X = lambda m, j: (m * nu + j) * n                               # offset of x[m, j]
        Zi = lambda m, j: nx + m * nu + j                               # index of z[m, j]
        # cost
        c = np.zeros(nx + nz)
        a = (np.einsum("mr,jrpk->mjpk", self.occw, self.WA)).reshape(nb, nu, n)
        b = (np.einsum("mr,mjr->mj", self.occw, self.WZ) + self.rho * self.Pblk[None, :])
        for m in range(nb):
            for j in range(nu):
                c[X(m, j):X(m, j) + n] = a[m, j]
                c[Zi(m, j)] = b[m, j]
        rows, cols, vals, lb_eq, ub_eq = [], [], [], [], []
        r = 0
        # sum_j z = 1
        for m in range(nb):
            for j in range(nu):
                rows.append(r); cols.append(Zi(m, j)); vals.append(1.0)
            lb_eq.append(1.0); ub_eq.append(1.0); r += 1
        # initial state
        c0 = self.c0.reshape(n)
        for i in range(n):
            for j in range(nu):
                rows.append(r); cols.append(X(0, j) + i); vals.append(1.0)
            lb_eq.append(c0[i]); ub_eq.append(c0[i]); r += 1
        # dynamics: sum_j x[m+1,j] - sum_j (Ablk_j x[m,j] + Z_mj z_mj) = 0
        Ab = self.Apow[:, self.block]                                    # (U, P, Z, Z)
        Ablk = np.zeros((nu, n, n))
        for j in range(nu):
            for p in range(NP):
                Ablk[j, p * NZ:(p + 1) * NZ, p * NZ:(p + 1) * NZ] = Ab[j, p]
        Zb = self.Z[:, :, self.block].reshape(nb, nu, n)
        for m in range(nb - 1):
            for i in range(n):
                for j in range(nu):
                    rows.append(r); cols.append(X(m + 1, j) + i); vals.append(1.0)
                    nzc = np.nonzero(Ablk[j, i])[0]
                    for k in nzc:
                        rows.append(r); cols.append(X(m, j) + k); vals.append(-Ablk[j, i, k])
                    rows.append(r); cols.append(Zi(m, j)); vals.append(-Zb[m, j, i])
                lb_eq.append(0.0); ub_eq.append(0.0); r += 1
        # box: x - hi z <= 0 and lo z - x <= 0
        lo_f, hi_f = lo[:-1].reshape(nb, n), hi[:-1].reshape(nb, n)
        for m in range(nb):
            for j in range(nu):
                for i in range(n):
                    rows += [r, r]; cols += [X(m, j) + i, Zi(m, j)]; vals += [1.0, -hi_f[m, i]]
                    lb_eq.append(-np.inf); ub_eq.append(0.0); r += 1
                    rows += [r, r]; cols += [X(m, j) + i, Zi(m, j)]; vals += [-1.0, lo_f[m, i]]
                    lb_eq.append(-np.inf); ub_eq.append(0.0); r += 1
        A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nx + nz))
        integ = np.r_[np.zeros(nx), np.ones(nz) * (0 if relax else 1)]
        bnds = Bounds(np.r_[np.full(nx, -np.inf), np.zeros(nz)], np.r_[np.full(nx, np.inf), np.ones(nz)])
        if relax and lp_method is not None:            # same LP by linprog (e.g. interior point with crossover)
            from scipy.optimize import linprog
            lb_a, ub_a = np.asarray(lb_eq), np.asarray(ub_eq)
            eq = lb_a == ub_a
            res = linprog(c, A_ub=A[~eq], b_ub=ub_a[~eq], A_eq=A[eq], b_eq=ub_a[eq],
                          bounds=list(zip(bnds.lb, bnds.ub)), method=lp_method,
                          options={"time_limit": time_limit, "disp": False})
            return (float(res.fun) if res.status == 0 else None), None, int(res.status), None
        res = milp(c, constraints=LinearConstraint(A, lb_eq, ub_eq), integrality=integ, bounds=bnds,
                   options={"time_limit": time_limit, "mip_rel_gap": rel_gap, "disp": False})
        if res.x is None:
            return None, None, res.status, None
        zsol = res.x[nx:].reshape(nb, nu)
        idx = zsol.argmax(axis=1) if not relax else None
        gap = getattr(res, "mip_gap", None)
        return float(res.fun), idx, int(res.status), gap

    # ------------------------------------------------------------------ search
    def improve(self, idx, max_iter=500, tol=1e-10):
        idx = np.asarray(idx).copy()
        Cs, J = self.forward(idx)
        it = 0
        for it in range(max_iter):
            mu = self.costate(idx)
            D = self.deltas(idx, Cs, mu)
            bj = D.argmin(axis=1)
            bd = D[np.arange(self.nb), bj]
            cand = np.flatnonzero(bd < -tol)
            if cand.size == 0:
                break
            order = cand[np.argsort(bd[cand])]
            k = order.size
            while True:
                new = idx.copy()
                new[order[:k]] = bj[order[:k]]
                Cs2, J2 = self.forward(new)
                if J2 < J - tol or k == 1:
                    break
                k = max(1, k // 2)
            if J2 >= J - tol:
                break
            idx, Cs, J = new, Cs2, J2
        return idx, J, it

    def schedule(self, idx):
        """Duty at every 10-s plant step."""
        return np.repeat(self.duties[np.asarray(idx)], self.block)

    def to_idx(self, U):
        """Block duty indices from a 10-s duty trace that is constant within blocks (value at the block start)."""
        u = np.asarray(U)[:: self.block][: self.nb]
        return np.abs(u[:, None] - self.duties[None, :]).argmin(axis=1)


class Playback:
    """Open-loop playback of a precomputed 10-s duty schedule (to evaluate it with the closed-loop simulator)."""
    def __init__(self, U, name="Clairvoyant"):
        self.U, self.name = np.asarray(U, dtype=float), name

    def reset(self, shift, ctx):
        pass

    def act(self, i, t, y):
        return float(self.U[i])
