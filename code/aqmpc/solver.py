"""Exact solution of the blocked discrete economic MPC problem.

Problem (one receding-horizon step):
  min_{v in U^Nu}  sum_{k=1..N} [ o_k <W, c_k> + M <1, (c_k - cb)_+ / cb> ] dt + rho sum_{k=0..N-1} P(u_k) dt
  s.t. c_{k+1} = A(u_k) c_k + G(u_k) d_k  (per pollutant, exact ZOH),  u_k = v_{b(k)},
       sum_k P_fan u_k^3 dt <= E_allow                           (hard)
Optimality is exact over the finite set U^Nu (move-blocked); nothing is claimed beyond it.

An optional terminal cost <Wf, c_N> with Wf >= 0 (stationary costate, see controllers.py)
approximates the cost-to-go beyond the horizon; it is non-decreasing in c_N, so the bound
below stays admissible when the comparison trajectory's terminal cost is added.

Admissible bound for branch-and-bound. Every A(u) = expm(M(u) dt) and G(u) are elementwise
non-negative because M(u) is Metzler. With A_lb = min_u A(u), G_lb = min_u G(u) (elementwise)
and c, d >= 0, any completion satisfies c_k >= x_k where x_{k+1} = A_lb x_k + G_lb d_k
(induction, using A_lb >= 0). Stage cost and penalty are non-decreasing in c and energy is >= 0,
so the cost of the comparison trajectory with zero energy is a lower bound on every completion.
"""
import numpy as np
from .params import NP, NZ


def block_lengths(N, Nu):
    return [N // Nu + (1 if i < N % Nu else 0) for i in range(Nu)]


class MPCProblem:
    def __init__(self, disc, W, cb, M, rho, p_fan, N, Nu, p_exp=3.0, blens=None):
        self.disc = disc
        self.A = disc.A                    # (nu,P,Z,Z)
        self.G = disc.G
        self.duties = disc.duties
        self.dt = disc.dt
        self.W = W                         # (P,Z) occupancy-free index weights phi/g * psi
        self.cb = cb[:, None] * np.ones((1, NZ))
        self.M = M
        self.rho = rho
        self.p_fan = p_fan
        if blens:                          # explicit (non-uniform) move blocks
            N, Nu = int(sum(blens)), len(blens)
        self.N, self.Nu = N, Nu
        self.blens = list(blens) if blens else block_lengths(N, Nu)
        self.e_step = p_fan * self.duties ** p_exp * self.dt      # fan energy per step, per duty
        self.A_lb = self.A.min(axis=0)
        self.G_lb = self.G.min(axis=0)

    # ---- stage cost -------------------------------------------------------------------
    def _stage(self, C, occ_k):
        """C (...,P,Z) -> stage cost (...)"""
        idx = np.einsum("...pz,pz->...", C, self.W) * occ_k
        pen = (np.maximum(C - self.cb, 0.0) / self.cb).sum(axis=(-1, -2))
        return (idx + self.M * pen) * self.dt

    # ---- exhaustive, vectorised breadth-first ------------------------------------------
    def solve_exhaustive(self, c0, D, occ, e_allow, Wf=None):
        """c0 (P,Z), D (N,P,Z) drive forecast, occ (N,) occupancy of predicted steps 1..N,
        Wf (P,Z) optional non-negative terminal weight (linear terminal cost <Wf, c_N>)."""
        nu = len(self.duties)
        C = c0[None].copy()
        J = np.zeros(1)
        E = np.zeros(1)
        seqs = np.zeros((1, 0), dtype=int)
        k = 0
        for L in self.blens:
            n = C.shape[0]
            Cn = np.broadcast_to(C[:, None], (n, nu, NP, NZ)).copy()
            Jn = np.broadcast_to(J[:, None], (n, nu)).copy()
            En = E[:, None] + L * self.e_step[None, :]
            for _ in range(L):
                Cn = np.einsum("upij,nupj->nupi", self.A, Cn) + np.einsum("upij,pj->upi", self.G, D[k])[None]
                Jn += self._stage(Cn, occ[k])
                k += 1
            Jn += self.rho * L * self.e_step[None, :]
            ok = En <= e_allow + 1e-12
            ii, uu = np.nonzero(ok)
            C = Cn[ii, uu]
            J = Jn[ii, uu]
            E = En[ii, uu]
            seqs = np.concatenate([seqs[ii], uu[:, None]], axis=1)
        if Wf is not None:
            J = J + np.einsum("npz,pz->n", C, Wf)
        b = int(np.argmin(J))
        return seqs[b], float(J[b]), seqs.shape[0]

    # ---- evaluation of one sequence ----------------------------------------------------
    def evaluate(self, seq, c0, D, occ, Wf=None):
        c = c0.copy()
        J = 0.0
        E = 0.0
        k = 0
        for b, L in enumerate(self.blens):
            iu = seq[b]
            for _ in range(L):
                c = np.einsum("pij,pj->pi", self.A[iu], c) + np.einsum("pij,pj->pi", self.G[iu], D[k])
                J += self._stage(c, occ[k])
                k += 1
            E += L * self.e_step[iu]
            J += self.rho * L * self.e_step[iu]
        if Wf is not None:
            J += float((c * Wf).sum())
        return J, E

    # ---- lower bound on the cost-to-go from step k with state c ------------------------
    def lower_bound(self, c, k, D, occ, Wf=None):
        x = c.copy()
        lb = 0.0
        for j in range(k, self.N):
            x = np.einsum("pij,pj->pi", self.A_lb, x) + np.einsum("pij,pj->pi", self.G_lb, D[j])
            lb += self._stage(x, occ[j])
        if Wf is not None:
            lb += float((x * Wf).sum())
        return lb

    # ---- depth-first branch and bound (reference implementation, counts nodes) ---------
    def solve_bnb(self, c0, D, occ, e_allow, warm=None, Wf=None):
        nu = len(self.duties)
        starts = np.cumsum([0] + self.blens)
        best = {"J": np.inf, "seq": None}
        stats = {"nodes": 0, "leaves": 0, "pruned_bound": 0, "pruned_energy": 0}
        # incumbent: warm start (shifted previous plan) and the constant sequences
        cands = [np.full(self.Nu, iu) for iu in range(nu)]
        if warm is not None:
            cands.insert(0, np.asarray(warm))
        for s in cands:
            Js, Es = self.evaluate(s, c0, D, occ, Wf)
            if Es <= e_allow + 1e-12 and Js < best["J"]:
                best = {"J": Js, "seq": np.array(s)}
        order0 = list(range(nu))

        def rec(b, c, J, E, prefix):
            stats["nodes"] += 1
            if b == self.Nu:
                stats["leaves"] += 1
                if Wf is not None:
                    J = J + float((c * Wf).sum())
                if J < best["J"]:
                    best["J"], best["seq"] = J, np.array(prefix)
                return
            order = order0
            if warm is not None:
                order = [int(warm[b])] + [u for u in order0 if u != int(warm[b])]
            for iu in order:
                L = self.blens[b]
                E2 = E + L * self.e_step[iu]
                if E2 > e_allow + 1e-12:
                    stats["pruned_energy"] += 1
                    continue
                x = c
                J2 = J + self.rho * L * self.e_step[iu]
                for k in range(starts[b], starts[b + 1]):
                    x = np.einsum("pij,pj->pi", self.A[iu], x) + np.einsum("pij,pj->pi", self.G[iu], D[k])
                    J2 += self._stage(x, occ[k])
                if J2 + self.lower_bound(x, starts[b + 1], D, occ, Wf) >= best["J"]:
                    stats["pruned_bound"] += 1
                    continue
                rec(b + 1, x, J2, E2, prefix + [iu])

        rec(0, c0, 0.0, 0.0, [])
        return best["seq"], best["J"], stats
