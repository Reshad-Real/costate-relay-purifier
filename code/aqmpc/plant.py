"""Exact zero-order-hold discretisation of the multizone mass balance.

dc_p/dt = (M0_p + u M1_p) c_p + d_p,  d_p = lam0 * c_out_p + S_p / V   (per zone)
For a duty u held over dt and drive d held over dt:
    c+ = A_p(u) c + G_p(u) d,  A = expm(M dt),  G = M^{-1}(A - I).
"""
import numpy as np
from scipy.linalg import expm
from .params import NP, NZ


class Discretised:
    """Per-duty transition matrices for a given plant and step."""

    def __init__(self, plant, dt, duties):
        self.plant = plant
        self.dt = dt
        self.duties = np.asarray(duties, dtype=float)
        M0, M1 = plant.rate_matrices()
        self.M0, self.M1 = M0, M1
        nu = len(self.duties)
        self.A = np.zeros((nu, NP, NZ, NZ))
        self.G = np.zeros((nu, NP, NZ, NZ))
        for iu, u in enumerate(self.duties):
            for p in range(NP):
                M = M0[p] + u * M1[p]
                A = expm(M * dt)
                self.A[iu, p] = A
                self.G[iu, p] = np.linalg.solve(M, A - np.eye(NZ))
        self.vol = plant.vol
        self._cache = {}

    def mats(self, u):
        """(A, G) for an arbitrary duty u held over one step (cached)."""
        key = round(float(u), 3)
        if key not in self._cache:
            A = np.zeros((NP, NZ, NZ)); G = np.zeros((NP, NZ, NZ))
            for p in range(NP):
                M = self.M0[p] + key * self.M1[p]
                A[p] = expm(M * self.dt)
                G[p] = np.linalg.solve(M, A[p] - np.eye(NZ))
            self._cache[key] = (A, G)
        return self._cache[key]

    def drive(self, c_out, S):
        """Drive d (P,Z) from outdoor levels c_out (P,) and sources S (P,Z) in units*m3/h."""
        return self.plant.lam0 * c_out[:, None] + S / self.vol[None, :]

    def step(self, c, iu, d):
        """c (P,Z) -> next state under duty index iu and drive d (P,Z)."""
        return np.einsum("pij,pj->pi", self.A[iu], c) + np.einsum("pij,pj->pi", self.G[iu], d)

    def steady_state(self, iu, d):
        M = self.M0 + self.duties[iu] * self.M1
        return np.stack([np.linalg.solve(-M[p], d[p]) for p in range(NP)])


class TruthSim:
    """Truth plant at a fine step with arbitrary (continuous) duty, used for closed-loop runs."""

    def __init__(self, plant, dt_sim):
        self.plant = plant
        self.dt = dt_sim
        self.M0, self.M1 = plant.rate_matrices()
        self._cache = {}
        self.vol = plant.vol

    def _mats(self, u):
        key = round(float(u), 6)
        if key not in self._cache:
            A = np.zeros((NP, NZ, NZ))
            G = np.zeros((NP, NZ, NZ))
            for p in range(NP):
                M = self.M0[p] + u * self.M1[p]
                Ap = expm(M * self.dt)
                A[p] = Ap
                G[p] = np.linalg.solve(M, Ap - np.eye(NZ))
            self._cache[key] = (A, G)
        return self._cache[key]

    def drive(self, c_out, S):
        return self.plant.lam0 * c_out[:, None] + S / self.vol[None, :]

    def step(self, c, u, d):
        A, G = self._mats(u)
        return np.einsum("pij,pj->pi", A, c) + np.einsum("pij,pj->pi", G, d)

    def steady_state(self, u, d):
        M = self.M0 + u * self.M1
        return np.stack([np.linalg.solve(-M[p], d[p]) for p in range(NP)])
