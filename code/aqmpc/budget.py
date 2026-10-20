"""Shift-level energy-budget allocation by adapting the economic weight (shadow price) rho.

The hard budget-state constraint of the MPC guarantees that cumulative fan energy never exceeds the
shift budget, but says nothing about *when* the budget is spent. Allocation is delegated to rho:
  'fixed'   : rho tuned offline, never changed
  'pacing'  : feedback only - rho is raised/lowered when the spent share runs ahead of/behind the
              elapsed share of the occupied shift (integral action on log rho)
  'rollout' : forecast-based - rho is set so that a model rollout of the costate band law over the
              remaining window, driven by the schedule forecast, spends exactly the remaining budget
Re-allocation runs every `every` control steps.
"""
import numpy as np
from .params import NP, NZ
from .scenarios import SHIFT
from .forecast import presence

RHO_GRID = np.exp(np.linspace(np.log(1e-4), np.log(5.0), 40))


def occupied_share(t, t_end):
    """Share of the remaining occupied time already elapsed, for the pacing law."""
    a, b = SHIFT
    tot = (b - a) + 0.5
    done = np.clip(t - (a - 0.5), 0, tot)
    return done / tot


class Allocator:
    def __init__(self, mode, rho0, e_shift, every=15, gain=1.5):
        self.mode = mode
        self.rho = rho0
        self.e_shift = e_shift
        self.every = every
        self.gain = gain
        self.k = 0

    def update(self, t, t_end, e_used, rollout_fn=None):
        self.k += 1
        if self.mode == "fixed" or (self.k - 1) % self.every != 0:
            return self.rho
        rem = max(self.e_shift - e_used, 0.0)
        if self.mode == "pacing":
            spent = e_used / self.e_shift
            el = occupied_share(t, t_end)
            if el > 0.02:
                self.rho = float(np.clip(self.rho * np.exp(self.gain * (spent - el)), 1e-4, 5.0))
        elif self.mode == "rollout" and rollout_fn is not None:
            use = rollout_fn(RHO_GRID)              # energy used by the rollout for each rho
            ok = np.nonzero(use <= rem)[0]
            self.rho = float(RHO_GRID[ok[0]]) if len(ok) else float(RHO_GRID[-1])
        return self.rho


def cf_rollout(disc, Minv_T, W, cb, M, p_fan, p_exp, x0, D, t0, dt, rhos):
    """Energy spent by the costate band law over the forecast window for a batch of rho values.
    x0 (P,Z) state, D (K,P,Z) forecast drive, rhos (R,). Returns (R,) energy in Wh."""
    R = len(rhos)
    duties = disc.duties
    X = np.broadcast_to(x0, (R, NP, NZ)).copy()
    E = np.zeros(R)
    pcost = p_fan * duties ** p_exp
    for k in range(D.shape[0]):
        t = t0 + k * dt
        occ = 1.0 if (SHIFT[0] - 0.5) <= t < SHIFT[1] else 0.0
        w = W[None] * occ + M * (X > cb[None]) / cb[None]
        lam = np.einsum("pij,rpj->rpi", Minv_T, w)
        B = -np.einsum("rpi,pij,rpj->r", lam, disc.M1, X)
        cost = rhos[:, None] * pcost[None, :] - duties[None, :] * B[:, None]
        iu = np.argmin(cost, axis=1)
        E += pcost[iu] * dt
        X = np.einsum("rpij,rpj->rpi", disc.A[iu], X) + np.einsum("rpij,pj->rpi", disc.G[iu], D[k])
    return E
