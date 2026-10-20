"""Disturbance (drive) estimation and schedule-based forecasting.

Drive per zone and pollutant: d = lam0*c_out + S/V. It is not measured; the controller recovers
it by inverting its own discrete model on consecutive (averaged) measurements,
    d_inv(k-1) = G(u_{k-1})^{-1} (y_k - A(u_{k-1}) y_{k-1}),
which is model-dependent (reviewer 2, comment 5). In simulation the true drive is also known, so
the forecaster can be trained and scored against either.

Forecast:  d_hat(t_k + l) = b_k + m(t_k + l) + alpha^l r_k
  m(t)  = sum_type a_type * x_type(t) + a_pres * presence(t)     (planned-activity profile)
  x_type(t) = trapezoidal indicator of PLANNED windows of that type (from the production plan)
  a_*   identified by least squares on training shifts with per-shift intercepts
  b_k   slowly varying intercept (outdoor infiltration), tracked online
  r_k   fast residual, AR(1) with coefficient alpha fitted on training residuals
"""
import numpy as np
from .params import NP, NZ
from .scenarios import EVENT_TYPES, SHIFT, LUNCH, _trap

RAMP = 2.0 / 60


def presence(t):
    t = np.asarray(t, dtype=float)
    occ = ((t >= SHIFT[0]) & (t < SHIFT[1])).astype(float)
    return occ * np.where((t >= LUNCH[0]) & (t < LUNCH[1]), 0.2, 1.0)


def regressors(t, plan):
    """(n, n_types+1) planned-activity regressors at times t."""
    t = np.asarray(t, dtype=float)
    X = np.zeros((len(t), len(EVENT_TYPES) + 1))
    for e in plan:
        X[:, EVENT_TYPES.index(e.kind)] += _trap(t, e.t0, e.t1, RAMP)
    X[:, -1] = presence(t)
    return X


def invert_one(disc, y_next, y, u):
    A, G = disc.mats(u)
    r = y_next - np.einsum("pij,pj->pi", A, y)
    return np.stack([np.linalg.solve(G[p], r[p]) for p in range(NP)])


def invert_drive(disc, Y, Ubar):
    """Y (K,P,Z) interval-averaged measurements at control steps, Ubar (K,) mean duty applied
    over [k,k+1). Returns D (K-1,P,Z) with D[k] the drive over [k,k+1)."""
    return np.stack([invert_one(disc, Y[k + 1], Y[k], Ubar[k]) for k in range(Y.shape[0] - 1)])


class ScheduleForecaster:
    def __init__(self, n_types=len(EVENT_TYPES)):
        self.a = None          # (n_types+1, P, Z)
        self.alpha = None      # (P,)
        self.sd_r = None       # (P,Z)

    def fit(self, records):
        """records: list of (t (K,), plan, D (K,P,Z)) with D the drive used for training."""
        Xs, Ys, ids = [], [], []
        for s, (t, plan, D) in enumerate(records):
            Xs.append(regressors(t, plan))
            Ys.append(D.reshape(len(t), -1))
            ids.append(np.full(len(t), s))
        X = np.concatenate(Xs)
        Y = np.concatenate(Ys)
        ids = np.concatenate(ids)
        ns = len(records)
        Dm = np.zeros((len(ids), ns))
        Dm[np.arange(len(ids)), ids] = 1.0
        Xf = np.hstack([X, Dm])
        coef, *_ = np.linalg.lstsq(Xf, Y, rcond=None)
        a = np.maximum(coef[: X.shape[1]], 0.0)
        self.a = a.reshape(X.shape[1], NP, NZ)
        # residual AR(1) per pollutant, pooled over zones and shifts
        res = (Y - Xf @ np.vstack([a, coef[X.shape[1]:]])).reshape(len(ids), NP, NZ)
        alpha = np.zeros(NP)
        sd = np.zeros((NP, NZ))
        for p in range(NP):
            num = den = 0.0
            for s in range(ns):
                r = res[ids == s, p, :]
                num += (r[1:] * r[:-1]).sum()
                den += (r[:-1] ** 2).sum()
            alpha[p] = np.clip(num / den if den > 0 else 0.0, 0.0, 0.98)
            sd[p] = res[:, p, :].std(axis=0) + 1e-9
        self.alpha, self.sd_r = alpha, sd
        return self

    def profile(self, t, plan):
        X = regressors(t, plan)
        return np.einsum("nk,kpz->npz", X, self.a)


class OnlineDrive:
    """Online estimate of the drive from measurements plus the forecast used by the MPC."""

    def __init__(self, disc, forecaster, plan, mode="learned", smooth=0.35, slow_tau=20.0):
        self.disc = disc
        self.f = forecaster
        self.plan = plan
        self.mode = mode
        self.smooth = smooth
        self.g_slow = 1.0 - np.exp(-1.0 / slow_tau)       # per control step (1 min)
        self.d_s = None
        self.b = None
        self.r = np.zeros((NP, NZ))
        self.prev_y = None

    def init_steady(self, y, u):
        M = self.disc.M0 + u * self.disc.M1
        d0 = np.stack([-(M[p] @ y[p]) for p in range(NP)])
        self.d_s = np.maximum(d0, 0.0)
        self.b = self.d_s.copy()
        self.prev_y = y.copy()

    def update(self, t, y, u_mean):
        """Called at control step t with the new interval-averaged measurement y and the mean
        duty applied over the elapsed interval."""
        d_inv = invert_one(self.disc, y, self.prev_y, u_mean)
        self.d_s = (1 - self.smooth) * self.d_s + self.smooth * d_inv
        if self.f is not None and self.f.a is not None:
            prof = self.f.profile(np.array([t]), self.plan)[0]
            resid = self.d_s - prof - self.b
            self.b = np.maximum(self.b + self.g_slow * resid, 0.0)
            self.r = self.d_s - prof - self.b
        self.prev_y = y.copy()
        return d_inv

    def forecast(self, t, N, dt):
        tt = t + dt * np.arange(N)
        if self.mode == "persistence" or self.f is None:
            return np.broadcast_to(np.maximum(self.d_s, 0.0), (N, NP, NZ)).copy()
        prof = self.f.profile(tt, self.plan)
        decay = self.f.alpha[None, :, None] ** np.arange(1, N + 1)[:, None, None]
        return np.maximum(self.b[None] + prof + decay * self.r[None], 0.0)
