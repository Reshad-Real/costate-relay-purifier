"""Additional baselines requested for the ISA Transactions version.

THC          tuned multilevel threshold controller on the worst particulate reading, with hysteresis and
             optional occupancy gating (the strongest rule-based competitor; thresholds tuned on the same
             cost as PI and the timer).
MPC-track    conventional (zone-)tracking MPC: same model, horizon, blocking, forecast and exact solver as
             MPC-FF, but the stage cost penalises the squared exceedance of particulate setpoints during
             occupied hours instead of the health-weighted index; no economic terminal cost.
MPC-RR       relax-and-round: the economic MPC problem solved over continuous duties u in [0,1]^Nu by a
             local NLP solver (L-BFGS-B, warm-started), then rounded to the nearest admissible duty.
"""
import numpy as np
from dataclasses import replace
from scipy.optimize import minimize
from .params import DUTY_LEVELS, NP, NZ, iPM25, iPM10, MPCSettings
from .scenarios import SHIFT
from .controllers import MPC
from .solver import MPCProblem

PM = [iPM25, iPM10]


def occupied_window(t):
    """Same enabling window as the occupancy-gated relay: 30 min before the shift to its end."""
    return (SHIFT[0] - 0.5) <= t < SHIFT[1]


class HysteresisThreshold:
    """u = DUTY_LEVELS[level]; level rises as soon as the worst particulate reading s reaches a threshold
    and falls only when s drops below that threshold minus the hysteresis band."""
    period = 1

    def __init__(self, th1=35.0, dth=20.0, hyst=5.0, r10=2.5, gate=True, avg=1, e_shift=None,
                 p_fan=6.96, p_exp=3.0, dt=10 / 3600, name="THC"):
        self.th = np.array([th1, th1 + dth, th1 + 3 * dth], dtype=float)
        self.hyst, self.r10, self.gate, self.name = float(hyst), float(r10), bool(gate), name
        self.avg = int(avg)                    # moving average over `avg` 10-s readings (1 = raw)
        self.e_shift, self.p_fan, self.p_exp, self.dt = e_shift, p_fan, p_exp, dt   # optional energy cut-off

    def reset(self, shift, ctx):
        self.level = 0
        self.hist = []
        self.e_used = 0.0

    def act(self, i, t, y):
        u = self._law(i, t, y)
        if self.e_shift is not None:
            if self.e_used + self.p_fan * u ** self.p_exp * self.dt > self.e_shift:
                u = 0.0
            self.e_used += self.p_fan * u ** self.p_exp * self.dt
        return u

    def _law(self, i, t, y):
        if self.avg > 1:
            self.hist.append(y)
            self.hist = self.hist[-self.avg:]
            y = np.mean(self.hist, axis=0)
        if self.gate and not occupied_window(t):
            self.level = 0
            return 0.0
        s = max(y[iPM25].max(), y[iPM10].max() / self.r10)
        up = int((s >= self.th).sum())
        down = int((s >= self.th - self.hyst).sum())
        self.level = max(up, min(self.level, down))
        return float(DUTY_LEVELS[self.level])


class TrackingProblem(MPCProblem):
    """Exact blocked MPC with a one-sided quadratic tracking stage cost on the particulate channels."""

    def __init__(self, *a, q=0.01, r25=25.0, r10_ratio=2.5, **kw):
        super().__init__(*a, **kw)
        self.q = float(q)
        self.r = np.array([r25, r10_ratio * r25])[:, None] * np.ones((1, NZ))

    def _stage(self, C, occ_k):
        dev = np.maximum(C[..., PM, :] - self.r, 0.0) / self.r
        trk = (dev ** 2).sum(axis=(-1, -2)) * self.q * occ_k
        pen = (np.maximum(C - self.cb, 0.0) / self.cb).sum(axis=(-1, -2))
        return (trk + self.M * pen) * self.dt


class TrackingMPC(MPC):
    def __init__(self, model_plant, forecaster=None, settings=MPCSettings(), q=0.01, r25=25.0, name="MPC-track"):
        settings = replace(settings, terminal=False)
        super().__init__(model_plant, forecaster, forecast="learned", settings=settings, name=name)
        self.prob = TrackingProblem(self.disc, self.W, settings.bound, settings.M, settings.rho,
                                    model_plant.fan.p_fan, settings.N, settings.Nu, model_plant.fan.p_exp,
                                    q=q, r25=r25)


class RelaxedProblem:
    """Continuous relaxation of the blocked economic MPC problem: u_b in [0,1], A(u), G(u) interpolated
    linearly from a fine grid of exact discretisations, P(u) = P_fan u^p."""

    def __init__(self, prob: MPCProblem, disc, n_grid=101):
        self.p = prob
        self.ug = np.linspace(0.0, 1.0, n_grid)
        self.Ag = np.stack([disc.mats(u)[0] for u in self.ug])        # (n,P,Z,Z)
        self.Gg = np.stack([disc.mats(u)[1] for u in self.ug])
        self.p_exp = None

    def _AG(self, u):
        x = np.clip(u, 0.0, 1.0) * (len(self.ug) - 1)
        i = min(int(x), len(self.ug) - 2)
        f = x - i
        return (1 - f) * self.Ag[i] + f * self.Ag[i + 1], (1 - f) * self.Gg[i] + f * self.Gg[i + 1]

    def cost(self, v, c0, D, occ, Wf, p_fan, p_exp):
        pr = self.p
        c = c0.copy()
        J = 0.0
        k = 0
        for b, L in enumerate(pr.blens):
            A, G = self._AG(v[b])
            for _ in range(L):
                c = np.einsum("pij,pj->pi", A, c) + np.einsum("pij,pj->pi", G, D[k])
                J += pr._stage(c, occ[k])
                k += 1
            J += pr.rho * L * p_fan * max(v[b], 0.0) ** p_exp * pr.dt
        if Wf is not None:
            J += float((c * Wf).sum())
        return J


class RelaxRoundMPC(MPC):
    """Economic MPC solved by continuous relaxation (local NLP) and nearest-level rounding."""

    def __init__(self, model_plant, forecaster=None, settings=MPCSettings(), name="MPC-RR"):
        super().__init__(model_plant, forecaster, forecast="learned", settings=settings, name=name)
        self.rel = RelaxedProblem(self.prob, self.disc)
        self.p_fan = model_plant.fan.p_fan
        self.nlp_evals = []

    def reset(self, shift, ctx):
        super().reset(shift, ctx)
        self.v_prev = np.full(self.s.Nu, 0.8)
        # route the exact solver call of the parent through the relaxation
        self.prob.solve_exhaustive = self._solve_relaxed

    def _solve_relaxed(self, c0, D, occ, e_allow, Wf=None):
        f = lambda v: self.rel.cost(v, c0, D, occ, Wf, self.p_fan, self.p_exp)
        x0 = np.concatenate([self.v_prev[1:], self.v_prev[-1:]])
        res = minimize(f, x0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * self.s.Nu,
                       options={"maxiter": 60, "eps": 1e-4})
        self.nlp_evals.append(res.nfev)
        v = np.clip(res.x, 0.0, 1.0)
        self.v_prev = v
        seq = np.array([int(np.argmin(np.abs(self.s.duties - vb))) for vb in v])
        return seq, float(res.fun), 0


class ScheduledDuty:
    """Constant duty u during the occupied shift, optionally switched on `pre` hours before it."""
    period = 1

    def __init__(self, u=0.8, pre=0.0, name=None):
        self.u, self.pre = float(u), float(pre)
        self.name = name or f"Sched-{int(round(100 * u))}"

    def reset(self, shift, ctx):
        pass

    def act(self, i, t, y):
        return self.u if (SHIFT[0] - self.pre) <= t < SHIFT[1] else 0.0


class PIGated:
    """PI on the 60-s mean of the worst normalised particulate reading, occupancy-gated, with conditional
    integration (anti-windup) and first-order sigma-delta modulation of the continuous output onto the
    admissible duties (the applied duty averages to the PI output)."""
    period = 6

    def __init__(self, r25=25.0, kp=0.5, ki=1.0, r10_ratio=2.4, gate=True, dt=1 / 60, name="PI-SD"):
        self.r25, self.r10 = float(r25), float(r10_ratio * r25)
        self.kp, self.ki, self.gate, self.dt, self.name = float(kp), float(ki), bool(gate), dt, name

    def reset(self, shift, ctx):
        self.I, self.err, self.u, self.buf = 0.0, 0.0, 0.0, []

    def act(self, i, t, y):
        self.buf.append(y)
        if i % self.period != 0:
            return self.u
        if self.gate and not occupied_window(t):
            self.I, self.err, self.u = 0.0, 0.0, 0.0
            return 0.0
        ya = np.mean(self.buf[-self.period:], axis=0)
        e = max(ya[iPM25].max() / self.r25, ya[iPM10].max() / self.r10) - 1.0
        v = self.kp * e + self.I
        if 0.0 < v < 1.0 or (v >= 1.0 and e < 0) or (v <= 0.0 and e > 0):
            self.I += self.ki * e * self.dt
        target = float(np.clip(self.kp * e + self.I, 0.0, 1.0))
        x = target + self.err
        self.u = float(DUTY_LEVELS[int(np.argmin(np.abs(DUTY_LEVELS - x)))])
        self.err = float(np.clip(x - self.u, -1.0, 1.0))
        return self.u


class RolloutCostate:
    """Costate law in rollout form: one-step lookahead whose terminal cost is the exact cost-to-go of the
    constant-duty base policy u = u_base up to the end of the day (linear in the state):
        u_k = argmin_{u in U, budget-feasible}  rho P(u) dt + sum_p (o_{k+1} dt W_p + Lam_{p,k+1})^T c_{p,k+1}(u)
                                                 + M pen(c_{k+1}(u)) dt,
        c_{p,k+1}(u) = A_p(u) c_{p,k} + G_p(u) d_{p,k},
        Lam_{p,j} = A_p(u_base)^T (o_{j+1} dt W_p + Lam_{p,j+1}),  Lam_{p,K} = 0.
    o is the shift occupancy indicator used by the exposure index. With exact state and drive and M = 0 the
    shift cost of this law never exceeds that of the base policy (rollout improvement)."""
    period = 6

    def __init__(self, model_plant, settings=MPCSettings(), u_base=0.8, info="estimate", penalty=False,
                 name="Rollout-CL"):
        # penalty=False (mu = 0) is the setting of Theorem 1 and the default since the measured-outdoor rerun: with
        # the soft-bound penalty, which is not part of the scored cost J, the law loses at first order whenever the
        # bounds are active (16 % of occupied steps under the Dhaka outdoor record).
        from .plant import Discretised
        from .exposure import weight_tensor
        from .scenarios import T_END
        self.s, self.u_base, self.info, self.name = settings, float(u_base), info, name
        self.disc = Discretised(model_plant, settings.dt, settings.duties)
        Ab, _ = self.disc.mats(self.u_base)
        self.W = weight_tensor()
        dt = settings.dt
        self.K = int(round(T_END / dt))
        tk = dt * np.arange(self.K + 1)
        self.o = ((tk >= SHIFT[0]) & (tk < SHIFT[1])).astype(float)
        Lam = np.zeros((self.K + 1, NP, NZ))
        for j in range(self.K - 1, -1, -1):
            Lam[j] = np.einsum("pji,pj->pi", Ab, self.o[j + 1] * dt * self.W + Lam[j + 1])
        self.Lam = Lam
        self.cb = settings.bound[:, None] * np.ones((1, NZ))
        self.M = settings.M if penalty else 0.0
        self.p_fan, self.p_exp = model_plant.fan.p_fan, model_plant.fan.p_exp
        self.pcost = self.p_fan * settings.duties ** self.p_exp

    def reset(self, shift, ctx):
        from .forecast import OnlineDrive
        self.ctx, self.buf, self.ubuf = ctx, [], []
        self.u, self.e_used, self.k = 0.0, 0.0, 0
        self.od = OnlineDrive(self.disc, None, shift.plan, mode="persistence")

    def act(self, i, t, y):
        self.buf.append(y)
        if i % self.period != 0:
            return self._apply(self.u)
        dt = self.s.dt
        yavg = np.mean(self.buf[-self.period:], axis=0)
        if self.k == 0:
            self.od.init_steady(yavg, 0.0)
        else:
            self.od.update(t, yavg, float(np.mean(self.ubuf[-self.period:])))
        self.k += 1
        if self.info == "oracle":
            c, d = self.ctx["truth_state"](), self.ctx["true_drive"](t, 1, dt)[0]
        else:
            c, d = yavg, self.od.forecast(t, 1, dt)[0]
        k = min(int(round(t / dt)), self.K - 1)
        C1 = np.einsum("upij,pj->upi", self.disc.A, c) + np.einsum("upij,pj->upi", self.disc.G, d)
        wgt = self.o[k + 1] * dt * self.W + self.Lam[k + 1]
        cost = self.s.rho * self.pcost * dt + np.einsum("upz,pz->u", C1, wgt)
        if self.M > 0:
            cost = cost + self.M * (np.maximum(C1 - self.cb, 0.0) / self.cb).sum(axis=(1, 2)) * dt
        rem = self.s.e_shift - self.e_used
        cost = np.where(self.pcost * dt <= rem + 1e-12, cost, np.inf)
        self.u = float(self.s.duties[int(np.argmin(cost))])
        return self._apply(self.u)

    def _apply(self, u):
        self.ubuf.append(u)
        self.e_used += self.p_fan * u ** self.p_exp * self.s.dt / self.period
        return u
