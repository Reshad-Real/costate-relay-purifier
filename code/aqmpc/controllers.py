"""Controllers compared in the study. All act on the delayed, noisy sensor stream."""
import numpy as np
from .params import (RELAY_THRESHOLDS, DUTY_LEVELS, C_BOUND, NP, NZ, iPM25, iPM10, MPCSettings)
from .plant import Discretised
from .solver import MPCProblem
from .exposure import weight_tensor
from .forecast import OnlineDrive, presence
from .scenarios import SHIFT, T_END
from .budget import Allocator, cf_rollout


def relay_level(y, species=None):
    """Firmware law: worst pollutant (max over nodes) -> number of thresholds reached."""
    ymax = y.max(axis=1)                                   # DAG forwards per-pollutant max [FW]
    lv = (ymax[:, None] >= RELAY_THRESHOLDS).sum(axis=1)
    if species is not None:
        lv = lv[species]
    return int(lv.max())


class Relay:
    name = "Relay"
    period = 1                       # acts every 10 s step (CCU loop runs every ~5 s) [FW]

    def __init__(self, species=None, name=None):
        self.species = species
        if name:
            self.name = name

    def reset(self, shift, ctx):
        pass

    def act(self, i, t, y):
        return DUTY_LEVELS[relay_level(y, self.species)]


class AlwaysOn:
    period = 1

    def __init__(self, u):
        self.u = u
        self.name = f"AO-{int(round(100 * u))}"

    def reset(self, shift, ctx):
        pass

    def act(self, i, t, y):
        return self.u


def quantise(v):
    if v < 0.4:
        return 0.0
    if v < 0.85:
        return 0.8
    if v < 0.95:
        return 0.9
    return 1.0


class PI:
    """PI on the worst normalised particulate reading, quantised to the admissible duties."""
    name = "PI"
    period = 1

    def __init__(self, r25=25.0, r10=60.0, kp=1.0, ki=6.0, dt=10 / 3600):
        self.r25, self.r10, self.kp, self.ki, self.dt = r25, r10, kp, ki, dt

    def reset(self, shift, ctx):
        self.I = 0.0

    def act(self, i, t, y):
        e = max(y[iPM25].max() / self.r25, y[iPM10].max() / self.r10) - 1.0
        v = self.kp * e + self.I
        if 0.0 < v < 1.0 or (v >= 1.0 and e < 0) or (v <= 0.0 and e > 0):
            self.I += self.ki * e * self.dt                 # conditional integration (anti-windup)
        return quantise(self.kp * e + self.I)


class TimerFF:
    """Relay plus a schedule timer: run at u_ff from `lead` before each planned event to `tail` after."""
    name = "Timer+Relay"
    period = 1

    def __init__(self, lead=10 / 60, tail=10 / 60, u_ff=0.8, species=None):
        self.lead, self.tail, self.u_ff, self.species = lead, tail, u_ff, species

    def reset(self, shift, ctx):
        self.windows = [(e.t0 - self.lead, e.t1 + self.tail) for e in shift.plan]

    def act(self, i, t, y):
        u = DUTY_LEVELS[relay_level(y, self.species)]
        if any(a <= t < b for a, b in self.windows):
            u = max(u, self.u_ff)
        return u


class MPC:
    """Exposure-economic MPC with move blocking, solved exactly over U^Nu.

    forecast: 'learned' (schedule profile + AR residual), 'persistence' (feedback only),
              'oracle' (true future drive, true state).
    """
    period = 6                      # 60 s control interval at 10 s simulation steps

    def __init__(self, model_plant, forecaster=None, forecast="learned", settings=MPCSettings(),
                 fallback=False, W=None, name=None):
        self.s = settings
        self.disc = Discretised(model_plant, settings.dt, settings.duties)
        self.W = weight_tensor() if W is None else W
        self.prob = MPCProblem(self.disc, self.W, settings.bound, settings.M, settings.rho,
                               model_plant.fan.p_fan, settings.N, settings.Nu, model_plant.fan.p_exp,
                               blens=settings.blocks)
        # stationary costate at the reference duty: lam_p = (-M_p(u_ref)^T)^{-1} W_p
        Mref = self.disc.M0 + 0.8 * self.disc.M1
        self.Lam = np.stack([np.linalg.solve(-Mref[p].T, self.W[p]) for p in range(NP)])
        self.Minv_T = np.stack([np.linalg.inv(-Mref[p].T) for p in range(NP)])
        if settings.terminal_rollout:
            # Lambda_j = A(0.8)^T (o_{j+1} dt W + Lambda_{j+1}), Lambda_K = 0: exact cost-to-go weight of the base duty
            Ab, _ = self.disc.mats(0.8)
            K = int(round(T_END / settings.dt))
            tk = settings.dt * np.arange(K + 1)
            o = ((tk >= SHIFT[0]) & (tk < SHIFT[1])).astype(float)
            self.LamR = np.zeros((K + 1, NP, NZ))
            for j in range(K - 1, -1, -1):
                self.LamR[j] = np.einsum("pji,pj->pi", Ab, o[j + 1] * settings.dt * self.W + self.LamR[j + 1])
        self.cbz = settings.bound[:, None] * np.ones((1, NZ))
        self.p_exp = model_plant.fan.p_exp
        self.forecaster = forecaster
        self.mode = forecast
        self.fallback = fallback
        self.name = name or {"learned": "MPC-FF", "persistence": "MPC-FB", "oracle": "MPC-oracle"}[forecast]

    def reset(self, shift, ctx):
        self.shift = shift
        self.ctx = ctx
        self.buf = []
        self.ubuf = []
        self.iu = 0
        self.e_used = 0.0
        self.n_steps_total = int(round(shift.t[-1] / self.s.dt)) + 1
        self.k = 0
        self.od = OnlineDrive(self.disc, self.forecaster, shift.plan,
                              mode="persistence" if self.mode == "persistence" else "learned")
        self.alloc = Allocator(self.s.alloc, self.s.rho, self.s.e_shift, self.s.alloc_every, self.s.alloc_gain)
        self.prob.rho = self.s.rho
        self.fb_until = -1.0
        self.fb_count = 0
        self.fb_steps = 0
        self.alarm = 0
        self.log = []

    def act(self, i, t, y):
        self.buf.append(y)
        if i % self.period != 0:
            u = self.s.duties[self.iu] if not self._in_fallback(t) else self._relay(y)
            self._account(u)
            return u
        yavg = np.mean(self.buf[-self.period:], axis=0)
        if self.k == 0:
            self.od.init_steady(yavg, 0.0)
        else:
            self.od.update(t, yavg, float(np.mean(self.ubuf[-self.period:])))
        dt = self.s.dt
        N = self.s.N
        occ = presence(t + dt * np.arange(1, N + 1)) > 0
        if self.mode == "oracle":
            c0 = self.ctx["truth_state"]()
            D = self.ctx["true_drive"](t, N, dt)
        else:
            c0 = yavg
            D = self.od.forecast(t, N, dt)
        if self.s.alloc != "fixed":
            Krem = max(int(round((T_END - t) / dt)), 1)
            Drem = (self.ctx["true_drive"](t, Krem, dt) if self.mode == "oracle"
                    else self.od.forecast(t, Krem, dt))
            roll = lambda rhos: cf_rollout(self.disc, self.Minv_T, self.W, self.cbz, self.s.M,
                                           self.prob.p_fan, self.p_exp, c0, Drem, t, dt, rhos)
            self.prob.rho = self.alloc.update(t, T_END, self.e_used, roll)
        if self.s.use_budget_state:
            e_allow = max(self.s.e_shift - self.e_used, 0.0)
        else:
            e_allow = self.s.e_shift * N / self.n_steps_total
        if self.s.terminal_rollout:
            Wf = self.LamR[min(int(round(t / dt)) + N, len(self.LamR) - 1)]
        else:
            Wf = self.Lam * float(presence(t + dt * N) > 0) if self.s.terminal else None
        seq, J, _ = self.prob.solve_exhaustive(c0, D, occ.astype(float), e_allow, Wf)
        self.iu = int(seq[0])
        self.k += 1
        # supervisory fallback on the normalised particulate residual of the drive estimate
        if self.fallback and self.forecaster is not None and self.k > 5:
            z = np.abs(self.od.r[iPM25]) / self.forecaster.sd_r[iPM25]
            self.alarm = self.alarm + 1 if z.max() > 3.0 else 0
            if self.alarm >= 3 and not self._in_fallback(t):
                self.fb_until = t + 10 / 60
                self.fb_count += 1
        u = self.s.duties[self.iu]
        if self._in_fallback(t):
            self.fb_steps += 1
            u = self._relay(y)
            self.iu = int(np.argmin(np.abs(self.s.duties - u)))
        self._account(u)
        self.log.append((t, self.iu, J, e_allow))
        return u

    def _account(self, u):
        self.ubuf.append(u)
        self.e_used += self.prob.p_fan * u ** self.p_exp * self.s.dt / self.period

    def _in_fallback(self, t):
        return t < self.fb_until

    def _relay(self, y):
        return DUTY_LEVELS[relay_level(y)]


class CostateFeedback:
    """Pontryagin-guided band law: u = argmin_u rho*P(u) - u*B(x),
    B(x) = -sum_p lam_p^T M1_p x_p,  lam_p = (-M_p(u_ref)^T)^{-1} w_p(x)   (stationary costate),
    w_p(x) = occupancy-weighted index weights + soft-bound penalty gradient M/cb where x > cb.
    Same sampling (60 s, interval-averaged readings) as the MPC; no forecast."""
    period = 6

    def __init__(self, model_plant, settings=MPCSettings(), u_ref=0.8, W=None, name="Costate-FB",
                 forecaster=None):
        self.s = settings
        self.name = name
        self.forecaster = forecaster
        self.disc = Discretised(model_plant, settings.dt, settings.duties)
        self.M0, self.M1 = model_plant.rate_matrices()
        Mref = self.M0 + u_ref * self.M1
        self.Minv_T = np.stack([np.linalg.inv(-Mref[p].T) for p in range(NP)])
        self.W = weight_tensor() if W is None else W
        self.cb = settings.bound[:, None] * np.ones((1, NZ))
        self.p_fan = model_plant.fan.p_fan
        self.p_exp = model_plant.fan.p_exp

    def reset(self, shift, ctx):
        self.buf = []
        self.ubuf = []
        self.u = 0.0
        self.e_used = 0.0
        self.k = 0
        self.ctx = ctx
        self.rho = self.s.rho
        self.alloc = Allocator(self.s.alloc, self.s.rho, self.s.e_shift, self.s.alloc_every, self.s.alloc_gain)
        self.od = OnlineDrive(self.disc, self.forecaster, shift.plan,
                              mode="learned" if self.forecaster is not None else "persistence")

    def act(self, i, t, y):
        self.buf.append(y)
        if i % self.period != 0:
            return self._apply(self.u)
        x = np.mean(self.buf[-self.period:], axis=0)
        if self.k == 0:
            self.od.init_steady(x, 0.0)
        else:
            self.od.update(t, x, float(np.mean(self.ubuf[-self.period:])))
        self.k += 1
        if self.s.alloc != "fixed":
            dt = self.s.dt
            Krem = max(int(round((T_END - t) / dt)), 1)
            Drem = self.od.forecast(t, Krem, dt)
            roll = lambda rhos: cf_rollout(self.disc, self.Minv_T, self.W, self.cb, self.s.M,
                                           self.p_fan, self.p_exp, x, Drem, t, dt, rhos)
            self.rho = self.alloc.update(t, T_END, self.e_used, roll)
        occ = 1.0 if (SHIFT[0] - 0.5) <= t < SHIFT[1] else 0.0
        w = self.W * occ + self.s.M * (x > self.cb) / self.cb
        lam = np.einsum("pij,pj->pi", self.Minv_T, w)
        B = -np.einsum("pi,pij,pj->", lam, self.M1, x)
        cost = self.rho * self.p_fan * self.s.duties ** self.p_exp - self.s.duties * B
        # hard budget: only duties whose energy over the next interval fits the remaining budget
        rem = self.s.e_shift - self.e_used
        cost = np.where(self.p_fan * self.s.duties ** self.p_exp * self.s.dt <= rem + 1e-12, cost, np.inf)
        self.u = float(self.s.duties[int(np.argmin(cost))])
        return self._apply(self.u)

    def _apply(self, u):
        self.ubuf.append(u)
        self.e_used += self.p_fan * u ** self.p_exp * self.s.dt / self.period
        return u


class RelayBudget(Relay):
    """Deployed relay law with a naive budget guard: fans stop once the shift budget is spent."""
    name = "Relay+cutoff"

    def __init__(self, e_shift, p_fan=6.96, p_exp=3.0, dt=10 / 3600):
        super().__init__()
        self.e_shift, self.p_fan, self.p_exp, self.dt = e_shift, p_fan, p_exp, dt

    def reset(self, shift, ctx):
        self.e_used = 0.0

    def act(self, i, t, y):
        u = DUTY_LEVELS[relay_level(y)]
        if self.e_used + self.p_fan * u ** self.p_exp * self.dt > self.e_shift:
            u = 0.0
        self.e_used += self.p_fan * u ** self.p_exp * self.dt
        return u


class RelayOccupied(Relay):
    """Deployed relay law, enabled only from 30 min before to the end of the occupied shift."""
    name = "Relay-occ"

    def act(self, i, t, y):
        if not ((SHIFT[0] - 0.5) <= t < SHIFT[1]):
            return 0.0
        return DUTY_LEVELS[relay_level(y)]
