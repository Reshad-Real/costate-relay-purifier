"""Nominal parameters of the evaluation platform and of the simulated factory.

Every value carries a provenance tag so the manuscript can state exactly where it comes from:
  [FW]   deployed firmware (Mesh/CCU/CCU.ino, Mesh/DAG/DAG.ino)
  [DS]   component datasheet (Arctic P12 Max, ESP32)
  [REP]  FYDP-C final report (design calculations / single industrial visit)
  [LIT]  typical literature value (cited in the manuscript)
  [ASM]  modelling assumption, varied in the robustness study
Units: time in hours, PM in ug/m3, TVOC and NO2 in ppb, eCO2 and CO in ppm, volumes in m3,
flows in m3/h, power in W, energy in Wh.
"""
from dataclasses import dataclass, field, replace
import numpy as np

POLLUTANTS = ["PM2.5", "PM10", "TVOC", "eCO2", "CO", "NO2"]
ZONES = ["A", "B", "C"]
NP, NZ = len(POLLUTANTS), len(ZONES)
iPM25, iPM10, iTVOC, iCO2, iCO, iNO2 = range(NP)

# ---- relay law of the deployed CCU [FW] -------------------------------------------------
# calculateFanSpeed(): level = number of thresholds (LOW, MEDIUM, HIGH) reached by the
# worst pollutant; the DAG forwards the per-pollutant maximum over the nodes [FW].
RELAY_THRESHOLDS = np.array([          # columns: level-1 (80%), level-2 (90%), level-3 (100%)
    [35.0, 55.0, 150.0],               # PM2.5
    [100.0, 150.0, 250.0],             # PM10
    [660.0, 2200.0, 5500.0],           # TVOC
    [1000.0, 1500.0, 2000.0],          # eCO2
    [25.0, 50.0, 100.0],               # CO
    [100.0, 200.0, 400.0],             # NO2
])
DUTY_LEVELS = np.array([0.0, 0.8, 0.9, 1.0])     # FAN_OFF/LOW/MEDIUM/HIGH = 0/204/230/255 PWM [FW]

# ---- exposure index (manuscript Table 'params'; policy choices, varied in sensitivity) --
PHI = np.array([0.30, 0.20, 0.15, 0.05, 0.12, 0.18])   # pollutant weights (sum 1)
PSI = np.array([0.45, 0.35, 0.20])                       # zone (occupancy) weights (sum 1)
G_NORM = np.array([12.0, 50.0, 220.0, 800.0, 9.0, 53.0]) # normalisers g_p
C_BOUND2 = RELAY_THRESHOLDS[:, 1].copy()                 # level-2 threshold (90% duty) [FW]
C_BOUND3 = RELAY_THRESHOLDS[:, 2].copy()                 # level-3 threshold (100% duty) [FW]
C_BOUND = C_BOUND3                                       # operating bound used by the MPC


@dataclass
class FanModel:
    """Purifier fan/filter model.

    Operating flow at full speed is the intersection of the two-fan parallel curve
    (free-air 2 x 137.7 m3/h, shut-off 42.7 Pa [DS]) with a quadratic system curve through the
    report's design point (12.5 Pa at 243 m3/h [REP]); with a quadratic system curve the
    affinity laws make the operating flow proportional to speed, and the firmware sets speed
    proportional to duty [FW]. Power follows the cube law above an idle term; P_fan is the
    datasheet maximum (2 x 0.29 A x 12 V [DS]). Neither relation has been measured on the unit.
    """
    q_free: float = 2 * 137.7
    dp_shut: float = 42.7
    dp_design: float = 12.5
    q_design: float = 243.0
    p_fan: float = 6.96
    p_idle: float = 0.0          # electronics power is common to all controllers; excluded
    p_exp: float = 3.0           # power-law exponent (cube law = ideal affinity law) [ASM]

    @property
    def q_op(self):
        k = self.dp_design / self.q_design ** 2
        a = self.q_free * k / self.dp_shut          # Q = q_free (1 - k Q^2 / dp_shut)
        return (-1.0 + np.sqrt(1.0 + 4.0 * a * self.q_free)) / (2.0 * a)

    def power(self, u):
        u = np.asarray(u, dtype=float)
        return self.p_idle + self.p_fan * u ** self.p_exp


@dataclass
class Plant:
    """Three-zone well-mixed model of the 140 m2 floor with one central recirculating purifier."""
    area: np.ndarray = field(default_factory=lambda: np.array([55.0, 45.0, 40.0]))  # 140 m2 [REP]; split [ASM]
    height: float = 3.0                                    # [ASM] not recorded
    lam0: float = 1.0                                      # infiltration/ventilation, 1/h [ASM]
    k_dep: np.ndarray = field(default_factory=lambda: np.array([0.4, 1.5, 0.1, 0.0, 0.0, 0.5]))  # 1/h [LIT]
    eta: np.ndarray = field(default_factory=lambda: np.array([0.90, 0.95, 0.10, 0.0, 0.0, 0.05]))  # single pass [ASM]
    w_in: np.ndarray = field(default_factory=lambda: np.array([0.40, 0.35, 0.25]))  # purifier intake share [ASM]
    f_ab: float = 300.0                                    # interzonal exchange, m3/h [ASM]
    f_bc: float = 300.0
    f_ac: float = 150.0
    q_scale: float = 1.0                                   # multiplier on fan.q_op (filter loading, mismatch)
    fan: FanModel = field(default_factory=FanModel)

    @property
    def vol(self):
        return self.area * self.height

    @property
    def q_op(self):
        return self.fan.q_op * self.q_scale

    def exchange_matrix(self):
        F = np.zeros((NZ, NZ))
        F[0, 1] = F[1, 0] = self.f_ab
        F[1, 2] = F[2, 1] = self.f_bc
        F[0, 2] = F[2, 0] = self.f_ac
        return F

    def rate_matrices(self):
        """Return M0 (P,Z,Z) and M1 (P,Z,Z) with dc/dt = (M0 + u M1) c + drive."""
        V = self.vol
        F = self.exchange_matrix()
        L = (F - np.diag(F.sum(axis=1))) / V[:, None]          # interzonal mixing
        M0 = np.zeros((NP, NZ, NZ))
        M1 = np.zeros((NP, NZ, NZ))
        w = self.w_in / self.w_in.sum()
        Q = self.q_op
        for p in range(NP):
            M0[p] = L - np.eye(NZ) * (self.lam0 + self.k_dep[p])
            # purifier draws Q*w_j from zone j, returns (1-eta) of the mixed intake by the same shares
            M1[p] = (Q / V)[:, None] * (w[:, None] * ((1.0 - self.eta[p]) * w[None, :]) - np.diag(w))
        return M0, M1


@dataclass
class Background:
    """Outdoor levels entering by infiltration (sampled per shift in scenarios.py)."""
    c_out: np.ndarray = field(default_factory=lambda: np.array([40.0, 75.0, 60.0, 420.0, 0.5, 25.0]))


# Nominal control settings (manuscript Table 'params')
@dataclass
class MPCSettings:
    dt: float = 1.0 / 60.0         # control interval 60 s
    N: int = 30                    # prediction horizon (steps)
    Nu: int = 6                    # number of move blocks
    rho: float = 0.04              # economic weight on fan energy, (index h) per Wh
    M: float = 100.0               # soft-bound penalty per unit normalised exceedance-hour
    e_shift: float = 1e9           # shift fan-energy budget, Wh (inactive unless set)
    use_budget_state: bool = True  # remaining-budget formulation (False -> fixed per-horizon cap)
    terminal: bool = True          # costate-based terminal cost (cost-to-go beyond the horizon)
    terminal_rollout: bool = False # terminal weight = time-varying cost-to-go of constant 80 % duty to the end of the
                                   # day (the rollout costate Lambda_{k+N}) instead of the stationary costate
    blocks: tuple = ()            # explicit move-block lengths (sum = N); () -> Nu equal blocks
    alloc: str = "fixed"           # budget allocation: fixed | pacing | rollout (budget.py)
    alloc_every: int = 15          # re-allocation period, control steps
    alloc_gain: float = 1.5        # pacing gain on log rho per unit of (spent - elapsed) share
    bound: np.ndarray = field(default_factory=lambda: C_BOUND.copy())
    duties: np.ndarray = field(default_factory=lambda: DUTY_LEVELS.copy())


def nominal_plant():
    return Plant()


def scaled_plant(base: Plant, **factors):
    """Return a copy of `base` with selected parameters multiplied by the given factors."""
    kw = {}
    for name, fct in factors.items():
        val = getattr(base, name)
        kw[name] = val * fct
    return replace(base, **kw)
