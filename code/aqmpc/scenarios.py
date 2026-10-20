"""Stochastic shift scenarios for a light-electronics floor (welding, soldering, alcohol cleaning).

A scenario has two layers:
  * the production PLAN (event type, zone, planned start/end) - what the forecaster may use;
  * the REALISATION - actual onset (planned + jitter), duration, amplitude, cancellations,
    unscheduled events, outdoor levels and occupancy - what the truth plant sees.
All emission magnitudes are assumptions [ASM] chosen so that zone-A PM2.5 peaks under the
relay law fall in the 60-150 ug/m3 range recorded during the industrial visit [REP].
Outdoor PM2.5 is MEASURED by default: every simulated day is mapped to a distinct real working day of the U.S. Embassy
Dhaka monitor (hourly readings, 2021-10 to 2024-01; data/outdoor_dhaka_*.csv, built by data_prep/prep_outdoor.py and
checked against the EPA's official daily means of the same monitor). Outdoor PM10 = measured PM2.5 x a drawn ratio.
"""
import os
import functools
from dataclasses import dataclass, field
import numpy as np
from .params import NP, NZ, iPM25, iPM10, iTVOC, iCO2, iCO, iNO2

T_END = 10.0          # simulated window 07:00-17:00 (h)
SHIFT = (1.0, 9.0)    # occupied shift 08:00-16:00
LUNCH = (5.0, 6.0)    # 12:00-13:00
DT_SIM = 10.0 / 3600.0

# emission rate per event type at amplitude 1: units*m3/h, columns = zones A,B,C
_rate = {}
_r = np.zeros((NP, NZ)); _r[iPM25, 0] = 45e3; _r[iPM10, 0] = 65e3; _r[iNO2, 0] = 40e3; _r[iCO, 0] = 2.5e3
_r[iTVOC, 0] = 40e3
_rate["weld"] = _r
_r = np.zeros((NP, NZ)); _r[iPM25, 0] = 18e3; _r[iPM10, 0] = 20e3; _r[iTVOC, 0] = 120e3; _r[iNO2, 0] = 4e3
_rate["solder"] = _r
_r = np.zeros((NP, NZ)); _r[iTVOC, 1] = 900e3; _r[iPM25, 1] = 3e3; _r[iPM10, 1] = 4e3
_rate["clean"] = _r
EVENT_TYPES = ["weld", "solder", "clean"]
EVENT_ZONE = {"weld": 0, "solder": 0, "clean": 1}

OCC_CO2 = 18e3                                  # ppm*m3/h per worker (0.3 L/min CO2) [LIT]
OCCUPANTS = np.array([4.0, 3.0, 1.0])           # workers per zone [ASM]
ACTIVITY = np.zeros((NP, NZ)); ACTIVITY[iPM25] = 3e3; ACTIVITY[iPM10] = 9e3   # resuspension [ASM]


@dataclass
class ScenarioConfig:
    onset_sd: float = 5.0 / 60       # h, sd of actual-minus-planned onset
    dur_sd: float = 0.15             # relative sd of duration
    amp_sd: float = 0.30             # log-sd of event amplitude
    cancel_p: float = 0.05           # probability a planned event does not occur
    unsched_rate: float = 1.0        # unscheduled events per shift (Poisson)
    amp_scale: float = 1.0           # global emission scale
    pm_out_range: tuple = (15.0, 90.0)
    ramp: float = 2.0 / 60           # h, trapezoid ramp
    onset_bias: float = 0.0          # h, systematic lateness of every planned event (plan bias)
    outdoor: str = "dhaka"           # "dhaka": measured outdoor PM2.5 of a real working day; "synthetic": pm_out_range
    pm_out_scale: float = 1.0        # multiplies the measured outdoor PM2.5 (robustness studies)


@dataclass
class Event:
    kind: str
    zone: int
    t0: float
    t1: float
    amp: float = 1.0
    planned: bool = True


@dataclass
class Shift:
    t: np.ndarray
    S: np.ndarray            # (n, P, Z) true source
    c_out: np.ndarray        # (n, P) outdoor level
    occ: np.ndarray          # (n,) 1 during the occupied shift
    plan: list = field(default_factory=list)
    actual: list = field(default_factory=list)
    date: str = ""           # real calendar day of the outdoor record (outdoor = "dhaka")


_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")


@functools.lru_cache(maxsize=1)
def _outdoor_pool():
    """Measured hourly PM2.5 (local time) and a fixed season-mixing permutation of the usable working days."""
    import pandas as pd
    h = pd.read_csv(os.path.join(_DATA, "outdoor_dhaka_pm25.csv"), parse_dates=[0], index_col=0)["pm25"]
    days = pd.read_csv(os.path.join(_DATA, "outdoor_dhaka_days.csv"), parse_dates=["date"]).date.tolist()
    perm = np.random.default_rng(20261012).permutation(len(days))
    return h, [days[i] for i in perm]


def _day_slot(seed, n):
    """Disjoint days for tuning (100-139), test (1000-1199) and training/auxiliary (0-99) seeds; others reuse the rest."""
    if 100 <= seed < 140:
        return seed - 100
    if 1000 <= seed < 1200:
        return 40 + seed - 1000
    if 0 <= seed < 100:
        return 240 + seed
    return 340 + seed % (n - 340)


def outdoor_trace(seed, t):
    """Measured outdoor PM2.5 on the 07:00-17:00 window of the real day mapped to `seed`, at times t (h after 07:00)."""
    h, days = _outdoor_pool()
    d = days[_day_slot(int(seed), len(days))]
    w = h[d + np.timedelta64(6, "h"): d + np.timedelta64(18, "h")]          # 06:00-18:00 for interpolation
    th = (w.index - (d + np.timedelta64(7, "h"))) / np.timedelta64(1, "h") + 0.5   # hour-centred values
    v = w.values.astype(float)
    ok = np.isfinite(v)
    return np.interp(t, th[ok], v[ok]), str(d.date())


def _plan(rng):
    """Draw a production plan on a 15-min grid without overlap within a zone."""
    plan = []
    busy = {0: [], 1: []}
    counts = {"weld": rng.integers(2, 5), "solder": rng.integers(1, 4), "clean": rng.integers(2, 4)}
    durs = {"weld": (30, 60), "solder": (20, 45), "clean": (15, 30)}
    for kind in EVENT_TYPES:
        z = EVENT_ZONE[kind]
        placed, tries = 0, 0
        while placed < counts[kind] and tries < 200:
            tries += 1
            d = rng.integers(durs[kind][0] // 5, durs[kind][1] // 5 + 1) * 5 / 60
            t0 = SHIFT[0] + 0.25 + rng.integers(0, int((SHIFT[1] - SHIFT[0] - 0.5 - d) * 4) + 1) / 4
            t1 = t0 + d
            if t0 < LUNCH[1] and t1 > LUNCH[0]:
                continue
            if any(t0 < b1 + 1 / 3 and t1 > b0 - 1 / 3 for b0, b1 in busy[z]):
                continue
            busy[z].append((t0, t1))
            plan.append(Event(kind, z, t0, t1))
            placed += 1
    plan.sort(key=lambda e: e.t0)
    return plan


def _trap(t, t0, t1, ramp):
    up = np.clip((t - t0) / ramp, 0, 1)
    dn = np.clip((t1 - t) / ramp, 0, 1)
    return np.minimum(up, dn) * ((t > t0) & (t < t1))


def make_shift(rng, cfg=ScenarioConfig(), plan=None):
    t = np.arange(0.0, T_END, DT_SIM)
    n = len(t)
    plan = _plan(rng) if plan is None else plan
    actual = []
    for e in plan:
        if rng.random() < cfg.cancel_p:
            continue
        jit = np.clip(rng.normal(0, cfg.onset_sd), -4 * cfg.onset_sd - 1e-9, 4 * cfg.onset_sd + 1e-9) if cfg.onset_sd > 0 else 0.0
        dur = (e.t1 - e.t0) * max(0.3, 1 + rng.normal(0, cfg.dur_sd))
        amp = np.exp(rng.normal(0, cfg.amp_sd) - 0.5 * cfg.amp_sd ** 2)
        t0a = e.t0 + jit + cfg.onset_bias
        actual.append(Event(e.kind, e.zone, t0a, t0a + dur, amp, True))
    for _ in range(rng.poisson(cfg.unsched_rate)):
        kind = EVENT_TYPES[rng.integers(0, 3)]
        t0 = rng.uniform(SHIFT[0], SHIFT[1] - 0.3)
        dur = rng.uniform(10, 30) / 60
        amp = np.exp(rng.normal(0, 0.4) - 0.08)
        actual.append(Event(kind, EVENT_ZONE[kind], t0, t0 + dur, amp, False))

    S = np.zeros((n, NP, NZ))
    for e in actual:
        S += cfg.amp_scale * e.amp * _trap(t, e.t0, e.t1, cfg.ramp)[:, None, None] * _rate[e.kind][None]
    occ = ((t >= SHIFT[0]) & (t < SHIFT[1])).astype(float)
    presence = occ * np.where((t >= LUNCH[0]) & (t < LUNCH[1]), 0.2, 1.0)
    S[:, iCO2, :] += presence[:, None] * OCCUPANTS[None, :] * OCC_CO2
    S += presence[:, None, None] * ACTIVITY[None] * cfg.amp_scale

    lo, hi = cfg.pm_out_range
    pm = np.exp(rng.uniform(np.log(lo), np.log(hi)))
    base = np.array([pm, pm * rng.uniform(1.5, 2.2), rng.uniform(50, 150), 420.0,
                     rng.uniform(0.3, 1.0), rng.uniform(15, 45)])
    phase = rng.uniform(0, 2 * np.pi)
    drift = 1 + 0.2 * np.sin(2 * np.pi * t / 24 + phase)
    c_out = base[None, :] * drift[:, None]
    c_out[:, iCO2] = 420.0
    date = ""
    if cfg.outdoor == "dhaka":                     # measured outdoor PM2.5 (all random draws above are kept)
        seed = rng.bit_generator.seed_seq.entropy
        pm25, date = outdoor_trace(seed, t)
        ratio = base[iPM10] / base[iPM25]
        c_out[:, iPM25] = cfg.pm_out_scale * pm25
        c_out[:, iPM10] = cfg.pm_out_scale * pm25 * ratio
    return Shift(t, S, c_out, occ, plan, actual, date)
