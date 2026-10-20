"""Low-cost sensor chain: first-order lag, residual calibration gain error, noise, transport delay.

eCO2 of the ENS160/CCS811 is a VOC-derived equivalent, so the reading responds to alcohol
vapour; the report records eCO2 rising together with TVOC during the alcohol bath [REP].
"""
from dataclasses import dataclass
import numpy as np
from .params import NP, NZ, iTVOC, iCO2

TAU = np.array([0.5, 0.5, 1.5, 1.5, 1.0, 1.0]) / 60.0       # h, sensor time constants [ASM]
NOISE_ABS = np.array([1.5, 3.0, 10.0, 15.0, 0.1, 3.0])     # floor of noise sd per reading [ASM]
NOISE_REL = np.array([0.04, 0.05, 0.08, 0.05, 0.05, 0.06])  # relative noise sd [ASM]


@dataclass
class SensorConfig:
    gain_sd: float = 0.05        # residual calibration gain error (per channel and shift)
    lag_scale: float = 1.0
    noise_scale: float = 1.0
    delay_steps: int = 2         # 20 s detection + mesh transport at 10 s steps [REP: 4+12 s]
    eco2_xsens: float = 0.25     # ppm eCO2 per ppb TVOC above 100 ppb [ASM]
    drift_per_h: float = 0.0     # relative gain drift per hour (calibration drift)
    drop_p: float = 0.0          # probability that a reading is lost in the mesh (last value held)
    stuck_zone: int = -1         # zone whose node freezes at its first reading (-1: none)


class SensorChain:
    def __init__(self, rng, dt, cfg=SensorConfig()):
        self.rng = rng
        self.cfg = cfg
        self.a = np.exp(-dt / (TAU * cfg.lag_scale))[:, None]
        self.gain = 1.0 + rng.normal(0, cfg.gain_sd, (NP, NZ))
        self.state = None
        self.buf = []
        self.dt = dt
        self.n = 0
        self.last = None
        self.first = None

    def reset(self, c0):
        self.state = c0.copy()
        self.buf = [self._read(self.state) for _ in range(self.cfg.delay_steps + 1)]

    def _read(self, x):
        y = self.gain * x if self.cfg.drift_per_h == 0.0 else self.gain * (1.0 + self.cfg.drift_per_h * self.n * self.dt) * x
        y[iCO2] = y[iCO2] + self.cfg.eco2_xsens * np.maximum(x[iTVOC] - 100.0, 0.0)
        sd = self.cfg.noise_scale * np.maximum(NOISE_ABS[:, None], NOISE_REL[:, None] * np.abs(y))
        return np.maximum(y + self.rng.normal(0, 1, y.shape) * sd, 0.0)

    def step(self, c_true):
        self.state = self.a * self.state + (1 - self.a) * c_true
        self.n += 1
        self.buf.append(self._read(self.state))
        y = self.buf.pop(0)
        if self.cfg.drop_p > 0.0:
            if self.last is not None and self.rng.random() < self.cfg.drop_p:
                y = self.last
            self.last = y
        if self.cfg.stuck_zone >= 0:
            if self.first is None:
                self.first = y[:, self.cfg.stuck_zone].copy()
            y = y.copy(); y[:, self.cfg.stuck_zone] = self.first
        return y
