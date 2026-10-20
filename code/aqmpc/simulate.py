"""Closed-loop simulation of one shift."""
import numpy as np
from .params import NP, NZ
from .plant import TruthSim
from .sensors import SensorChain, SensorConfig
from .scenarios import DT_SIM
from .exposure import metrics


def run_shift(shift, controller, truth_plant, sensor_seed, sensor_cfg=SensorConfig(),
              return_traces=False, metric_kw=None):
    sim = TruthSim(truth_plant, DT_SIM)
    n = len(shift.t)
    drive = sim.plant.lam0 * shift.c_out[:, :, None] + shift.S / sim.vol[None, None, :]
    c = sim.steady_state(0.0, drive[0])
    sens = SensorChain(np.random.default_rng(sensor_seed), DT_SIM, sensor_cfg)
    sens.reset(c)
    state = {"c": c}

    def true_drive(t, N, dt):
        # interval-averaged true drive over the next N control intervals
        i0 = int(round(t / DT_SIM))
        m = int(round(dt / DT_SIM))
        out = np.zeros((N, NP, NZ))
        for k in range(N):
            a, b = i0 + k * m, min(i0 + (k + 1) * m, n)
            out[k] = drive[min(a, n - 1):max(b, a + 1)].mean(axis=0) if a < n else drive[-1]
        return out

    ctx = {"truth_state": lambda: state["c"].copy(), "true_drive": true_drive}
    controller.reset(shift, ctx)
    C = np.zeros((n, NP, NZ))
    Y = np.zeros((n, NP, NZ)) if return_traces else None
    U = np.zeros(n)
    for i in range(n):
        y = sens.step(c)
        u = controller.act(i, shift.t[i], y)
        C[i] = c
        U[i] = u
        if return_traces:
            Y[i] = y
        c = sim.step(c, u, drive[i])
        state["c"] = c
    m = metrics(C, U, shift.occ, DT_SIM, truth_plant.fan, **(metric_kw or {}))
    if return_traces:
        return m, {"t": shift.t, "C": C, "Y": Y, "U": U, "S": shift.S, "drive": drive}
    return m
