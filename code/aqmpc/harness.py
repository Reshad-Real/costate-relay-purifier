"""Experiment harness: reproducible scenarios, forecaster training and parallel evaluation."""
import os
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .params import Plant, MPCSettings, NP, NZ
from .scenarios import make_shift, ScenarioConfig, DT_SIM
from .sensors import SensorConfig
from .controllers import Relay, AlwaysOn, PI, TimerFF, MPC, CostateFeedback, RelayBudget, RelayOccupied
from .extra_controllers import (HysteresisThreshold, TrackingMPC, RelaxRoundMPC, ScheduledDuty, PIGated,
                                RolloutCostate)
from .simulate import run_shift
from .plant import Discretised
from .forecast import ScheduleForecaster, invert_drive
from .params import iPM25, iPM10

PM_ONLY = [iPM25, iPM10]
TRAIN_SEEDS = list(range(10))          # ten relay-operated training shifts (two working weeks)
TEST_SEEDS = list(range(1000, 1200))   # independent test shifts


def build_controller(spec, model_plant, forecaster):
    kind, kw = spec
    kw = dict(kw)
    if kind == "Off":
        c = AlwaysOn(0.0); c.name = "Off"; return c
    if kind == "Relay":
        return Relay()
    if kind == "RelayBudget":
        return RelayBudget(kw["e_shift"], p_fan=model_plant.fan.p_fan, p_exp=model_plant.fan.p_exp)
    if kind == "Relay-occ":
        return RelayOccupied()
    if kind == "Relay-PM":
        return Relay(species=PM_ONLY, name="Relay-PM")
    if kind == "AO":
        return AlwaysOn(kw["u"])
    if kind == "PI":
        return PI(**kw)
    if kind == "Timer":
        c = TimerFF(species=PM_ONLY, **kw); c.name = "Timer+Relay-PM"; return c
    if kind == "CF":
        use_fc = kw.pop("use_forecaster", False)
        return CostateFeedback(model_plant, forecaster=forecaster if use_fc else None, **kw)
    if kind == "MPC":
        settings = kw.pop("settings", MPCSettings())
        return MPC(model_plant, forecaster if kw.get("forecast", "learned") != "persistence" else None,
                   settings=settings, **kw)
    if kind == "THC":
        return HysteresisThreshold(**kw)
    if kind == "MPC-track":
        settings = kw.pop("settings", MPCSettings())
        return TrackingMPC(model_plant, forecaster, settings=settings, **kw)
    if kind == "MPC-RR":
        settings = kw.pop("settings", MPCSettings())
        return RelaxRoundMPC(model_plant, forecaster, settings=settings, **kw)
    if kind == "Sched":
        return ScheduledDuty(**kw)
    if kind == "PI-SD":
        return PIGated(**kw)
    if kind == "RCL":
        settings = kw.pop("settings", MPCSettings())
        return RolloutCostate(model_plant, settings=settings, **kw)
    raise ValueError(kind)


def _job(a):
    spec, seed, scfg, truth, model, sens_cfg, fc, metric_kw = a
    shift = make_shift(np.random.default_rng(seed), scfg)
    ctrl = build_controller(spec, model, fc)
    m = run_shift(shift, ctrl, truth, sensor_seed=seed + 7919, sensor_cfg=sens_cfg, metric_kw=metric_kw)
    m["seed"] = seed
    m["policy"] = ctrl.name if spec[0] != "MPC" or "name" not in spec[1] else spec[1]["name"]
    if hasattr(ctrl, "fb_steps"):
        m["fallback_frac"] = ctrl.fb_steps / max(len(shift.t), 1)
    return m


def evaluate(specs, seeds, scfg=ScenarioConfig(), truth=None, model=None, sens_cfg=SensorConfig(),
             forecaster=None, metric_kw=None, workers=None):
    truth = truth or Plant()
    model = model or truth
    jobs = [(s, sd, scfg, truth, model, sens_cfg, forecaster, metric_kw) for s in specs for sd in seeds]
    workers = workers or int(os.environ.get("AQ_WORKERS", max(1, (os.cpu_count() or 2) - 2)))
    if workers == 1:
        return [_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_job, jobs, chunksize=1))


def _collect(a):
    """Run one training shift under the deployed relay and return the control-step record."""
    seed, scfg, truth, model, sens_cfg, use_true = a
    shift = make_shift(np.random.default_rng(seed), scfg)
    m, tr = run_shift(shift, Relay(), truth, sensor_seed=seed + 7919, sensor_cfg=sens_cfg, return_traces=True)
    st = int(round((1 / 60) / DT_SIM))
    K = len(shift.t) // st
    Y = tr["Y"][: K * st].reshape(K, st, NP, NZ).mean(axis=1)
    Ub = tr["U"][: K * st].reshape(K, st)
    disc = Discretised(model, 1 / 60, MPCSettings().duties)
    Ubar = Ub.mean(axis=1)
    t = shift.t[::st][:K]
    if use_true:
        D = tr["drive"][: K * st].reshape(K, st, NP, NZ).mean(axis=1)[:-1]
        # express the true drive in the model's coordinates (identical when model == truth)
    else:
        D = invert_drive(disc, Y, Ubar)
    return (t[:-1], shift.plan, D)


def train_forecaster(truth=None, model=None, scfg=ScenarioConfig(), sens_cfg=SensorConfig(),
                     seeds=TRAIN_SEEDS, use_true=False, workers=None):
    truth = truth or Plant()
    model = model or truth
    jobs = [(sd, scfg, truth, model, sens_cfg, use_true) for sd in seeds]
    workers = workers or min(len(jobs), int(os.environ.get("AQ_WORKERS", max(1, (os.cpu_count() or 2) - 2))))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(_collect, jobs))
    return ScheduleForecaster().fit(recs)


def objective(rows, rho):
    return float(np.mean([r["CE"] + rho * r["energy_Wh"] for r in rows]))
