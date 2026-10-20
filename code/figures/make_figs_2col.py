"""Result figures for the two-column manuscript, drawn at printed size (see style2.py).
Output: figures2/*.pdf (vector, used by LaTeX) and *.png previews."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))
import numpy as np
import pandas as pd
from style2 import setup, panel, save, ENT, INK, INK2, SINGLE, DOUBLE, plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures2")
os.makedirs(FIG, exist_ok=True)
setup()


def fam(p):
    return p.split("|")[0].split("/")[0]


def mk(ent):
    c, m = ENT[ent]
    return dict(color=c, marker=m, mec="white", mew=0.4, ms=4.2 if m != "*" else 6.0)


# ------------------------------------------------------------------ closed-loop response of one shift
def fig_response(seed=1007):
    from aqmpc.params import Plant, MPCSettings, iPM25
    from aqmpc.scenarios import make_shift, ScenarioConfig, SHIFT
    from aqmpc.controllers import Relay, CostateFeedback, MPC
    from aqmpc.simulate import run_shift
    from aqmpc.harness import train_forecaster
    T = json.load(open(os.path.join(RES, "tuning.json")))
    pl = Plant()
    fc = train_forecaster(pl, workers=4)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    st = MPCSettings(rho=T["rho_iso"])
    runs = {}
    for name, ctrl in [("Relay", Relay()), ("Costate-FB", CostateFeedback(pl, st)),
                       ("MPC-FF", MPC(pl, fc, "learned", st))]:
        m, tr = run_shift(sh, ctrl, pl, sensor_seed=seed + 7919, return_traces=True)
        runs[name] = (m, tr)
    json.dump({k: v[0] for k, v in runs.items()}, open(os.path.join(RES, "e1", "response_example.json"), "w"), indent=1)
    t = runs["Relay"][1]["t"] + 7.0
    fig, axs = plt.subplots(3, 1, figsize=(DOUBLE, 4.3), sharex=True,
                            gridspec_kw={"height_ratios": [0.9, 2.0, 1.4], "hspace": 0.25})
    ax = axs[0]
    row = {"weld": 2, "solder": 1, "clean": 0}
    for e in sh.plan:
        ax.add_patch(plt.Rectangle((e.t0 + 7, row[e.kind] - 0.36), e.t1 - e.t0, 0.30, color="#b9b8b2", lw=0))
    for e in sh.actual:
        ax.add_patch(plt.Rectangle((e.t0 + 7, row[e.kind] + 0.04), e.t1 - e.t0, 0.30,
                                   color=INK if e.planned else "#e34948", lw=0))
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["clean (B)", "solder (A)", "weld (A)"])
    ax.set_ylim(-0.5, 2.5); ax.grid(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#b9b8b2"), plt.Rectangle((0, 0), 1, 1, color=INK),
                       plt.Rectangle((0, 0), 1, 1, color="#e34948")],
              labels=["planned", "actual", "unscheduled"], ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.32))
    panel(ax, "a", y=1.06)
    ax = axs[1]
    for name in ["Relay", "Costate-FB", "MPC-FF"]:
        c, _ = ENT[name]
        ax.plot(t, runs[name][1]["C"][:, iPM25, 0], color=c, lw=1.1, label=name)
    ax.set_ylabel("PM$_{2.5}$, zone A (µg m$^{-3}$)")
    ax.legend(loc="upper right", ncol=3)
    panel(ax, "b")
    ax = axs[2]
    for name, lw in [("Relay", 0.8), ("Costate-FB", 1.1), ("MPC-FF", 1.1)]:
        c, _ = ENT[name]
        ax.step(t, runs[name][1]["U"] * 100, where="post", color=c, lw=lw, label=name)
    ax.set_ylabel("Fan duty (%)"); ax.set_xlabel("Time of day (h)")
    ax.set_ylim(-5, 108); ax.set_yticks([0, 50, 80, 100])
    panel(ax, "c")
    for a in axs:
        a.axvspan(7, SHIFT[0] + 7, color="#f0efec", lw=0, zorder=0)
        a.axvspan(SHIFT[1] + 7, 17, color="#f0efec", lw=0, zorder=0)
        a.set_xlim(7, 17)
    save(fig, os.path.join(FIG, "fig_response.pdf"))


# ------------------------------------------------------------------ exposure-energy fronts
def _front(ax, df, family, label, lw=1.2):
    sub = df[df.policy.str.startswith(family + "|")]
    g = sub.groupby("policy")[["Efrac", "captured"]].mean().sort_values("Efrac")
    ax.plot(g.Efrac, g.captured, "-", lw=lw, label=label, zorder=3, **mk(family))


def _point(ax, df, pol, ent, label):
    sub = df[df.policy == pol]
    if not sub.empty:
        k = mk(ent); k["ms"] = 6
        ax.plot(sub.Efrac.mean(), sub.captured.mean(), ls="none", label=label, zorder=4, **k)


def fig_fronts():
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.75), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, sc, letter, title in [(axs[0], 1.0, "a", r"As-built purifier ($\bar Q\approx213$ m$^3$ h$^{-1}$)"),
                                  (axs[1], 4.0, "b", r"Purifier with $4\times$ flow (hypothetical)")]:
        df = pd.read_csv(os.path.join(RES, f"e2/fronts_scale{sc}.csv"))
        _front(ax, df, "PI", "PI (setpoint sweep)")
        _front(ax, df, "Costate-FB", "Costate feedback (no forecast)")
        _front(ax, df, "MPC-FB", "MPC, feedback only")
        _front(ax, df, "MPC-FF", "MPC, schedule feedforward")
        _front(ax, df, "MPC-oracle", "MPC, oracle feedforward", lw=0.8)
        _point(ax, df, "Relay", "Relay", "Relay (testbed firmware)")
        _point(ax, df, "Timer+Relay-PM", "Timer+Relay-PM", "Timer pre-activation + relay")
        for u in ["AO-80", "AO-90", "AO-100"]:
            sub = df[df.policy == u]
            k = mk("AO"); k["ms"] = 5.5
            ax.plot(sub.Efrac.mean(), sub.captured.mean(), ls="none", zorder=4,
                    label="Always-on 80/90/100 %" if u == "AO-80" else None, **k)
            ax.annotate(u.replace("AO-", "") + " %", (sub.Efrac.mean(), sub.captured.mean()),
                        textcoords="offset points", xytext=(3, -8), fontsize=7, color=INK2)
        ax.set_xlabel("Fan energy / energy at full duty")
        ax.set_title(title, color=INK)
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.04)
        panel(ax, letter)
    axs[0].set_ylabel("Share of achievable reduction $\\kappa$")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    save(fig, os.path.join(FIG, "fig_fronts.pdf"))


# ------------------------------------------------------------------ budget allocation
def fig_budget():
    df = pd.concat([pd.read_csv(os.path.join(RES, "e3/budget_main.csv")),
                    pd.read_csv(os.path.join(RES, "e3/budget_sweep.csv"))])
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    series = [("Relay+cutoff", "Relay", "-", "Relay + cut-off"),
              ("Costate-FB/fixed", "Costate-FB", ":", "Costate law, fixed $\\rho$"),
              ("MPC-FB/fixed", "MPC-FB", ":", "MPC-FB, fixed $\\rho$"),
              ("MPC-FB/rollout", "MPC-FB", "-", "MPC-FB, rollout (persistence)"),
              ("Costate-FB/rollout", "Costate-FB", "-", "Costate law, rollout (schedule)"),
              ("MPC-FF/rollout", "MPC-FF", "-", "MPC-FF, rollout (schedule)"),
              ("MPC-oracle/rollout", "MPC-oracle", "-", "MPC-oracle, rollout")]
    for pol, ent, ls, lab in series:
        g = df[df.policy == pol].groupby("budget_frac").captured.mean()
        ax.plot(g.index * 100, g.values, ls, lw=1.1, label=lab, **mk(ent))
    ax.set_xlabel("Shift energy budget (% of full-duty energy)")
    ax.set_ylabel("Share of achievable reduction $\\kappa$")
    ax.set_xticks([15, 25, 35, 45])
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.2), fontsize=6.8)
    save(fig, os.path.join(FIG, "fig_budget.pdf"))


# ------------------------------------------------------------------ robustness
def fig_robust():
    df = pd.read_csv(os.path.join(RES, "e4/robustness.csv"))
    cfgs = list(dict.fromkeys(df.config))
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 4.9), sharey=True, gridspec_kw={"wspace": 0.06})
    y = np.arange(len(cfgs))[::-1]
    for ax, pols, letter, title in [
            (axs[0], [("Relay", 0.2), ("Costate-FB", 0.07), ("MPC-FB", -0.07), ("MPC-FF", -0.2)], "a",
             r"Unconstrained energy ($\rho$ tuned to relay energy)"),
            (axs[1], [("Costate-FB/fixed", 0.15), ("MPC-FB/rollout", 0.0), ("MPC-FF/rollout", -0.15)], "b",
             "Shift budget 25 % of full-duty energy")]:
        for pol, off in pols:
            r = df[df.policy == pol].groupby("config").captured.mean().reindex(cfgs)
            ax.plot(r.values, y + off, ls="none", label=pol, **mk(fam(pol) if "/" in pol else pol))
        ax.set_xlabel("Share of achievable reduction $\\kappa$")
        ax.set_title(title, color=INK)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.085), ncol=4 if letter == "a" else 3)
        panel(ax, letter)
    labels = [c.replace("lam0", "$\\lambda_0$").replace("k_dep", "$k_p$").replace("eta_PM", "$\\eta_{PM}$")
               .replace("Q x", "$\\bar Q\\,\\times$").replace(" x", " $\\times$").replace("+-30%", "$\\pm$30 %")
               for c in cfgs]
    axs[0].set_yticks(y); axs[0].set_yticklabels(labels, fontsize=7.2)
    axs[0].set_ylim(-0.8, len(cfgs) - 0.2)
    save(fig, os.path.join(FIG, "fig_robust.pdf"))


# ------------------------------------------------------------------ weights
def fig_weights():
    df = pd.read_csv(os.path.join(RES, "e5/weights.csv"))
    nom = df[df.design == "nominal"]
    piv = nom.pivot_table(index=["weights", "seed"], columns="policy", values="CE")
    cap = {pol: ((piv["Off"] - piv[pol]) / (piv["Off"] - piv["AO-100"])).groupby("weights").mean()
           for pol in ["Relay", "AO-80", "Costate-FB", "MPC-FF"]}
    fig, ax = plt.subplots(figsize=(SINGLE, 2.9))
    # constant 80 % lies far below the others (kappa about 0.84); the legend sits below the axes so it hides no series
    for z, (pol, lab) in enumerate([("Costate-FB", "CL-s (stationary costate law)"), ("MPC-FF", "EMPC-FF"),
                                    ("AO-80", "Constant 80 %")]):
        ax.plot(cap["Relay"], cap[pol], ls="none", label=lab, zorder=3 + z, **mk("AO" if pol.startswith("AO") else pol))
    lo = min(v.min() for v in cap.values()) - 0.01; hi = max(v.max() for v in cap.values()) + 0.01
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.7, ls="--")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Relay: share of achievable reduction")
    ax.set_ylabel("Policy: share of achievable reduction")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)  # outside: hides no data
    save(fig, os.path.join(FIG, "fig_weights.pdf"))


# ------------------------------------------------------------------ disturbance forecast accuracy
def fig_forecast():
    s = pd.read_csv(os.path.join(RES, "e9/forecast_summary.csv"))
    d = s[s.channel == "PM2.5 zone A"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.7))
    for (cfg, fc), g in d.groupby(["config", "forecaster"]):
        c = "#2a78d6" if "inverted" in fc else "#4a3aa7"
        ls = "-" if cfg.startswith("nominal") else "--"
        g = g.sort_values("lead_min")
        ax.plot(g.lead_min, g.rmse_vs_true / g.true_sd, ls, color=c, marker="o", ms=3.4, lw=1.1)
        ax.plot(g.lead_min, g.rmse_vs_inv / g.true_sd, ls, color=c, marker="x", ms=3.8, lw=0.7, alpha=0.75)
    ax.set_xlabel("Forecast lead (min)")
    ax.set_ylabel("RMSE / s.d. of true drive")
    proxy = [Line2D([], [], color="#2a78d6", lw=1.3, label="trained on inverted drive"),
             Line2D([], [], color="#4a3aa7", lw=1.3, label="trained on true drive"),
             Line2D([], [], color=INK2, marker="o", ls="none", ms=3.4, label="scored vs true drive"),
             Line2D([], [], color=INK2, marker="x", ls="none", ms=3.8, label="scored vs inverted drive"),
             Line2D([], [], color=INK2, ls="-", label="nominal truth"),
             Line2D([], [], color=INK2, ls="--", label="mismatched truth")]
    ax.legend(handles=proxy, ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.2), fontsize=6.8)
    save(fig, os.path.join(FIG, "fig_forecast.pdf"))


# ------------------------------------------------------------------ computation
def fig_compute():
    df = pd.read_csv(os.path.join(RES, "e6/timing_scaling.csv"))
    names = {0: "Exhaustive enumeration", 1: "B&B, forward bound", 2: "B&B, affine bound"}
    cols = {0: ("#eb6834", "s"), 1: ("#1baf7a", "D"), 2: ("#2a78d6", "o")}
    fig, axs = plt.subplots(1, 4, figsize=(DOUBLE, 1.95), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (fac, lab) in zip(axs, [("nz", "zones $|\\mathcal{Z}|$"), ("np", "pollutants $|\\mathcal{P}|$"),
                                    ("Nu", "move blocks $N_u$"), ("nu", "duty levels $|\\mathcal{U}|$")]):
        d = df[df.factor == fac]
        for mode in [0, 1, 2]:
            g = d[d["mode"] == mode].groupby("value").time_us.median() / 1000
            c, m = cols[mode]
            ax.plot(g.index, g.values, "-", color=c, marker=m, ms=3.6, lw=1.1, mec="white", mew=0.4,
                    label=names[mode])
        ax.set_yscale("log"); ax.set_xlabel(lab)
    axs[0].set_ylabel("Median solve time (ms)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    save(fig, os.path.join(FIG, "fig_compute.pdf"))


if __name__ == "__main__":
    which = sys.argv[1:] or ["fronts", "budget", "robust", "weights", "forecast", "compute", "response"]
    for w in which:
        globals()[f"fig_{w}"]()
        print("figure", w, "done", flush=True)
