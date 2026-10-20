"""Generate all result figures from results/ (or results_quick/ with QUICK=1)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))
import numpy as np
import pandas as pd
from style import setup, panel, save, ENT, INK, INK2, plt

QUICK = os.environ.get("QUICK") == "1"
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results_quick" if QUICK else "results")
FIG = os.path.join(ROOT, "figures_quick" if QUICK else "figures")
os.makedirs(FIG, exist_ok=True)
setup()


def fam(p):
    return p.split("|")[0].split("/")[0]


def front(ax, df, family, label=None, lw=1.6):
    sub = df[df.policy.str.startswith(family + "|")]
    if sub.empty:
        return
    g = sub.groupby("policy")[["Efrac", "captured"]].mean().sort_values("Efrac")
    c, m = ENT[family]
    ax.plot(g.Efrac, g.captured, "-", color=c, marker=m, ms=4.5 if m != "*" else 7, lw=lw,
            label=label or family, mec="white", mew=0.6, zorder=3)


def point(ax, df, pol, ent, label, ms=7):
    sub = df[df.policy == pol]
    if sub.empty:
        return
    c, m = ENT[ent]
    ax.plot(sub.Efrac.mean(), sub.captured.mean(), m, color=c, ms=ms, mec="white", mew=0.7, label=label, zorder=4)


# ------------------------------------------------------------------ Fig: fronts
def fig_fronts():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, sc, letter, title in [(axs[0], 1.0, "a", "As-built purifier ($Q\\approx213$ m$^3$/h)"),
                                  (axs[1], 4.0, "b", "Purifier with 4$\\times$ airflow (hypothetical)")]:
        p = os.path.join(RES, f"e2/fronts_scale{sc}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        front(ax, df, "PI", "PI (setpoint sweep)")
        front(ax, df, "Costate-FB", "Costate feedback (no forecast)")
        front(ax, df, "MPC-FB", "MPC, feedback only")
        front(ax, df, "MPC-FF", "MPC, schedule forecast")
        front(ax, df, "MPC-oracle", "MPC, oracle forecast", lw=1.0)
        point(ax, df, "Relay", "Relay", "Relay (deployed firmware)")
        point(ax, df, "Timer+Relay-PM", "Timer+Relay-PM", "Timer pre-activation + relay")
        for u in ["AO-80", "AO-90", "AO-100"]:
            sub = df[df.policy == u]
            if not sub.empty:
                c, m = ENT["AO"]
                ax.plot(sub.Efrac.mean(), sub.captured.mean(), m, color=c, ms=6, mec="white", mew=0.7,
                        label="Always-on (80/90/100 %)" if u == "AO-80" else None, zorder=4)
                ax.annotate(u.replace("AO-", "") + "%", (sub.Efrac.mean(), sub.captured.mean()),
                            textcoords="offset points", xytext=(4, -9), fontsize=7, color=INK2)
        ax.set_xlabel("Fan energy / energy at continuous full duty")
        ax.set_title(title, color=INK)
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
        panel(ax, letter)
    axs[0].set_ylabel("Share of achievable exposure reduction")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    save(fig, os.path.join(FIG, "fig_fronts.png"))


# ------------------------------------------------------------------ Fig: example traces
def fig_traces(seed=1007):
    from aqmpc.params import Plant, MPCSettings, iPM25
    from aqmpc.scenarios import make_shift, ScenarioConfig, SHIFT
    from aqmpc.controllers import Relay, CostateFeedback, MPC
    from aqmpc.simulate import run_shift
    from aqmpc.harness import train_forecaster
    T = json.load(open(os.path.join(RES, "tuning.json")))
    pl = Plant()
    fc = train_forecaster(pl)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    st = MPCSettings(rho=T["rho_iso"])
    runs = {}
    for name, ctrl in [("Relay", Relay()), ("Costate-FB", CostateFeedback(pl, st)),
                       ("MPC-FF", MPC(pl, fc, "learned", st))]:
        m, tr = run_shift(sh, ctrl, pl, sensor_seed=seed + 7919, return_traces=True)
        runs[name] = (m, tr)
    t = runs["Relay"][1]["t"] + 7.0
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 2.2, 1.6]})
    ax = axs[0]
    for e in sh.plan:
        ax.add_patch(plt.Rectangle((e.t0 + 7, {"weld": 2, "solder": 1, "clean": 0}[e.kind] - 0.35), e.t1 - e.t0, 0.3,
                                   color="#b9b8b2", lw=0))
    for e in sh.actual:
        ax.add_patch(plt.Rectangle((e.t0 + 7, {"weld": 2, "solder": 1, "clean": 0}[e.kind] + 0.02), e.t1 - e.t0, 0.3,
                                   color=INK if e.planned else "#e34948", lw=0))
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["clean (B)", "solder (A)", "weld (A)"])
    ax.set_ylim(-0.5, 2.5); ax.grid(False)
    ax.set_title("Planned (grey, lower) and actual (black, upper; red = unscheduled) emission events", color=INK)
    panel(ax, "a")
    ax = axs[1]
    for name in ["Relay", "Costate-FB", "MPC-FF"]:
        c, _ = ENT[name]
        ax.plot(t, runs[name][1]["C"][:, iPM25, 0], color=c, lw=1.2, label=name)
    ax.set_ylabel("PM$_{2.5}$, zone A (µg/m$^3$)")
    ax.legend(loc="upper right", ncol=3)
    panel(ax, "b")
    ax = axs[2]
    for k, name in enumerate(["Relay", "Costate-FB", "MPC-FF"]):
        c, _ = ENT[name]
        ax.step(t, runs[name][1]["U"] * 100 + k * 0.0, where="post", color=c, lw=1.0, label=name)
    ax.set_ylabel("Fan duty (%)"); ax.set_xlabel("Time of day (h)")
    ax.set_ylim(-5, 108)
    for a in axs:
        a.axvspan(7, SHIFT[0] + 7, color="#f0efec", lw=0, zorder=0)
        a.axvspan(SHIFT[1] + 7, 17, color="#f0efec", lw=0, zorder=0)
    panel(ax, "c")
    txt = "; ".join(f"{n}: CE {runs[n][0]['CE']:.2f}, {runs[n][0]['energy_Wh']:.1f} Wh, {runs[n][0]['switches']} switches"
                    for n in runs)
    fig.text(0.5, -0.01, txt, ha="center", fontsize=7, color=INK2)
    save(fig, os.path.join(FIG, "fig_traces.png"))


# ------------------------------------------------------------------ Fig: budget allocation
def fig_budget():
    p1, p2 = os.path.join(RES, "e3/budget_main.csv"), os.path.join(RES, "e3/budget_sweep.csv")
    if not os.path.exists(p1):
        return
    df = pd.concat([pd.read_csv(p1), pd.read_csv(p2)]) if os.path.exists(p2) else pd.read_csv(p1)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axs[0]
    series = [("Relay+cutoff", "Relay", "-", "Relay + cut-off"),
              ("Costate-FB/fixed", "Costate-FB", ":", "Costate law, fixed $\\rho$"),
              ("MPC-FB/pacing", "MPC-FB", "--", "MPC-FB, pacing"),
              ("MPC-FB/rollout", "MPC-FB", "-", "MPC-FB, rollout (persistence)"),
              ("Costate-FB/rollout", "Costate-FB", "-", "Costate law, rollout (schedule)"),
              ("MPC-FF/rollout", "MPC-FF", "-", "MPC-FF, rollout (schedule)"),
              ("MPC-oracle/rollout", "MPC-oracle", "-", "MPC-oracle, rollout")]
    for pol, ent, ls, lab in series:
        sub = df[df.policy == pol]
        if sub.empty:
            continue
        g = sub.groupby("budget_frac").captured.mean()
        c, m = ENT[ent]
        ax.plot(g.index, g.values, ls, color=c, marker=m, ms=4.5 if m != "*" else 7, lw=1.3, label=lab,
                mec="white", mew=0.5)
    ax.set_xlabel("Shift energy budget / full-duty energy")
    ax.set_ylabel("Share of achievable exposure reduction")
    ax.legend(ncol=2, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    panel(ax, "a")
    ax = axs[1]
    d = df[df.budget_frac == 0.25]
    order = ["Relay+cutoff", "Costate-FB/fixed", "Costate-FB/pacing", "MPC-FB/fixed", "MPC-FB/pacing",
             "MPC-FB/rollout", "MPC-FF/fixed", "MPC-FF/pacing", "Costate-FB/rollout", "MPC-FF/rollout",
             "MPC-oracle/rollout"]
    order = [o for o in order if o in set(d.policy)]
    for i, pol in enumerate(order):
        s = d[d.policy == pol].captured
        ent = "Relay" if pol.startswith("Relay") else fam(pol)
        c, m = ENT[ent]
        lo, hi = np.percentile([np.mean(np.random.default_rng(k).choice(s.values, len(s))) for k in range(400)], [2.5, 97.5])
        ax.errorbar(s.mean(), i, xerr=[[s.mean() - lo], [hi - s.mean()]], fmt=m, color=c, ms=5, capsize=2, lw=1)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel("Share of achievable reduction (budget = 25 %)")
    ax.grid(axis="y", visible=False)
    panel(ax, "b")
    fig.tight_layout()
    save(fig, os.path.join(FIG, "fig_budget.png"))


# ------------------------------------------------------------------ Fig: robustness
def fig_robust():
    p = os.path.join(RES, "e4/robustness.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    cfgs = list(dict.fromkeys(df.config))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 0.18 * len(cfgs) + 1.2), sharey=True)
    y = np.arange(len(cfgs))[::-1]
    ax = axs[0]
    for pol, off in [("Relay", 0.18), ("Costate-FB", 0.06), ("MPC-FB", -0.06), ("MPC-FF", -0.18)]:
        g = df[df.policy == pol].groupby("config")
        r = g.captured.mean().reindex(cfgs)
        c, m = ENT[pol]
        ax.plot(r.values, y + off, m, color=c, ms=4, label=pol, mec="white", mew=0.4)
    ax.set_yticks(y); ax.set_yticklabels(cfgs, fontsize=6.8)
    ax.set_xlabel("Share of achievable reduction")
    ax.set_title("Unconstrained energy ($\\rho$ tuned to relay energy)", color=INK)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=2)
    panel(ax, "a")
    ax = axs[1]
    for pol, off in [("Costate-FB/fixed", 0.12), ("MPC-FB/rollout", 0.0), ("MPC-FF/rollout", -0.12)]:
        r = df[df.policy == pol].groupby("config").captured.mean().reindex(cfgs)
        c, m = ENT[fam(pol)]
        ax.plot(r.values, y + off, m, color=c, ms=4, label=pol, mec="white", mew=0.4)
    ax.set_xlabel("Share of achievable reduction")
    ax.set_title("Energy budget 25 % of full duty", color=INK)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=2)
    panel(ax, "b")
    fig.tight_layout()
    save(fig, os.path.join(FIG, "fig_robust.png"))


# ------------------------------------------------------------------ Fig: weights
def fig_weights():
    p = os.path.join(RES, "e5/weights.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    nom = df[df.design == "nominal"]
    piv = nom.pivot_table(index=["weights", "seed"], columns="policy", values="CE")
    cap = {}
    for pol in ["Relay", "AO-80", "Costate-FB", "MPC-FF"]:
        cap[pol] = ((piv["Off"] - piv[pol]) / (piv["Off"] - piv["AO-100"])).groupby("weights").mean()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for pol in ["AO-80", "Costate-FB", "MPC-FF"]:
        ent = "AO" if pol.startswith("AO") else pol
        c, m = ENT[ent]
        ax.plot(cap["Relay"], cap[pol], m, color=c, ms=4, label=pol, mec="white", mew=0.4)
    lo = min(v.min() for v in cap.values()) - 0.01; hi = max(v.max() for v in cap.values()) + 0.01
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Relay: share of achievable reduction")
    ax.set_ylabel("Policy: share of achievable reduction")
    ax.legend(fontsize=7)
    save(fig, os.path.join(FIG, "fig_weights.png"))


# ------------------------------------------------------------------ Fig: computation
def fig_compute():
    p = os.path.join(RES, "e6/timing_scaling.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    names = {0: "Exhaustive enumeration", 1: "B&B, forward bound", 2: "B&B, affine bound"}
    cols = {0: ("#eb6834", "s"), 1: ("#1baf7a", "D"), 2: ("#2a78d6", "o")}
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2), sharey=True)
    for ax, (fac, lab) in zip(axs, [("nz", "zones $|\\mathcal{Z}|$"), ("np", "pollutants $|\\mathcal{P}|$"),
                                    ("Nu", "move blocks $N_u$"), ("nu", "duty levels $|\\mathcal{U}|$")]):
        d = df[df.factor == fac]
        for mode in [0, 1, 2]:
            g = d[d["mode"] == mode].groupby("value").time_us.median() / 1000
            c, m = cols[mode]
            ax.plot(g.index, g.values, "-", color=c, marker=m, ms=4, lw=1.3, label=names[mode])
        ax.set_yscale("log"); ax.set_xlabel(lab)
    axs[0].set_ylabel("Median solve time (ms)")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    save(fig, os.path.join(FIG, "fig_compute.png"))


# ------------------------------------------------------------------ Fig: forecast accuracy
def fig_forecast():
    p = os.path.join(RES, "e9/forecast_summary.csv")
    if not os.path.exists(p):
        return
    s = pd.read_csv(p)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=False)
    for ax, ch, letter in [(axs[0], "PM2.5 zone A", "a"), (axs[1], "TVOC zone B", "b")]:
        d = s[s.channel == ch]
        for (cfg, fc), g in d.groupby(["config", "forecaster"]):
            c = "#2a78d6" if "inverted" in fc else "#4a3aa7"
            ls = "-" if cfg.startswith("nominal") else "--"
            g = g.sort_values("lead_min")
            ax.plot(g.lead_min, g.rmse_vs_true / g.true_sd, ls, color=c, marker="o", ms=3.5, lw=1.2,
                    label=f"{fc}; {cfg}; vs true")
            ax.plot(g.lead_min, g.rmse_vs_inv / g.true_sd, ls, color=c, marker="x", ms=4, lw=0.8, alpha=0.7,
                    label=f"{fc}; {cfg}; vs inverted")
        ax.set_xlabel("Forecast lead (min)"); ax.set_title(ch, color=INK)
        panel(ax, letter)
    axs[0].set_ylabel("RMSE / sd of true drive")
    from matplotlib.lines import Line2D
    proxy = [Line2D([], [], color="#2a78d6", lw=1.5, label="trained on inverted drive"),
             Line2D([], [], color="#4a3aa7", lw=1.5, label="trained on true drive (simulation only)"),
             Line2D([], [], color=INK2, marker="o", ls="none", label="scored against true drive"),
             Line2D([], [], color=INK2, marker="x", ls="none", label="scored against inverted drive"),
             Line2D([], [], color=INK2, ls="-", label="nominal truth"),
             Line2D([], [], color=INK2, ls="--", label="mismatched truth (Q x0.6, infiltration x1.5)")]
    fig.legend(handles=proxy, loc="upper center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, -0.05))
    save(fig, os.path.join(FIG, "fig_forecast.png"))


if __name__ == "__main__":
    which = sys.argv[1:] or ["fronts", "budget", "robust", "weights", "compute", "forecast", "traces"]
    for w in which:
        globals()[f"fig_{w}"]()
        print("figure", w, "done", flush=True)
