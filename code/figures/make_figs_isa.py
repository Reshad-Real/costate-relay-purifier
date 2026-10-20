"""Result figures for the ISA Transactions manuscript (manuscript2), drawn at printed size (style2.py).
Replaces fig_response, fig_fronts, fig_budget, fig_robust and fig_compute of make_figs_2col.py."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis"))
import numpy as np
import pandas as pd
from style2 import setup, panel, save, ENT, INK, INK2, SINGLE, DOUBLE, plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures2")
setup()
RHO = json.load(open(os.path.join(RES, "tuning.json")))["rho_iso"]

# colour/marker per controller; one name per controller, as in the manuscript
STY = {
    "Relay": ("#eb6834", "s", "Relay (baseline)"),
    "Relay-occ": ("#eb6834", "<", "Relay, occupancy-gated"),
    "AO": ("#008300", "P", "Constant duty 80/90/100 %"),
    "Sched": ("#008300", "X", "Scheduled duty (tuned)"),
    "Timer": ("#e34948", "X", "Timer + relay (tuned)"),
    "THC": ("#e87ba4", "v", "Threshold + hysteresis (tuned)"),
    "PI-SD": ("#9c6b30", "p", "PI, sigma-delta (tuned)"),
    "Costate-FB": ("#1baf7a", "D", "Stationary costate law (CL-s)"),
    "Rollout-CL": ("#0b6e4f", "h", "Rollout costate law (CL-r)"),
    "MPC-track": ("#6b6b6b", "p", "Zone-tracking MPC (tuned)"),
    "MPC-RR": ("#8ab4ea", "o", "EMPC, relax-and-round"),
    "MPC-FB": ("#eda100", "^", "EMPC, feedback only"),
    "MPC-FF": ("#2a78d6", "o", "EMPC, schedule feedforward"),
    "MPC-oracle": ("#4a3aa7", "*", "EMPC, oracle"),
    "Rollout-CL-ex": ("#0b6e4f", "H", "CL-r, exact state and drive"),
    "MPC-FB-L": ("#c0392b", "d", "EMPC-$\\Lambda$, feedback only"),
    "MPC-oracle-L": ("#c0392b", "d", "EMPC-$\\Lambda$, exact state and drive"),
    "Clairvoyant": ("#111111", "o", "Clairvoyant (whole-day preview)"),
}


def mk(key, ms=4.2):
    c, m, _ = STY[key]
    return dict(color=c, marker=m, mec="white", mew=0.4, ms=ms if m != "*" else ms + 1.8)


def load_frame():
    from summarize_isa import frame
    return frame()


# ------------------------------------------------------------------ representative day
def fig_response():
    from aqmpc.params import Plant, MPCSettings, iPM25
    from aqmpc.scenarios import make_shift, ScenarioConfig, SHIFT, DT_SIM
    from aqmpc.controllers import Relay, MPC
    from aqmpc.extra_controllers import RolloutCostate
    from aqmpc.simulate import run_shift
    from aqmpc.harness import train_forecaster
    from aqmpc.exposure import index_series
    df = load_frame()
    ce = df.pivot(index="seed", columns="policy", values="CE")
    en = df.pivot(index="seed", columns="policy", values="energy_Wh")
    dce = 100 * (ce["MPC-FF"] - ce["Relay"]) / ce["Relay"]
    den = en["MPC-FF"] - en["Relay"]
    z = ((dce - dce.median()) / (dce.quantile(0.75) - dce.quantile(0.25))) ** 2 + \
        ((den - den.median()) / (den.quantile(0.75) - den.quantile(0.25))) ** 2
    seed = int(z.idxmin())
    pl = Plant()
    fc = train_forecaster(pl, workers=4)
    sh = make_shift(np.random.default_rng(seed), ScenarioConfig())
    st = MPCSettings(rho=RHO)
    runs = {}
    for name, ctrl in [("Relay", Relay()), ("Rollout-CL", RolloutCostate(pl, st)), ("MPC-FF", MPC(pl, fc, "learned", st))]:
        runs[name] = run_shift(sh, ctrl, pl, sensor_seed=seed + 7919, return_traces=True)
    meta = {k: v[0] for k, v in runs.items()}
    meta["seed"] = seed
    json.dump(meta, open(os.path.join(RES, "e1", "response_example.json"), "w"), indent=1)
    t = runs["Relay"][1]["t"] + 7.0
    fig, axs = plt.subplots(5, 1, figsize=(DOUBLE, 6.2), sharex=True,
                            gridspec_kw={"height_ratios": [0.85, 1.8, 1.2, 1.25, 1.25], "hspace": 0.28})
    ax = axs[0]
    row = {"weld": 2, "solder": 1, "clean": 0}
    for e in sh.plan:
        ax.add_patch(plt.Rectangle((e.t0 + 7, row[e.kind] - 0.36), e.t1 - e.t0, 0.30, facecolor="#b9b8b2",
                                   edgecolor="white", lw=0.5))
    for e in sh.actual:
        ax.add_patch(plt.Rectangle((e.t0 + 7, row[e.kind] + 0.04), e.t1 - e.t0, 0.30,
                                   facecolor=INK if e.planned else "#e34948", edgecolor="white", lw=0.5))
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["clean (B)", "solder (A)", "weld (A)"])
    ax.set_ylim(-0.5, 2.5); ax.grid(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#b9b8b2"), plt.Rectangle((0, 0), 1, 1, color=INK),
                       plt.Rectangle((0, 0), 1, 1, color="#e34948")],
              labels=["planned", "actual", "unscheduled"], ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.36))
    panel(ax, "a", y=1.06)
    names = ["Relay", "MPC-FF", "Rollout-CL"]          # CL-r drawn last and dashed: it nearly coincides with EMPC-FF
    ls = {"Relay": "-", "MPC-FF": "-", "Rollout-CL": (0, (3, 2))}
    lw = {"Relay": 0.9, "MPC-FF": 1.5, "Rollout-CL": 1.1}
    cum_ce = {n: np.cumsum(index_series(runs[n][1]["C"]) * sh.occ * DT_SIM) for n in names}
    cum_e = {n: np.cumsum(pl.fan.power(runs[n][1]["U"]) * DT_SIM) for n in names}
    for name in names:
        kw = dict(color=STY[name][0], ls=ls[name], lw=lw[name])
        axs[1].plot(t, runs[name][1]["C"][:, iPM25, 0], label=STY[name][2], **kw)
        axs[2].step(t, runs[name][1]["U"] * 100, where="post", **kw)
        if name != "Relay":
            axs[3].plot(t, cum_ce[name] - cum_ce["Relay"], **kw)
            axs[4].plot(t, cum_e[name] - cum_e["Relay"], **kw)
    for a in axs[3:]:
        a.axhline(0, color=STY["Relay"][0], lw=0.9)
    axs[1].set_ylabel("PM$_{2.5}$, zone A\n(µg m$^{-3}$)")
    axs[2].set_ylabel("Fan duty (%)"); axs[2].set_ylim(-5, 108); axs[2].set_yticks([0, 50, 80, 100])
    axs[3].set_ylabel("Cumulative CE\nminus relay (index h)")
    axs[4].set_ylabel("Cumulative energy\nminus relay (Wh)"); axs[4].set_xlabel("Time of day (h)")
    for a, letter in zip(axs[1:], "bcde"):
        panel(a, letter)
    for a in axs:
        a.axvspan(7, SHIFT[0] + 7, color="#f0efec", lw=0, zorder=0)
        a.axvspan(SHIFT[1] + 7, 17, color="#f0efec", lw=0, zorder=0)
        a.set_xlim(7, 17)
    fig.align_ylabels(axs)
    h, l = axs[1].get_legend_handles_labels()             # legend outside the axes: no overlap with the data
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=3)
    save(fig, os.path.join(FIG, "fig_response.pdf"))


# ------------------------------------------------------------------ exposure-energy fronts
def _front(ax, df, prefix, key, lw=1.2):
    sub = df[df.policy.str.startswith(prefix + "|")]
    if sub.empty:
        return
    g = sub.groupby("policy")[["Efrac", "captured"]].mean().sort_values("Efrac")
    ax.plot(g.Efrac, g.captured, "-", lw=lw, label=STY[key][2], zorder=3, **mk(key, 3.6))


def _point(ax, df, pol, key, label=True, ms=5.5):
    sub = df[df.policy == pol]
    if not sub.empty:
        ax.plot(sub.Efrac.mean(), sub.captured.mean(), ls="none", label=STY[key][2] if label else None, zorder=4,
                **mk(key, ms))


def _gap_panel(ax, df, nf, main):
    """Paired distance to the EMPC-FF front at equal energy (nominal purifier): for every day, the policy's captured
    share minus the day's EMPC-FF front (PCHIP over the rho sweep) at the policy's energy, averaged over the days."""
    from summarize import interp_front, boot_ci
    ff = {sd: g for sd, g in df[df.policy.str.startswith("MPC-FF|")].groupby("seed")}

    def dist(q):
        q = q[q.seed.isin(ff.keys())]
        return np.array([float(interp_front(ff[r.seed], r.Efrac)) - r.captured for r in q.itertuples()]), q.Efrac.mean()

    for src, prefix, key in [(nf, "THC", "THC"), (nf, "TRK", "MPC-track"), (df, "Costate-FB", "Costate-FB"),
                             (df, "MPC-FB", "MPC-FB"), (df, "MPC-oracle", "MPC-oracle")]:
        sub = src[src.policy.str.startswith(prefix + "|")] if not src.empty else src
        if sub.empty:
            continue
        pts = []
        for pol, q in sub.groupby("policy"):
            d, e = dist(q)
            if 0.12 < e < 0.86:
                pts.append((e, -d.mean()))
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", lw=1.1, zorder=3, **mk(key, 3.4))
    sub = main[main.seed.isin(df.seed.unique())]
    for pol, key in [("Relay", "Relay"), ("Relay-occ", "Relay-occ"), ("AO-80", "AO"), ("Sched", "Sched"),
                     ("Timer", "Timer"), ("PI-SD", "PI-SD"), ("Rollout-CL", "Rollout-CL"), ("MPC-RR", "MPC-RR")]:
        q = sub[sub.policy == pol]
        if q.empty:
            continue
        d, e = dist(q)
        lo, hi = boot_ci(-d)
        ax.errorbar(e, -d.mean(), yerr=[[-d.mean() - lo], [hi + d.mean()]], ls="none", zorder=4, elinewidth=0.8,
                    capsize=1.2, label=STY[key][2] if key not in ("Relay", "AO") else None, **mk(key, 5.0))
    ax.axhline(0, color=STY["MPC-FF"][0], lw=1.0)
    ax.set_xlim(0.12, 0.86)
    ax.set_xlabel("Fan energy / energy at full duty")
    ax.set_ylabel(r"$\kappa-\kappa_{\mathrm{EMPC\!-\!FF}}$ at equal energy")
    ax.set_title("Distance to the EMPC-FF front", color=INK)


def fig_fronts():
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.8), gridspec_kw={"wspace": 0.32, "width_ratios": [1, 1.15, 1]})
    main = load_frame()
    nf = pd.read_csv(os.path.join(RES, "e11/fronts.csv")) if os.path.exists(os.path.join(RES, "e11/fronts.csv")) else pd.DataFrame()
    df1 = pd.read_csv(os.path.join(RES, "e2/fronts_scale1.0.csv"))
    _gap_panel(axs[1], df1, nf, main)
    panel(axs[1], "b")
    for ax, sc, letter, title in [(axs[0], 1.0, "a", "Nominal purifier"),
                                  (axs[2], 4.0, "c", r"$4\times$ purifier flow (hypothetical)")]:
        df = pd.read_csv(os.path.join(RES, f"e2/fronts_scale{sc}.csv"))
        if sc == 1.0 and not nf.empty:
            _front(ax, nf, "TRK", "MPC-track")
            _front(ax, nf, "THC", "THC")
        _front(ax, df, "Costate-FB", "Costate-FB")
        _front(ax, df, "MPC-FB", "MPC-FB")
        _front(ax, df, "MPC-FF", "MPC-FF")
        _front(ax, df, "MPC-oracle", "MPC-oracle", lw=0.8)
        _point(ax, df, "Relay", "Relay")
        for u in ["AO-80", "AO-90", "AO-100"]:
            sub = df[df.policy == u]
            ax.plot(sub.Efrac.mean(), sub.captured.mean(), ls="none", zorder=4,
                    label=STY["AO"][2] if u == "AO-80" else None, **mk("AO", 5.5))
            ax.annotate(u.replace("AO-", "") + " %", (sub.Efrac.mean(), sub.captured.mean()),
                        textcoords="offset points", xytext=(0, -11) if u == "AO-100" else (4, -9), fontsize=7, color=INK2,
                        ha="center" if u == "AO-100" else "left", zorder=6,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.4))   # grid lines must not cross the label
        ax.set_xlabel("Fan energy / energy at full duty")
        ax.set_title(title, color=INK)
        ax.set_xlim(0, 1.06); ax.set_ylim(0, 1.04)
        ax.set_ylabel("Share of achievable reduction $\\kappa$")
        panel(ax, letter)
    h, l = axs[0].get_legend_handles_labels()
    h2, l2 = axs[1].get_legend_handles_labels()
    fig.legend(h + h2, l + l2, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    save(fig, os.path.join(FIG, "fig_fronts.pdf"))


# ------------------------------------------------------------------ budget
def fig_budget():
    d = pd.read_csv(os.path.join(RES, "e12/budget_fig.csv"))
    fam = {"Relay+cutoff": "Relay", "THC/cutoff*": "THC", "Costate-FB/fixed": "Costate-FB",
           "MPC-FB/fixed*": "MPC-FB", "MPC-FF/fixed*": "MPC-FF", "MPC-FB/pacing*": "MPC-FB",
           "MPC-FB/rollout": "MPC-FB", "Costate-FB/rollout": "Costate-FB", "MPC-FF/rollout": "MPC-FF",
           "MPC-oracle/rollout": "MPC-oracle", "RCL/fixed*": "Rollout-CL", "Clairvoyant/budget": "Clairvoyant"}
    fig, ax = plt.subplots(figsize=(SINGLE, 2.9))
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        k = fam.get(r.key, "MPC-FF")
        ax.errorbar(r.kappa, yi, xerr=[[r.kappa - r.lo], [r.hi - r.kappa]], ls="none", elinewidth=0.9, capsize=1.6,
                    **mk(k, 4.6))
        ax.annotate(f"{r.eps:.3f}", (1.0, yi), xycoords=("axes fraction", "data"), xytext=(2, 0),
                    textcoords="offset points", va="center", fontsize=6.8, color=INK2, annotation_clip=False)
    group = {"Relay+cutoff": 0, "THC/cutoff*": 0, "Costate-FB/fixed": 1, "RCL/fixed*": 1, "MPC-FB/fixed*": 1,
             "MPC-FB/pacing*": 1, "MPC-FB/rollout": 1, "Costate-FB/rollout": 2, "MPC-FF/fixed*": 2, "MPC-FF/rollout": 2,
             "MPC-oracle/rollout": 3, "Clairvoyant/budget": 3}
    gk = [group.get(k, 1) for k in d.key]
    for i in range(1, len(gk)):
        if gk[i] != gk[i - 1]:
            ax.axhline((y[i] + y[i - 1]) / 2, color=INK2, lw=0.5, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(d.policy.str.replace("\\,", " ", regex=False), fontsize=7)
    ax.set_xlabel("Share of achievable reduction $\\kappa$")
    ax.text(1.0, 1.0, "$\\bar\\varphi$", transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color=INK2)
    ax.grid(axis="y", lw=0)
    save(fig, os.path.join(FIG, "fig_budget.pdf"))


# ------------------------------------------------------------------ robustness
def fig_robust():
    from summarize import scenario_label
    R = pd.read_csv(os.path.join(RES, "e12/robust_summary.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 6.2), sharey=True,
                            gridspec_kw={"wspace": 0.06, "width_ratios": [1, 1, 1]})
    y = np.arange(len(R))[::-1].astype(float)
    nf = int((R.group == "fault").sum())

    def series(ax, col, key, off, lab):
        if col not in R or R[col].isna().all():
            return
        d = R[R[col].notna()]
        yy = y[R[col].notna().values]
        ax.errorbar(d[col], yy + off, xerr=[d[col] - d[col + "_lo"], d[col + "_hi"] - d[col]], ls="none",
                    elinewidth=0.8, capsize=1.2, label=lab, **mk(key, 3.6))

    series(axs[0], "dJ_relay", "MPC-FF", 0.17, "EMPC-FF")
    series(axs[0], "dJ_rcl_relay", "Rollout-CL", -0.17, "CL-r")
    axs[0].set_xlabel("$\\Delta J$ to the baseline relay (%)")
    axs[0].set_title("(a) Unconstrained, against relay", color=INK)
    series(axs[1], "dJ_thc", "MPC-FF", 0.17, "EMPC-FF")
    series(axs[1], "dJ_rcl_thc", "Rollout-CL", -0.17, "CL-r")
    axs[1].set_xlabel("$\\Delta J$ to the tuned THC (%)")
    axs[1].set_title("(b) Unconstrained, against THC", color=INK)
    series(axs[2], "db", "MPC-FF", 0.17, "EMPC-FF, forecast price")
    series(axs[2], "dbr", "Rollout-CL", -0.17, "CL-r, fixed price")
    axs[2].set_xlabel("$\\Delta\\kappa$ to tuned EMPC-FB, fixed price")
    axs[2].set_title("(c) Budget 25 % of full-duty energy", color=INK)
    for a in axs:
        a.axvline(0, color=INK2, lw=0.6)
        a.axhline(nf - 0.5, color=INK2, lw=0.6, ls="--")
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=1)
    labels = [scenario_label(c).replace("\\,", " ").replace("\\%", "%").replace("\\#", "#").replace("--", "–").replace("bias -", "bias −")
              for c in R.config]
    axs[0].set_yticks(y); axs[0].set_yticklabels(labels, fontsize=6.8)
    axs[0].set_ylim(-0.8, len(R) - 0.2)
    save(fig, os.path.join(FIG, "fig_robust.pdf"))


# ------------------------------------------------------------------ leverage sweep
def fig_leverage():
    R = pd.read_csv(os.path.join(RES, "e15/leverage_summary.csv"))
    key = {"Relay": "Relay", "Sched": "Sched", "CL-s": "Costate-FB", "EMPC-FB": "MPC-FB", "EMPC-oracle": "MPC-oracle",
           "EMPC-FB-L": "MPC-FB-L", "EMPC-oracle-L": "MPC-oracle-L", "CL-r": "Rollout-CL", "CL-r-exact": "Rollout-CL-ex"}
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.9), gridspec_kw={"wspace": 0.3})
    eps = np.sort(R.eps.unique())
    for pol, k in key.items():
        d = R[R.policy == pol].sort_values("eps")
        if d.empty:
            continue
        ls = "--" if pol in ("CL-r-exact", "EMPC-oracle-L") else "-"
        for ax, col, lo, hi, f in [(axs[0], "rel", "rel_lo", "rel_hi", 100.0), (axs[1], "abs_pct", "abs_lo", "abs_hi", 1.0)]:
            y = f * d[col].clip(lower=1e-6)
            yerr = [y - f * d[lo].clip(lower=1e-6), f * d[hi].clip(lower=1e-6) - y]
            ax.errorbar(d.eps, y, yerr=yerr, ls=ls, lw=1.1, elinewidth=0.7, capsize=1.2,
                        label=STY[k][2] if ax is axs[0] else None, **mk(k, 4.2))
    cf = os.path.join(RES, "e15/cert_summary.csv")
    if os.path.exists(cf):                          # certificate beta of CL-r (exact) from its own record
        C = pd.read_csv(cf).groupby("scale")[["brr", "br"]].mean()
        ev = R.drop_duplicates("scale").set_index("scale").eps.reindex(C.index)
        for ax, col in [(axs[0], "brr"), (axs[1], "br")]:
            ax.plot(ev, C[col], color="black", lw=0.9, ls=(0, (4, 1.5)), marker="s", ms=3, mfc="white", zorder=4,
                    label="Certificate $\\beta$, CL-r exact" if ax is axs[0] else None)
    # reference slopes through the CL-r (exact information) point at the smallest leverage
    d = R[R.policy == "CL-r-exact"].sort_values("eps")
    if not d.empty:
        x0 = d.eps.iloc[0]
        for ax, y0, slope, lab in [(axs[0], 100 * d.rel.iloc[0], 1, r"$\propto\varepsilon$"),
                                   (axs[1], d.abs_pct.iloc[0], 2, r"$\propto\varepsilon^2$")]:
            xs = np.array([eps.min(), eps.max()])
            # labelled in the legend, not in the plot area, so that no text sits on the data
            ax.plot(xs, y0 * (xs / x0) ** slope, color=INK2, lw=0.8, ls=":", zorder=1,
                    label=r"Slope reference: $\propto\varepsilon$ in (a), $\propto\varepsilon^2$ in (b)"
                    if ax is axs[0] else None)
    for ax, lab in [(axs[0], "Share of the controllable range lost (%)"), (axs[1], r"Cost above clairvoyant reference (% of $J^{\mathrm{c}}$)")]:
        ax.set_xscale("log"); ax.set_yscale("log")
        sc_of = R.drop_duplicates("eps").set_index("eps").scale
        ax.set_xticks(eps)                         # every leverage ticked, labels at the doubling points only
        ax.set_xticklabels([f"{e:.2f}" if sc_of[e] in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0) else "" for e in eps])
        ax.minorticks_off()
        ax.set_xlabel(r"Actuator leverage $\varepsilon$")
        ax.set_ylabel(lab)
    e_tb = float(R[R.scale == 1.0].eps.iloc[0])     # the testbed leverage (as-built purifier)
    axs[0].axvline(e_tb, color=INK2, lw=0.6, ls="--")
    axs[1].axvline(e_tb, color=INK2, lw=0.6, ls="--")
    axs[0].annotate("nominal", (e_tb, axs[0].get_ylim()[1]), textcoords="offset points", xytext=(3, -9), fontsize=7,
                    color=INK2)
    panel(axs[0], "a"); panel(axs[1], "b")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    save(fig, os.path.join(FIG, "fig_leverage.pdf"))


# ------------------------------------------------------------------ computation
def fig_compute():
    df = pd.read_csv(os.path.join(RES, "e6/timing_scaling.csv"))
    names = {0: "Exhaustive enumeration", 1: "B&B, bound (a)", 2: "B&B, bound (b)"}
    cols = {0: ("#eb6834", "s"), 1: ("#1baf7a", "D"), 2: ("#2a78d6", "o")}
    fig, axs = plt.subplots(1, 4, figsize=(DOUBLE, 1.95), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (fac, lab) in zip(axs, [("nz", "zones $|\\mathcal{Z}|$"), ("np", "pollutants $|\\mathcal{P}|$"),
                                    ("Nu", "move blocks $N_u$"), ("nu", "duty levels $|\\mathcal{U}|$")]):
        d = df[df.factor == fac]
        for mode in [0, 1, 2]:
            g = d[d["mode"] == mode].groupby("value").time_us.median() / 1000
            c, m = cols[mode]
            ax.plot(g.index, g.values, "-", color=c, marker=m, ms=3.6, lw=1.1, mec="white", mew=0.4, label=names[mode])
        ax.set_yscale("log"); ax.set_xlabel(lab)
    axs[0].set_ylabel("Median solve time (ms)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    save(fig, os.path.join(FIG, "fig_compute.pdf"))


if __name__ == "__main__":
    which = sys.argv[1:] or ["compute", "fronts", "budget", "robust", "response", "leverage"]
    for w in which:
        globals()[f"fig_{w}"]()
        print("figure", w, "done", flush=True)
