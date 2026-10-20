"""Statistics, LaTeX tables and number macros for the ISA Transactions manuscript (manuscript2/main.tex).

Reuses summarize.py (helpers and the unchanged experiment summaries) and replaces the main comparison (E1 + E11 + E12
baselines), the budget study (E3 + tuned E12 comparators), the robustness study (E4 + E12 faults, E4b, Monte Carlo) and
adds the structural checks (E13), solver alternatives (E6 methods / MILP / operation counts) and forecast skill (E14).
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize as S
from summarize import mac, paired, holm, prel, prel_le, pfmt, boot_ci, tex_table, texnum, scenario_label, RES, GEN

RHO = json.load(open(os.path.join(RES, "tuning.json")))["rho_iso"]
E_FULL = 6.96 * 10.0          # fan energy of constant full duty over the 10-h day (Wh), as in experiments/common.py
for _d in ("e12", "e13", "e14", "e15"):
    os.makedirs(os.path.join(RES, _d), exist_ok=True)


def rj(p):
    return json.load(open(os.path.join(RES, p)))


def rc(p):
    return pd.read_csv(os.path.join(RES, p))


def exists(p):
    return os.path.exists(os.path.join(RES, p))


KEY = {"Relay": "Relay", "Relay-occ": "RelayOcc", "AO-80": "AOEighty", "AO-100": "AOHundred", "Off": "Off",
       "Sched": "Sched", "PI-SD": "PISD", "Timer": "Timer", "THC": "THC", "Costate-FB": "CostateFB",
       "Rollout-CL": "RCL", "MPC-track": "MPCtrack", "MPC-RR": "MPCRR", "MPC-FB": "MPCFB", "MPC-FF": "MPCFF",
       "MPC-oracle": "MPCoracle", "Relay-PM": "RelayPM", "MPC-FB-L": "MPCFBL", "Clairvoyant": "Clair"}
LABEL = {"Off": "Fan off (reference)", "AO-100": "Constant 100\\,\\% (reference)", "Relay": "Relay (baseline)",
         "Relay-occ": "Relay, occupancy-gated", "AO-80": "Constant 80\\,\\%", "Sched": "Scheduled duty (tuned)",
         "PI-SD": "PI, sigma--delta output (tuned)", "Timer": "Timer + relay (tuned)",
         "THC": "Threshold + hysteresis (tuned)", "Costate-FB": "Stationary costate law (CL-s)",
         "Rollout-CL": "Rollout costate law (CL-r)", "MPC-track": "Zone-tracking MPC (tuned)",
         "MPC-RR": "EMPC, relax-and-round", "MPC-FB": "EMPC, feedback only", "MPC-FF": "EMPC, schedule feedforward",
         "MPC-oracle": "EMPC, oracle", "MPC-FB-L": "EMPC-$\\Lambda$ (feedback only)",
         "Clairvoyant": "Clairvoyant schedule (whole-day preview)"}
ORDER = ["Off", "AO-100", "Relay", "Relay-occ", "AO-80", "Sched", "PI-SD", "Timer", "THC", "Costate-FB", "Rollout-CL",
         "MPC-track", "MPC-RR", "MPC-FB", "MPC-FF", "MPC-oracle", "MPC-FB-L", "Clairvoyant"]


def thc_choice():
    """Raw-reading THC (E11, refined) or 60-s averaged THC (E12): whichever has the lower tuning cost."""
    j_raw = min(rc(p).J.min() for p in ["e11/grid_thc.csv", "e11/grid_thc_refine.csv"] if exists(p))
    j_avg = rc("e12/grid_thcavg.csv").J.min() if exists("e12/grid_thcavg.csv") else np.inf
    return ("THC-avg", rj("e12/tuning3.json")["thcavg"]) if j_avg < j_raw else ("THC", rj("e11/tuning2.json")["thc"])


def frame():
    d = rc("e1/main.csv")
    parts = [d[~d.policy.isin(["PI", "Timer+Relay-PM"])]]
    if exists("e1/relay_occ.csv"):
        parts.append(rc("e1/relay_occ.csv"))
    thc_name, _ = thc_choice()
    if exists("e11/test.csv"):
        t = rc("e11/test.csv")
        t = t[t.policy.isin(["MPC-track", "MPC-RR", "Timer*"] + (["THC"] if thc_name == "THC" else []))]
        parts.append(t.replace({"policy": {"Timer*": "Timer"}}))
    if exists("e12/test_base2.csv"):
        t = rc("e12/test_base2.csv")
        keep = ["Sched", "PI-SD", "Rollout-CL", "Rollout-CL-oracle", "Rollout-CL-pen"] + (["THC-avg"] if thc_name == "THC-avg" else [])
        parts.append(t[t.policy.isin(keep)].replace({"policy": {"THC-avg": "THC"}}))
    off = d[d.policy == "Off"].set_index("seed").CE
    full = d[d.policy == "AO-100"].set_index("seed").CE
    lamf = "e15/lambda1.csv" if exists("e15/lambda1.csv") else "e15/lambda.csv"
    for f, sel in [("e15/clair_main.csv", None), (lamf, "EMPC-FB-L")]:
        if exists(f):
            c = rc(f)
            if sel:
                c = c[(c.policy == sel) & (c.scale == 1.0)].replace({"policy": {sel: "MPC-FB-L"}})
            c["captured"] = [(off[sd] - ce) / (off[sd] - full[sd]) for sd, ce in zip(c.seed, c.CE)]
            c["Efrac"] = c.energy_Wh / E_FULL
            parts.append(c)
    cols = d.columns
    out = pd.concat([p[[c for c in cols if c in p.columns]] for p in parts], ignore_index=True)
    out["J"] = out.CE + RHO * out.energy_Wh
    return out


def e1_isa():
    df = frame()
    mac("rhoIso", RHO, "{:.3g}")
    mac("nTest", df.seed.nunique(), "{:d}")
    piv = {k: df.pivot(index="seed", columns="policy", values=k) for k in
           ["captured", "Efrac", "energy_Wh", "CE", "TWA_PM25", "switches", "exceed_L3_min", "J"]}
    J = piv["J"]
    ref = "Clairvoyant" if "Clairvoyant" in J else "MPC-FF"      # reference of the gap and Delta J columns
    jr, jf = J["Relay"].mean(), J[ref].mean()
    rows, ps, prow = [], [], []
    for pol in ORDER:
        if pol not in J:
            continue
        gap = 100 * (jr - J[pol].mean()) / (jr - jf)
        r = [LABEL[pol], f"{piv['captured'][pol].mean():.3f}", f"{piv['Efrac'][pol].mean():.3f}",
             f"{piv['CE'][pol].mean():.2f}", f"{piv['TWA_PM25'][pol].mean():.1f}", f"{piv['switches'][pol].mean():.0f}",
             f"{piv['exceed_L3_min'][pol].mean():.1f}", f"{J[pol].mean():.3f}",
             "--" if pol in ("Off", "AO-100") else f"{gap:.1f}"]
        if pol in (ref, "Off", "AO-100"):
            r.append("--")
        else:
            dm, lo, hi, p, _ = paired(100 * (J[pol] - J[ref]) / jf, np.zeros(len(J)))
            r.append([dm, lo, hi]); ps.append(p); prow.append(len(rows))
        rows.append(r)
    padj = holm(ps)
    for i, q in zip(prow, padj):
        dm, lo, hi = rows[i][-1]
        rows[i][-1] = f"{dm:+.3f} [{lo:+.3f}, {hi:+.3f}]" + ("$^\\dagger$" if q >= 0.05 else "")
    sig = [q for q in padj if q < 0.05]
    mac("eOneMaxHolmP", prel_le(max(sig)) if sig else "n.a.", "{}")
    hdr1 = ["Policy", "$\\bar\\kappa$", "$\\bar\\varphi$", "CE", "TWA$_{2.5}$", "Sw.", "$t_{>\\bar c}$", "$J$", "gap (\\%)",
            "$\\Delta J$ to " + ("clairvoyant" if ref == "Clairvoyant" else "EMPC-FF") + " (\\%) [95\\% CI]"]
    pols = [pol for pol in ORDER if pol in J]
    main1 = ["Off", "AO-100", "Relay", "THC", "Rollout-CL", "MPC-FF", "MPC-FB-L", "Clairvoyant"]
    tex_table(os.path.join(GEN, "tab_e1_full.tex"), hdr1, rows, "lrrrrrrrrl", label=lambda s: s)
    tex_table(os.path.join(GEN, "tab_e1.tex"), hdr1, [r for pol, r in zip(pols, rows) if pol in main1], "lrrrrrrrrl",
              label=lambda s: s)                       # the paper's table; the full one is in the supplement
    for pol, k in KEY.items():
        if pol not in J:
            continue
        mac(f"cap{k}", piv["captured"][pol].mean(), "{:.3f}")
        mac(f"ef{k}", piv["Efrac"][pol].mean(), "{:.3f}")
        mac(f"en{k}", piv["energy_Wh"][pol].mean(), "{:.1f}")
        mac(f"sw{k}", piv["switches"][pol].mean(), "{:.0f}")
        mac(f"twa{k}", piv["TWA_PM25"][pol].mean(), "{:.1f}")
        mac(f"ce{k}", piv["CE"][pol].mean(), "{:.2f}")
        mac(f"J{k}", J[pol].mean(), "{:.3f}")
        mac(f"gap{k}", 100 * (jr - J[pol].mean()) / (jr - jf), "{:.1f}")
    mac("swMPCFF", piv["switches"]["MPC-FF"].mean(), "{:.0f}")
    mac("exceedMaxAll", max(piv["exceed_L3_min"][k].mean() for k in ORDER if k in J and k != "Off"), "{:.1f}")
    mac("exceedFull", piv["exceed_L3_min"]["AO-100"].mean(), "{:.1f}")           # capacity limit: full duty all day
    for pol, k in [("Rollout-CL", "RCL"), ("Rollout-CL-pen", "RCLpen")]:      # what the soft-bound penalty buys
        if pol in J:
            mac(f"exceed{k}", piv["exceed_L3_min"][pol].mean(), "{:.1f}")
            mac(f"ef{k}", piv["Efrac"][pol].mean(), "{:.3f}")
    try:                                                     # where the full-duty exceedance happens (outdoor level)
        import sys as _s; _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from aqmpc.scenarios import make_shift, ScenarioConfig
        from aqmpc.params import iPM25
        ex = piv["exceed_L3_min"]["AO-100"]
        out_pm = pd.Series({sd: make_shift(np.random.default_rng(int(sd)), ScenarioConfig()).c_out[:, iPM25].mean()
                            for sd in ex.index})
        top = out_pm > out_pm.quantile(0.75)
        mac("exceedTopShare", 100 * ex[top].sum() / ex.sum(), "{:.0f}")
    except Exception as e:
        print("exceedTopShare skipped:", e)
    mac("exceedMinAll", min(piv["exceed_L3_min"][k].mean() for k in ORDER if k in J and k != "Off"), "{:.1f}")
    lev = (piv["CE"]["Off"] - piv["CE"]["AO-100"]) / piv["CE"]["Off"]
    mac("leverage", 100 * lev.mean(), "{:.1f}")
    mac("leverageLo", 100 * lev.quantile(0.05), "{:.1f}")
    mac("leverageHi", 100 * lev.quantile(0.95), "{:.1f}")
    mac("twaSpan", (piv["TWA_PM25"]["Off"] - piv["TWA_PM25"]["AO-100"]).mean(), "{:.1f}")
    # captured-share comparisons
    for a, b, nm in [("MPC-FF", "Relay", "FFvsRelay"), ("MPC-FF", "THC", "FFvsTHC"), ("THC", "Relay", "THCvsRelay"),
                     ("Rollout-CL", "Relay", "RCLvsRelay")]:
        if a in J and b in J:
            dm, lo, hi, p, share = paired(piv["captured"][a], piv["captured"][b])
            mac(f"dCap{nm}", dm, "{:+.3f}"); mac(f"dCap{nm}Lo", lo, "{:+.3f}"); mac(f"dCap{nm}Hi", hi, "{:+.3f}")
            mac(f"share{nm}", 100 * share, "{:.0f}")
            dce = paired(piv["CE"][a], piv["CE"][b])
            mac(f"dCE{nm}", 100 * dce[0] / piv["CE"][b].mean(), "{:+.2f}")
            dt = paired(piv["TWA_PM25"][a], piv["TWA_PM25"][b])
            mac(f"dTWA{nm}", dt[0], "{:+.2f}"); mac(f"dTWA{nm}Lo", dt[1], "{:+.2f}"); mac(f"dTWA{nm}Hi", dt[2], "{:+.2f}")
    # economic-cost comparisons (% of EMPC-FF's mean cost); shareJ = % of days on which the first has the lower J
    for a, b, nm in [("MPC-FF", "THC", "FFvsTHC"), ("MPC-FF", "Rollout-CL", "FFvsRCL"), ("MPC-FF", "MPC-RR", "FFvsRR"),
                     ("MPC-FF", "MPC-track", "FFvsTRK"), ("MPC-FF", "Relay", "FFvsRelay"), ("MPC-FF", "Sched", "FFvsSched"),
                     ("Rollout-CL", "THC", "RCLvsTHC"), ("Rollout-CL", "AO-80", "RCLvsAOEighty"),
                     ("Rollout-CL", "Costate-FB", "RCLvsCF"), ("MPC-FF", "MPC-FB", "FFvsFB"), ("MPC-oracle", "MPC-FF", "ORvsFF"),
                     ("MPC-FF", "Costate-FB", "FFvsCF"), ("Rollout-CL-oracle", "Rollout-CL", "RCLORvsRCL")]:
        if a not in J or b not in J:
            continue
        dm, lo, hi, p, _ = paired(J[a], J[b])
        base = J["MPC-FF"].mean()                       # all cost differences in % of EMPC-FF's mean cost
        mac(f"dJ{nm}", 100 * dm / base, "{:+.3f}"); mac(f"dJ{nm}Lo", 100 * lo / base, "{:+.3f}")
        mac(f"dJ{nm}Hi", 100 * hi / base, "{:+.3f}"); mac(f"pJ{nm}", prel(p), "{}")
        mac(f"shareJ{nm}", 100 * np.mean(J[a] < J[b]), "{:.0f}")
    for a, b, nm in [("MPC-FF", "MPC-FB", "FFvsFB"), ("MPC-oracle", "MPC-FF", "ORvsFF")]:
        dc, clo, chi, pc, _ = paired(piv["captured"][a], piv["captured"][b])
        mac(f"dCap{nm}", dc, "{:+.4f}"); mac(f"dCap{nm}Lo", clo, "{:+.4f}"); mac(f"dCap{nm}Hi", chi, "{:+.4f}")
        mac(f"pCap{nm}", prel(pc), "{}")
    # sample size for a physical paired crossover (paired permutation / t-type test)
    d = piv["captured"]["MPC-FF"] - piv["captured"]["Relay"]
    dz = abs(d.mean()) / d.std(ddof=1)
    nfor = lambda z: int(np.ceil(((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) / z) ** 2 + 0.5 * stats.norm.ppf(0.975) ** 2))
    mac("dzCap", dz, "{:.2f}"); mac("nPairsCap", nfor(dz), "{:d}")
    mac("nPairsHalf", nfor(0.5), "{:d}"); mac("nPairsThird", nfor(0.3), "{:d}")
    # tuned settings
    thc_name, th = thc_choice()
    mac("thcThOne", th["th1"], "{:g}"); mac("thcThTwo", th["th1"] + th["dth"], "{:g}")
    mac("thcThThree", th["th1"] + 3 * th["dth"], "{:g}"); mac("thcHyst", th["hyst"], "{:g}")
    mac("thcInput", ("60-s averaged" if thc_name == "THC-avg" else "raw") + " readings, "
        + ("with" if th.get("gate", True) else "without") + " occupancy gating", "{}")
    if exists("e12/tuning3.json"):
        t3 = rj("e12/tuning3.json")
        mac("schedDuty", 100 * t3["sched"]["u"], "{:.0f}"); mac("schedPre", 60 * t3["sched"]["pre"], "{:.0f}")
        mac("pisdR", t3["pisd"]["r25"], "{:g}"); mac("pisdKp", t3["pisd"]["kp"], "{:g}"); mac("pisdKi", t3["pisd"]["ki"], "{:g}")
    t2 = rj("e11/tuning2.json")
    mac("tmLead", 60 * t2["timer"]["lead"], "{:.0f}"); mac("tmTail", 60 * t2["timer"]["tail"], "{:.0f}")
    mac("tmDuty", 100 * t2["timer"]["u_ff"], "{:.0f}")
    mac("trkR", t2["track"]["r25"], "{:g}"); mac("trkQ", t2["track"]["q"], "{:g}")
    return df


def fronts_isa():
    if not (exists("e11/fronts.csv") and exists("e2/fronts_scale1.0.csv")):
        return
    mf = rc("e2/fronts_scale1.0.csv"); mf["fam"] = mf.policy.str.split("|").str[0]
    t = frame(); thc = t[t.policy == "THC"].set_index("seed")
    d_iso = [S.interp_front(g, thc.Efrac[sd]) - thc.captured[sd] for sd, g in mf[mf.fam == "MPC-FF"].groupby("seed")
             if sd in thc.index]
    mac("isoFFvsTHC", np.mean(d_iso), "{:+.3f}")
    lo, hi = boot_ci(d_iso); mac("isoFFvsTHCLo", lo, "{:+.3f}"); mac("isoFFvsTHCHi", hi, "{:+.3f}")
    # practical stakes: fan energy the economic front needs for the same captured share as a rule-based design
    # (per day, inverse interpolation of the swept EMPC-FF front), in % of that design's own energy
    for pol, key in [("Relay", "Relay"), ("THC", "THC")]:
        q = t[(t.policy == pol)].set_index("seed")
        sv = []
        for sd, g in mf[mf.fam == "MPC-FF"].groupby("seed"):
            if sd not in q.index:
                continue
            g = g.sort_values("Efrac")
            k, e = np.maximum.accumulate(g.captured.values), g.Efrac.values
            target = q.captured[sd]
            if target < k.min() or target > k.max():
                continue
            e_iso = float(np.interp(target, k, e))
            sv.append(100 * (1 - e_iso / q.Efrac[sd]))
        if sv:
            lo, hi = boot_ci(np.array(sv))
            mac(f"isoESave{key}", np.mean(sv), "{:.0f}"); mac(f"isoESave{key}Lo", lo, "{:.0f}")
            mac(f"isoESave{key}Hi", hi, "{:.0f}"); mac(f"isoESave{key}N", len(sv), "{:d}")
    # distance of every operating point below the EMPC-FF front at its own energy (paired by day)
    ff = {sd: g for sd, g in mf[mf.fam == "MPC-FF"].groupby("seed")}
    rec = []
    for pol in ["Relay", "Relay-occ", "AO-80", "Sched", "Timer", "PI-SD", "THC", "MPC-track", "Costate-FB",
                "Rollout-CL", "MPC-RR"]:
        q = t[(t.policy == pol) & t.seed.isin(ff.keys())]
        if q.empty:
            continue
        d = np.array([float(S.interp_front(ff[r.seed], r.Efrac)) - r.captured for r in q.itertuples()])
        lo, hi = boot_ci(d)
        key = KEY.get(pol, pol.replace("-", ""))
        mac(f"fd{key}", d.mean(), "{:.3f}"); mac(f"fd{key}Up", -d.mean(), "{:.3f}"); mac(f"fd{key}Lo", lo, "{:.3f}"); mac(f"fd{key}Hi", hi, "{:.3f}")
        rec.append({"policy": pol, "dist": d.mean(), "lo": lo, "hi": hi, "n": len(d)})
    pd.DataFrame(rec).to_csv(os.path.join(RES, "e12/front_distance.csv"), index=False)
    # largest shortfall of the swept rule-based and stationary-law fronts below the EMPC-FF front (means over days)
    mean_ff = mf[mf.fam == "MPC-FF"].groupby("policy")[["Efrac", "captured"]].mean()
    nf = rc("e11/fronts.csv")
    for fam, key, src in [("THC", "THC", nf), ("TRK", "TRK", nf), ("Costate-FB", "CLs", mf)]:
        g = src[src.policy.str.startswith(fam + "|")].groupby("policy")[["Efrac", "captured"]].mean()
        g = g[(g.Efrac >= mean_ff.Efrac.min()) & (g.Efrac <= mean_ff.Efrac.max())]
        if len(g):
            gap = S.interp_front(mean_ff, g.Efrac.values) - g.captured.values
            mac(f"frontMax{key}", gap.max(), "{:.3f}")
    # paired version (Fig. S5(b)): per day, swept point minus that day's EMPC-FF front at its energy, mean over days
    ffd = {sd: g for sd, g in mf[mf.fam == "MPC-FF"].groupby("seed")}
    for fam, key, src in [("THC", "THC", nf), ("TRK", "TRK", nf), ("Costate-FB", "CLs", mf), ("MPC-FB", "FB", mf),
                          ("MPC-oracle", "OR", mf)]:
        pts = []
        for pol, q in src[src.policy.str.startswith(fam + "|")].groupby("policy"):
            q = q[q.seed.isin(ffd.keys())]
            if q.empty:
                continue
            dd = np.array([float(S.interp_front(ffd[r.seed], r.Efrac)) - r.captured for r in q.itertuples()])
            e = q.Efrac.mean()
            if 0.12 < e < 0.86:
                pts.append((e, dd.mean()))
        if pts:
            e, worst = max(pts, key=lambda p: p[1])
            mac(f"pdMax{key}", worst, "{:.3f}"); mac(f"pdMaxE{key}", 100 * e, "{:.0f}")
            above = [p[1] for p in pts if p[0] >= 0.4]
            if above:
                mac(f"pdMaxAbove{key}", max(above), "{:.3f}")


def e3_isa():
    parts = [rc("e3/budget_main.csv")] + [rc(f) for f in ["e12/budget_test.csv", "e12/budget_rcl_test.csv"] if exists(f)]
    if exists("e15/clair_budget.csv"):                     # whole-day preview under the same budget
        cb, d1 = rc("e15/clair_budget.csv"), rc("e1/main.csv")
        off = d1[d1.policy == "Off"].set_index("seed").CE
        full = d1[d1.policy == "AO-100"].set_index("seed").CE
        cb["captured"] = [(off[sd] - ce) / (off[sd] - full[sd]) for sd, ce in zip(cb.seed, cb.CE)]
        cb["Efrac"] = cb.energy_Wh / E_FULL
        parts.append(cb[["seed", "policy", "captured", "Efrac"]])
    df = pd.concat(parts, ignore_index=True)
    piv = df.pivot(index="seed", columns="policy", values="captured")
    pe = df.pivot(index="seed", columns="policy", values="Efrac")
    cl = [c for c in piv.columns if c.startswith("Clairvoyant")]
    if cl and "RCL/fixed*" in piv.columns:                  # whole-day preview against the budgeted rollout law
        dd = (piv[cl[0]] - piv["RCL/fixed*"]).dropna()
        lo, hi = boot_ci(dd)
        mac("bdClairVsRCL", dd.mean(), "{:.3f}"); mac("bdClairVsRCLLo", lo, "{:.3f}"); mac("bdClairVsRCLHi", hi, "{:.3f}")
    # grouped: rule-based, feedback-only (persistence drive), schedule forecast, perfect information
    lab = {"Relay+cutoff": "Relay with cut-off", "THC/cutoff*": "THC with cut-off (tuned)",
           "Costate-FB/fixed": "CL-s, fixed price", "RCL/fixed*": "CL-r, fixed price (tuned)",
           "MPC-FB/fixed*": "EMPC-FB, fixed price (tuned)", "MPC-FB/pacing*": "EMPC-FB, pacing (tuned)",
           "MPC-FB/rollout": "EMPC-FB, forecast price", "Costate-FB/rollout": "CL-s, forecast price (schedule)",
           "MPC-FF/fixed*": "EMPC-FF, fixed price (tuned)", "MPC-FF/rollout": "EMPC-FF, forecast price",
           "MPC-oracle/rollout": "EMPC-OR, forecast price", "Clairvoyant/budget": "Clairvoyant (whole-day preview)"}
    order = [o for o in lab if o in piv]
    fbs = [o for o in ["MPC-FB/fixed*", "MPC-FB/pacing*"] if o in piv] or [o for o in ["MPC-FB/fixed", "MPC-FB/pacing"] if o in piv]
    best = max(fbs, key=lambda o: piv[o].mean())
    mac("bestFB", lab.get(best, best), "{}")
    rows, ps, idx = [], [], []
    for pol in order:
        r = [lab[pol], f"{piv[pol].mean():.3f}", f"{pe[pol].mean():.3f}", f"{pe[pol].max():.3f}"]
        if pol == best:
            r += ["--", "--", "--"]
        else:
            dm, lo, hi, p, share = paired(piv[pol], piv[best])
            r += [f"{dm:+.3f} [{lo:+.3f}, {hi:+.3f}]", p, f"{100 * share:.0f}"]
            ps.append(p); idx.append(len(rows))
        rows.append(r)
    padj = holm(ps)
    for i, q in zip(idx, padj):
        rows[i][5] = pfmt(q)
    hdr3 = ["Policy", "$\\bar\\kappa$", "$\\bar\\varphi$", "max $\\varphi$",
            "$\\Delta\\kappa$ to EMPC-FB, fixed price [95\\% CI]", "$p$ (Holm)", "better (\\%)"]
    tex_table(os.path.join(GEN, "tab_e3_full.tex"), hdr3, rows, "lrrrlrr", label=lambda s: s)
    main3 = ["Relay+cutoff", "THC/cutoff*", "RCL/fixed*", "MPC-FB/fixed*", "MPC-FF/rollout", "MPC-oracle/rollout",
             "Clairvoyant/budget"]                     # the paper's table; the full one is in the supplement
    tex_table(os.path.join(GEN, "tab_e3.tex"), hdr3, [r for o, r in zip(order, rows) if o in main3], "lrrrlrr",
              label=lambda s: s)
    grp = {"Relay+cutoff": 0, "THC/cutoff*": 0, "Costate-FB/rollout": 2, "MPC-FF/fixed*": 2, "MPC-FF/rollout": 2,
           "MPC-oracle/rollout": 3, "Clairvoyant/budget": 3}       # rule-based, feedback only, schedule forecast, perfect
    for fn in ["tab_e3_full.tex", "tab_e3.tex"]:
        path = os.path.join(GEN, fn)
        out, prev = [], None
        for line in open(path, encoding="utf-8").read().split("\n"):
            g = next((grp.get(o, 1) for o in order if line.startswith(lab[o] + " &")), None)
            if g is not None and prev is not None and g != prev:
                out.append("\\midrule")
            prev = g if g is not None else prev
            out.append(line)
        open(path, "w", encoding="utf-8").write("\n".join(out))
    holm_of = {order[i]: q for i, q in zip(idx, padj)}
    for pol, k in [("MPC-FF/rollout", "FFroll"), ("MPC-oracle/rollout", "ORroll"), (best, "BestFB"), ("Relay+cutoff", "RelayCut"),
                   ("Costate-FB/rollout", "CFroll"), ("MPC-FB/rollout", "FBroll"), ("THC/cutoff*", "THCcut"),
                   ("RCL/fixed*", "RCLfix"), ("MPC-FF/fixed*", "FFfix"), ("MPC-FB/pacing*", "FBpace"),
                   ("Clairvoyant/budget", "Clair")]:
        if pol in piv:
            mac(f"bcap{k}", piv[pol].mean(), "{:.3f}"); mac(f"bef{k}", pe[pol].mean(), "{:.3f}")
            if pol != best:
                dm, lo, hi, p, share = paired(piv[pol], piv[best])
                mac(f"bd{k}", dm, "{:+.3f}"); mac(f"bd{k}Lo", lo, "{:+.3f}"); mac(f"bd{k}Hi", hi, "{:+.3f}")
                mac(f"bd{k}P", prel(holm_of.get(pol, p)), "{}"); mac(f"bd{k}Share", 100 * share, "{:.0f}")
    mac("bMaxEps", max(pe[o].max() for o in order if o != "Relay"), "{:.3f}")
    if "MPC-oracle/rollout" in piv and "MPC-FF/rollout" in piv:     # value of perfect information, same allocation
        dm, lo, hi, p, share = paired(piv["MPC-oracle/rollout"], piv["MPC-FF/rollout"])
        mac("bORvsFFroll", dm, "{:+.3f}"); mac("bORvsFFrollLo", lo, "{:+.3f}"); mac("bORvsFFrollHi", hi, "{:+.3f}")
    dm, lo, hi, p, share = paired(piv["MPC-FF/rollout"], piv[best])
    mac("bFFrollVsBestFB", dm, "{:+.3f}"); mac("bFFrollVsBestFBLo", lo, "{:+.3f}"); mac("bFFrollVsBestFBHi", hi, "{:+.3f}")
    mac("bFFrollVsBestFBP", prel(holm_of.get("MPC-FF/rollout", p)), "{}"); mac("bFFrollVsBestFBShare", 100 * share, "{:.0f}")
    g = piv["MPC-FF/rollout"] - piv[best]
    q = np.percentile(g, [10, 50, 90])
    mac("bGainQlo", q[0], "{:+.3f}"); mac("bGainMed", q[1], "{:+.3f}"); mac("bGainQhi", q[2], "{:+.3f}")
    # data for the budget figure
    fig = [{"policy": lab[o], "key": o, "kappa": piv[o].mean(), "lo": boot_ci(piv[o])[0], "hi": boot_ci(piv[o])[1],
            "eps": pe[o].mean()} for o in order]
    pd.DataFrame(fig).to_csv(os.path.join(RES, "e12/budget_fig.csv"), index=False)


def robust_isa():
    """One-at-a-time (E4 + E4b) and fault (robust2) configurations: J and budget comparisons with intervals."""
    parts = []
    if exists("e4/robustness.csv"):
        a = rc("e4/robustness.csv")
        if exists("e12/e4b.csv"):
            b = rc("e12/e4b.csv")
            b = b[~b.policy.isin(["Off", "AO-100"])]
            a = pd.concat([a, b], ignore_index=True)
        a["group"] = "oat"
        parts.append(a)
    if exists("e12/robust2.csv"):
        f = rc("e12/robust2.csv"); f["group"] = "fault"
        parts.append(f)
    if not parts:
        return
    df = pd.concat(parts, ignore_index=True)
    df["J"] = df.CE + RHO * df.energy_Wh
    rows, recs = [], []
    for (grp, cfg), d in df.groupby(["group", "config"], sort=False):
        pj = d.pivot(index="seed", columns="policy", values="J")
        pc = d.pivot(index="seed", columns="policy", values="captured")
        pe = d.pivot(index="seed", columns="policy", values="Efrac")
        rec = {"group": grp, "config": cfg}
        base = pj["MPC-FF"].mean()
        for a, b, k in [("MPC-FF", "Relay", "relay"), ("MPC-FF", "THC", "thc"), ("Rollout-CL", "THC", "rcl_thc"),
                        ("Rollout-CL", "MPC-FF", "rcl_ff"), ("Rollout-CL", "Relay", "rcl_relay")]:
            if a in pj and b in pj:
                dm, lo, hi, p, _ = paired(100 * (pj[a] - pj[b]) / base, np.zeros(len(pj)))
                rec.update({f"dJ_{k}": dm, f"dJ_{k}_lo": lo, f"dJ_{k}_hi": hi})
        dm, lo, hi, p, _ = paired(pc["MPC-FF"], pc["Relay"])
        rec.update({"dk_relay": dm, "dk_relay_lo": lo, "dk_relay_hi": hi})
        comp = "MPC-FB/fixed*" if "MPC-FB/fixed*" in pc else ("Costate-FB/fixed" if "Costate-FB/fixed" in pc else None)
        if comp and "MPC-FF/rollout" in pc:
            dm, lo, hi, p, _ = paired(pc["MPC-FF/rollout"], pc[comp])
            rec.update({"db": dm, "db_lo": lo, "db_hi": hi, "comp": comp,
                        "E_roll": pe["MPC-FF/rollout"].mean(), "E_fix": pe[comp].mean()})
        if comp and "RCL/fixed*" in pc:
            dm, lo, hi, p, _ = paired(pc["RCL/fixed*"], pc[comp])
            rec.update({"dbr": dm, "dbr_lo": lo, "dbr_hi": hi})
        rec.update({k + "_kap": pc[k].mean() for k in ["Relay", "THC", "MPC-FF"] if k in pc})
        rec.update({k + "_eps": pe[k].mean() for k in ["Relay", "THC", "MPC-FF"] if k in pe})
        recs.append(rec)
        def cell(k, f="{:+.2f}"):
            if k not in rec or pd.isna(rec[k]):
                return "--"
            star = "$^*$" if rec[k + "_lo"] > 0 or rec[k + "_hi"] < 0 else ""
            return texnum(f.format(rec[k])) + star
        rows.append([cfg, f"{rec['MPC-FF_kap']:.2f}/{rec['MPC-FF_eps']:.2f}",
                     cell("dJ_relay"), cell("dJ_thc"), cell("dJ_rcl_thc"), cell("dJ_rcl_ff"),
                     cell("db", "{:+.3f}"), cell("dbr", "{:+.3f}")])
    R = pd.DataFrame(recs)
    R.to_csv(os.path.join(RES, "e12/robust_summary.csv"), index=False)
    tex_table(os.path.join(GEN, "tab_e4.tex"), ["Truth / scenario", "EMPC-FF $\\kappa/\\varphi$",
              "FF$-$relay", "FF$-$THC", "CL-r$-$THC", "CL-r$-$FF", "FF forecast$-$fixed", "CL-r$-$fixed"], rows, "llrrrrrr",
              label=scenario_label)
    oat = R[R.group == "oat"]
    mac("robNcfg", len(oat), "{:d}")
    mac("robMinRelayGain", oat.dk_relay.min(), "{:+.3f}"); mac("robMaxRelayGain", oat.dk_relay.max(), "{:+.3f}")
    mac("robNsigRelay", f"{int((oat.dk_relay_lo > 0).sum())} of {len(oat)}", "{}")
    if "dJ_thc" in oat:
        mac("robNposTHC", int((oat.dJ_thc < 0).sum()), "{:d}")
        mac("robNsigTHC", int((oat.dJ_thc_hi < 0).sum()), "{:d}")
    for k, name in [("dJ_rcl_thc", "RCLTHC"), ("dJ_rcl_ff", "RCLFF"), ("dJ_rcl_relay", "RCLRelay")]:
        if k in oat and oat[k].notna().any():
            o = oat[oat[k].notna()]
            mac(f"robNpos{name}", int((o[k] < 0).sum()), "{:d}")
            mac(f"robNsig{name}", int((o[k + "_hi"] < 0).sum()), "{:d}")
            mac(f"robNsigNeg{name}", int((o[k + "_lo"] > 0).sum()), "{:d}")
            mac(f"robN{name}", len(o), "{:d}")
            mac(f"robMax{name}", o[k].max(), "{:+.2f}"); mac(f"robMin{name}", o[k].min(), "{:+.2f}")
    if "dJ_relay" in oat:
        mac("robMinJRelay", oat.dJ_relay.min(), "{:+.2f}"); mac("robMaxJRelay", oat.dJ_relay.max(), "{:+.2f}")
        mac("robNsigJRelay", int((oat.dJ_relay_hi < 0).sum()), "{:d}")
    if "dbr" in oat and oat.dbr.notna().any():
        o = oat[oat.dbr.notna()]
        mac("robNnsBudgetRCL", int(((o.dbr_lo <= 0) & (o.dbr_hi >= 0)).sum()), "{:d}")
        mac("robNposBudgetRCL", int((o.dbr > 0).sum()), "{:d}")
        mac("robNsigBudgetRCL", int((o.dbr_lo > 0).sum()), "{:d}")
        mac("robNsigNegBudgetRCL", int((o.dbr_hi < 0).sum()), "{:d}")
        assert set(o.loc[o.dbr_hi < 0, "config"]) <= {"outdoor x0.25"}, "text names only this configuration"
        mac("robBudgetRCLWorstVal", o.dbr.min(), "{:+.3f}")
    if "db" in oat:
        mac("robNsigNegBudget", int((oat.db_hi < 0).sum()), "{:d}")
        mac("robNposBudget", int((oat.db > 0).sum()), "{:d}")
        mac("robNsigBudget", f"{int((oat.db_lo > 0).sum())}", "{}")
        mac("robMinBudgetGain", oat.db.min(), "{:+.3f}"); mac("robMaxBudgetGain", oat.db.max(), "{:+.3f}")
    fl = R[R.group == "fault"]
    if len(fl) and "dJ_rcl_thc" in fl:
        mac("faultNposRCLTHC", int((fl.dJ_rcl_thc < 0).sum()), "{:d}")
        mac("faultNposRCLFF", int((fl.dJ_rcl_ff < 0).sum()), "{:d}")
        mac("faultNsigNegRCLFF", int((fl.dJ_rcl_ff_lo > 0).sum()), "{:d}")
        mac("faultNsigRCLTHC", int((fl.dJ_rcl_thc_hi < 0).sum()), "{:d}")
        mac("faultWorstRCL", fl.loc[fl.dJ_rcl_thc.idxmax(), "config"].replace("%", "\\%"), "{}")
        mac("faultWorstRCLval", fl.dJ_rcl_thc.max(), "{:+.2f}")
    if len(fl) and "dbr" in fl and "db" in fl:
        mac("faultNposBudgetRCL", int((fl.dbr > 0).sum()), "{:d}")
        mac("faultNposBudget", int((fl.db > 0).sum()), "{:d}")
    if len(fl) and "dJ_thc" in fl:
        keep = int(((fl.dJ_thc < 0) & (fl.dk_relay > 0)).sum())
        mac("faultNkeep", keep, "{:d}"); mac("faultN", len(fl), "{:d}")
        mac("faultBest", fl.loc[fl.dJ_thc.idxmin(), "config"].replace("%", "\\%"), "{}")
        mac("faultWorst", fl.loc[fl.dJ_thc.idxmax(), "config"].replace("%", "\\%"), "{}")
        if "db" in fl:
            pb = fl[fl.config.str.startswith("plan bias +")]
            if len(pb):
                mac("faultPlanBiasBudget", pb.db.iloc[0], "{:+.3f}")
            pb = fl[fl.config.str.startswith("plan bias -")]
            if len(pb):
                mac("faultPlanBiasBudgetNeg", pb.db.iloc[0], "{:+.3f}")
            mac("faultWorstBudget", fl.loc[fl.db.idxmin(), "config"].replace("%", "\\%"), "{}")
    # fan exponent overrun of the rollout policy (modelled-power guarantee)
    e4 = rc("e4/robustness.csv")
    d = e4[e4.config == "fan exponent 2.5"]
    if len(d):
        mac("robExpOver", 100 * (d[d.policy == "MPC-FF/rollout"].Efrac.mean() / 0.25 - 1), "{:.0f}")
    if exists("e12/e4b.csv"):
        d = pd.concat([e4, rc("e12/e4b.csv")], ignore_index=True)
        d = d[d.config == "fan exponent 2.5"]
        bud = d[d.policy.isin(["MPC-FF/rollout", "MPC-FB/fixed*", "RCL/fixed*", "THC/cutoff*"])]
        if len(bud):
            over = 100 * (bud.groupby("policy").Efrac.mean() / 0.25 - 1)
            mac("robExpOverMin", over.min(), "{:.0f}"); mac("robExpOverMax", over.max(), "{:.0f}")
            mac("robExpOverDay", 100 * (bud.Efrac.max() / 0.25 - 1), "{:.0f}")


def mc_isa():
    if not exists("e12/mc.csv"):
        return
    df = rc("e12/mc.csv"); df["J"] = df.CE + RHO * df.energy_Wh
    g = df.groupby(["config", "policy"]).J.mean().unstack()
    rel = 100 * (g["MPC-FF"] - g["Relay"]) / g["MPC-FF"]
    thc = 100 * (g["MPC-FF"] - g["THC"]) / g["MPC-FF"]
    rcl = 100 * (g["Rollout-CL"] - g["THC"]) / g["MPC-FF"]
    mac("mcN", len(g), "{:d}")
    mac("mcPosRelay", 100 * (rel < 0).mean(), "{:.0f}"); mac("mcPosTHC", 100 * (thc < 0).mean(), "{:.0f}")
    mac("mcMedTHC", thc.median(), "{:+.2f}"); mac("mcMedRelay", rel.median(), "{:+.2f}")
    mac("mcPosRCLTHC", 100 * (rcl < 0).mean(), "{:.0f}"); mac("mcMedRCLTHC", rcl.median(), "{:+.2f}")
    rff = 100 * (g["Rollout-CL"] - g["MPC-FF"]) / g["MPC-FF"]
    rrel = 100 * (g["Rollout-CL"] - g["Relay"]) / g["MPC-FF"]
    mac("mcPosRCLFF", 100 * (rff < 0).mean(), "{:.0f}"); mac("mcMedRCLFF", rff.median(), "{:+.2f}")
    mac("mcMaxRCLFF", rff.max(), "{:+.2f}"); mac("mcPosRCLRelay", 100 * (rrel < 0).mean(), "{:.0f}")
    pd.DataFrame({"dJ_relay": rel, "dJ_thc": thc, "dJ_rcl_thc": rcl, "dJ_rcl_ff": rff,
                  "dJ_rcl_relay": rrel}).to_csv(os.path.join(RES, "e12/mc_summary.csv"))


TAGS = {0.125: "Eighth", 0.25: "Quarter", 0.375: "ThreeEighths", 0.5: "Half", 0.75: "ThreeQuarters", 1.0: "One",
        2.0: "Two", 4.0: "Four"}


def boot_slope(tab, eps, scales, B=2000, seed=0):
    """tab: DataFrame (index day, columns scale) of per-day values; fit log(mean) ~ log(eps) over `scales`.
    Returns (slope, lo, hi), the CI from resampling days (the same days at every scale)."""
    t = tab[scales].dropna()
    x = np.log(np.asarray([eps[s] for s in scales]))
    f = lambda m: np.polyfit(x, np.log(np.maximum(m, 1e-15)), 1)[0]
    b0 = f(t.mean().values)
    rng = np.random.default_rng(seed)
    v = t.values
    bs = [f(v[rng.integers(0, len(v), len(v))].mean(axis=0)) for _ in range(B)]
    return b0, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def e15_isa():
    """Clairvoyant reference (main comparison) and the leverage sweep."""
    if exists("e15/clair_main.csv"):
        c = rc("e15/clair_main.csv")
        mac("clairUnique", 100 * np.mean(c.clair_spread <= 1e-9), "{:.1f}")
        sp = c.clair_spread.max()
        mac("clairSpreadMax", "0" if sp <= 0 else f"{sp:.1e}", "{}")
        f = frame()
        J = f.pivot(index="seed", columns="policy", values="J")
        for pol, k in [("Rollout-CL", "RCL"), ("Rollout-CL-oracle", "RCLex"), ("Rollout-CL-pen", "RCLpen"),
                       ("MPC-FF", "FF"), ("MPC-FB", "FB"),
                       ("MPC-FB-L", "FBL"),
                       ("MPC-oracle", "OR"), ("THC", "THC"), ("Relay", "Relay"), ("Costate-FB", "CF"), ("MPC-RR", "RR")]:
            if pol in J:
                dm, lo, hi, p, share = paired(100 * (J[pol] - J["Clairvoyant"]) / J["Clairvoyant"].mean(),
                                              np.zeros(len(J)))
                mac(f"dJclair{k}", dm, "{:.3f}"); mac(f"dJclair{k}Lo", lo, "{:.3f}"); mac(f"dJclair{k}Hi", hi, "{:.3f}")
        off = J["Off"].mean(); cl = J["Clairvoyant"].mean()
        mac("rangeOffClair", 100 * (off - cl) / cl, "{:.1f}")          # cost range control can address, % of J*
        mac("prThresh", (off - cl) / cl, "{:.2f}")          # practical threshold: 1 % of the range, in % of J^c
        CEp = f.pivot(index="seed", columns="policy", values="CE")   # the same threshold as a captured share at equal energy
        mac("prThreshKappa", 0.01 * (off - cl) / (CEp["Off"] - CEp["AO-100"]).mean(), "{:.3f}")
        mac("relayClairGap", 100 * (J["Relay"].mean() - cl) / cl, "{:.2f}")
        for pol, k in [("Rollout-CL", "RCL"), ("MPC-FF", "FF"), ("MPC-FB-L", "FBL"), ("THC", "THC"), ("Relay", "Relay")]:
            if pol in J:
                v = 100 * (J[pol].mean() - cl) / (off - cl)
                mac(f"rgClair{k}", v, "{:.3f}" if v < 0.1 else "{:.2f}" if v < 10 else "{:.1f}")
    if not exists("e15/leverage.csv"):
        return
    L = rc("e15/leverage.csv")
    lamf = "e15/lambda1.csv" if exists("e15/lambda1.csv") else "e15/lambda.csv"
    if exists(lamf):                                    # EMPC-Lambda (Theorem 1(b) setting), same days
        lam = rc(lamf)
        lam = lam.merge(L[["scale", "seed", "eps"]].drop_duplicates(), on=["scale", "seed"])
        L = pd.concat([L, lam], ignore_index=True)
    J = L.pivot_table(index=["scale", "seed"], columns="policy", values="J")
    eps = L.groupby("scale").eps.first()
    SMALL = sorted(s for s in eps.index if s <= 1.0)      # exponent fits: all leverages up to the testbed's
    rows = []
    for pol in [c for c in J.columns if c not in ("Off", "AO-100", "Clairvoyant")]:
        for sc in eps.index:
            j = J.loc[sc]
            r = (j[pol] - j["Clairvoyant"]) / (j["Off"] - j["Clairvoyant"])          # share of the range lost
            a = 100 * (j[pol] - j["Clairvoyant"]) / j["Clairvoyant"]                 # gap in % of J*
            lo, hi = boot_ci(r)
            rows.append({"policy": pol, "scale": sc, "eps": eps[sc], "rel": r.mean(), "rel_lo": lo, "rel_hi": hi,
                         "abs_pct": a.mean(), "abs_lo": boot_ci(a)[0], "abs_hi": boot_ci(a)[1]})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RES, "e15/leverage_summary.csv"), index=False)
    mac("levEps", eps[1.0], "{:.2f}"); mac("levEpsMin", eps.min(), "{:.2f}"); mac("levEpsMax", eps.max(), "{:.1f}")
    mac("levN", L.seed.nunique(), "{:d}")
    names = {"Relay": "Relay", "Sched": "Sched", "CL-s": "CLs", "CL-r": "RCL", "CL-r-exact": "RCLex",
             "CL-r-exact-pen": "RCLexPen",
             "EMPC-FB": "FB", "EMPC-oracle": "OR", "EMPC-FB-L": "FBL", "EMPC-oracle-L": "ORL"}
    for sc, tag in TAGS.items():
        if sc in eps.index:
            mac(f"levEps{tag}", eps[sc], "{:.2f}")
    mac("levNEps", len(SMALL), "{:d}")
    for pol, k in names.items():
        d = R[R.policy == pol].sort_values("eps")
        if d.empty:
            continue
        for sc, tag in TAGS.items():
            x = d[d.scale == sc]
            if len(x):
                v = 100 * x.rel.iloc[0]
                mac(f"lev{k}{tag}", v, "{:.1e}" if v < 0.0001 else "{:.4f}" if v < 0.01 else "{:.3f}" if v < 0.1 else "{:.2f}" if v < 10 else "{:.1f}")
                a = x.abs_pct.iloc[0]
                mac(f"levAbs{k}{tag}", a, "{:.1e}" if a < 0.0001 else "{:.4f}" if a < 0.01 else "{:.3f}" if a < 1 else "{:.2f}")
        ok = d.rel > 0
        if ok.sum() >= 3:
            b = np.polyfit(np.log(d.eps[ok]), np.log(d.rel[ok]), 1)[0]
            mac(f"levSlope{k}", b, "{:.1f}")
        small = d[(d.scale <= 1.0) & (d.abs_pct > 0)]          # asymptotic range: up to the testbed leverage
        if len(small) >= 3:
            mac(f"levAbsSlope{k}", np.polyfit(np.log(small.eps), np.log(small.abs_pct), 1)[0], "{:.1f}")
            mac(f"levRelSlope{k}", np.polyfit(np.log(small.eps), np.log(small.rel), 1)[0], "{:.1f}")
    small3 = SMALL
    for pol, k in [("CL-r-exact", "RCLex"), ("CL-r-exact-pen", "RCLexPen"), ("EMPC-oracle-L", "ORL")]:
        if pol in J.columns:
            tab = (100 * (J[pol] - J["Clairvoyant"]) / J["Clairvoyant"]).unstack("scale")
            b, lo, hi = boot_slope(tab, eps, small3)
            mac(f"levAbsSlope{k}", b, "{:.1f}"); mac(f"levAbsSlope{k}Lo", lo, "{:.1f}"); mac(f"levAbsSlope{k}Hi", hi, "{:.1f}")
    if "EMPC-FB-L" in J.columns:                    # gain from optimisation over a horizon: CL-r minus EMPC-Lambda (FB)
        for sc, tag in [(0.5, "Half"), (1.0, "One"), (2.0, "Two")]:
            j = J.loc[sc]
            dlt = 100 * (j["CL-r"] - j["EMPC-FB-L"]) / (j["Off"] - j["Clairvoyant"])
            lo, hi = boot_ci(dlt)
            mac(f"hz{tag}", dlt.mean(), "{:.3f}"); mac(f"hz{tag}Lo", lo, "{:.3f}"); mac(f"hz{tag}Hi", hi, "{:.3f}")
            mac(f"hz{tag}Share", 100 * np.mean(dlt > 0), "{:.0f}")
    spread = L[L.policy == "Clairvoyant"].groupby("scale").clair_spread
    mac("levUniqueOne", 100 * np.mean(L[(L.policy == "Clairvoyant") & (L.scale == 1.0)].clair_spread <= 1e-9), "{:.0f}")
    mac("levSpreadFour", 100 * spread.max()[4.0] / J.loc[4.0]["Clairvoyant"].mean(), "{:.3f}")
    mac("levEpsHalf", eps[0.5], "{:.2f}"); mac("levEpsTwo", eps[2.0], "{:.2f}")
    if exists("e15/lambda.csv") and exists("e15/lambda1.csv"):    # earlier EMPC variant: uniform 5-min blocks, penalty
        old = rc("e15/lambda.csv")
        old = old[old.policy == "EMPC-oracle-L"].pivot_table(index=["scale", "seed"], values="J")["J"]
        ref = L[L.policy.isin(["Clairvoyant", "Off"])].pivot_table(index=["scale", "seed"], columns="policy", values="J")
        r = (100 * (old - ref["Clairvoyant"]) / (ref["Off"] - ref["Clairvoyant"])).dropna()
        for sc, tag in [(0.25, "Quarter"), (1.0, "One")]:
            mac(f"levOldORL{tag}", r.loc[sc].mean(), "{:.3f}")
    if exists("e15/blocks_test.csv"):               # ablation: first block and soft-bound penalty (exact information)
        A = rc("e15/blocks_test.csv").pivot_table(index=["scale", "seed"], columns="policy", values="J")
        ref = L[L.policy.isin(["Clairvoyant", "Off"])].pivot_table(index=["scale", "seed"], columns="policy", values="J")
        A = A.join(ref, how="inner")
        for pol, k in [("L-uniform", "Uni"), ("L-first1", "First"), ("L-uniform-M0", "UniMz"), ("L-first1-M0", "FirstMz")]:
            r = 100 * (A[pol] - A["Clairvoyant"]) / (A["Off"] - A["Clairvoyant"])
            for sc, tag in [(0.25, "Quarter"), (1.0, "One")]:
                mac(f"abl{k}{tag}", r.loc[sc].mean(), "{:.4f}" if r.loc[sc].mean() < 0.01 else "{:.3f}")
        mac("ablN", A.index.get_level_values(1).nunique(), "{:d}")
    if exists("e15/cert.csv"):                      # optimality certificate (Theorem 1(a))
        C = rc("e15/cert.csv")
        C["LB"] = np.maximum(C.LB, C.Jrex - C.beta_rex)         # best certified lower bound of the two schedules
        C["cg"] = 100 * (C.Jc - C.LB) / C.Jc                    # certified bound on J^c - J*, % of J^c
        C["cgr"] = 100 * (C.Jc - C.LB) / (C.Joff - C.LB)        # same, % of the controllable range
        C["br"] = 100 * C.beta_rex / C.Jc                        # CL-r (exact) certificate from its own record
        C["brr"] = 100 * C.beta_rex / (C.Joff - C.LB)
        C["rg"] = 100 * (C.Jrex - C.Jc) / C.Jc
        g = C.groupby("scale")
        for sc, tag in TAGS.items():
            if sc not in g.groups:
                continue
            x = g.get_group(sc)
            for col, nm in [("cg", "cert"), ("cgr", "certRel"), ("br", "certRex"), ("brr", "certRexRel")]:
                v, w = x[col].mean(), x[col].max()
                fmt = "{:.4f}" if w < 0.01 else "{:.3f}" if w < 1 else "{:.2f}" if w < 10 else "{:.1f}"
                mac(f"{nm}{tag}", v, fmt); mac(f"{nm}{tag}Max", w, fmt)
            mac(f"certRatio{tag}", x.br.mean() / max(x.rg.mean(), 1e-12), "{:.0f}")
        mac("certN", int((C.scale == 1.0).sum()), "{:d}")
        sw = C[C.seed.isin(L.seed.unique())]              # the leverage-sweep days only, for the sweep's text
        for sc, tag in TAGS.items():
            x = sw[sw.scale == sc]
            if len(x):
                w = x.cg.max()
                mac(f"certSw{tag}Max", w, "{:.4f}" if w < 0.01 else "{:.3f}" if w < 1 else "{:.2f}")
                mac(f"certRexSw{tag}", x.br.mean(), "{:.3f}")
                mac(f"certRatioSw{tag}", x.br.mean() / max(x.rg.mean(), 1e-12), "{:.0f}")
        if exists("e15/cert_lp.csv"):                 # LP certificate on the first test days
            C = C.merge(rc("e15/cert_lp.csv")[["scale", "seed", "LB_lp"]], on=["scale", "seed"], how="left")
            mac("certLPN", int(C.LB_lp.notna().sum()), "{:d}")
            mac("certLPTried", len(rc("e15/cert_lp.csv")), "{:d}")
        if "LB_lp" in C.columns:                     # LP relaxation of the exact MILP (convex-hull form)
            C["lp"] = 100 * (C.Jc - C.LB_lp) / C.Jc
            for sc, tag in [(0.25, "Quarter"), (1.0, "One")]:
                x = C[(C.scale == sc) & C.lp.notna()]
                if len(x):
                    fmt = "{:.4f}" if x.lp.max() < 0.01 else "{:.3f}"
                    mac(f"certLP{tag}", x.lp.mean(), fmt); mac(f"certLP{tag}Max", x.lp.max(), fmt)
                    mac(f"certLPRatio{tag}", (x.cg / x.lp.clip(lower=1e-9)).median(), "{:.0f}")
                    mac(f"certLPRex{tag}", (100 * (x.Jrex - np.maximum(x.LB_lp, x.LB)) / x.Jc).mean(), fmt)
        tab = C[C.seed.isin(L.seed.unique())].pivot(index="seed", columns="scale", values="br")
        b, lo, hi = boot_slope(tab, eps, [s for s in SMALL if s in tab.columns])
        mac("certSlope", b, "{:.1f}"); mac("certSlopeLo", lo, "{:.1f}"); mac("certSlopeHi", hi, "{:.1f}")
        C[["scale", "seed", "cg", "cgr", "br", "brr", "rg"]].to_csv(os.path.join(RES, "e15/cert_summary.csv"), index=False)
    if exists("e15/milpcheck.csv"):                  # exact MILP optimum on two-hour windows vs the local search
        M = rc("e15/milpcheck.csv")
        ok = M[M.status == 0]
        mac("milpWinN", len(M), "{:d}"); mac("milpWinSolved", len(ok), "{:d}")
        mac("milpWinMaxDiff", float(((ok.Jc - ok.Jmilp) / ok.Jmilp).abs().max()) if len(ok) else float("nan"), "{:.0e}")
        mac("milpWinMedT", float(ok.t_milp.median()) if len(ok) else float("nan"), "{:.0f}")
        un = M[M.status != 0]
        mac("milpWinUnsolved", len(un), "{:d}")
        mac("milpWinLSBetter", int(((un.Jc - un.Jmilp) / un.Jmilp <= 1e-8).sum()), "{:d}")
    if exists("e15/budget_sweep.csv"):              # budget: clairvoyant vs causal relays, by leverage
        B = rc("e15/budget_sweep.csv")
        bsc = sorted(B.scale.unique())
        gaps = []
        P = {sc: B[B.scale == sc].pivot(index="seed", columns="policy", values="captured") for sc in bsc}
        C = {sc: B[B.scale == sc].pivot(index="seed", columns="policy", values="CE") for sc in bsc}
        has_price = all("CL-r-exact/price*" in P[sc] for sc in bsc)
        for sc in bsc:
            tag, x = TAGS[sc], P[sc]
            g = x["Clairvoyant/budget"] - x["CL-r-exact/fixed*"]
            lo, hi = boot_ci(g)
            mac(f"bsw{tag}", g.mean(), "{:.3f}"); mac(f"bsw{tag}Lo", lo, "{:.3f}"); mac(f"bsw{tag}Hi", hi, "{:.3f}")
            mac(f"bswCap{tag}", x["CL-r-exact/fixed*"].mean(), "{:.3f}")
            mac(f"bswCapClair{tag}", x["Clairvoyant/budget"].mean(), "{:.3f}")
            gaps.append(g)
            mac(f"bswE{tag}", B[(B.scale == sc) & (B.policy == "CL-r-exact/fixed*")].energy_Wh.mean(), "{:.1f}")
            mac(f"bswEClair{tag}", B[(B.scale == sc) & (B.policy == "Clairvoyant/budget")].energy_Wh.mean(), "{:.1f}")
            if has_price:                            # value of knowing the day's price vs the rest of whole-day preview
                dp = x["CL-r-exact/price*"] - x["CL-r-exact/fixed*"]
                dr = x["Clairvoyant/budget"] - x["CL-r-exact/price*"]
                mac(f"bswPrice{tag}", dp.mean(), "{:.3f}"); mac(f"bswRest{tag}", dr.mean(), "{:.4f}" if abs(dr.mean()) < 0.01 else "{:.3f}")
                mac(f"bswPriceShare{tag}", 100 * dp.mean() / g.mean(), "{:.0f}")
                mac(f"bswEPrice{tag}", B[(B.scale == sc) & (B.policy == "CL-r-exact/price*")].energy_Wh.mean(), "{:.1f}")
        mac("bswPos", 100 * np.mean(np.concatenate([g.values > 0 for g in gaps])), "{:.0f}")
        mac("bswMin", min(g.mean() for g in gaps), "{:.3f}"); mac("bswMax", max(g.mean() for g in gaps), "{:.3f}")
        mac("bswN", B.seed.nunique(), "{:d}")
        mac("bswNEps", len(bsc), "{:d}")
        fits = [("bswSlope", "`CL-r-exact/fixed*` - `Clairvoyant/budget`")]
        if has_price:
            fits += [("bswSlopePrice", "`CL-r-exact/fixed*` - `CL-r-exact/price*`"),
                     ("bswSlopeRest", "`CL-r-exact/price*` - `Clairvoyant/budget`")]
        for nm, expr in fits:
            tab = pd.DataFrame({sc: C[sc].eval(expr) for sc in bsc})
            b, lo, hi = boot_slope(tab, eps, bsc)
            mac(nm, b, "{:.2f}"); mac(nm + "Lo", lo, "{:.2f}"); mac(nm + "Hi", hi, "{:.2f}")
    if exists("e15/cert_lp_all.csv") and exists("e15/cert.csv"):    # LP certificate on all test days
        A = rc("e15/cert_lp_all.csv").merge(rc("e15/cert.csv")[["scale", "seed", "Jc", "LB", "Jrex", "beta_rex"]],
                                            on=["scale", "seed"])
        A["LBbox"] = np.maximum(A.LB, A.Jrex - A.beta_rex)
        ok = A[A.LB_lp.notna()]
        g = 100 * (ok.Jc - ok.LB_lp) / ok.Jc
        mac("certLPAllN", len(ok), "{:d}"); mac("certLPAllTried", len(A), "{:d}")
        mac("certLPAll", g.mean(), "{:.4f}" if g.max() < 0.01 else "{:.3f}")
        mac("certLPAllMax", g.max(), "{:.4f}" if g.max() < 0.01 else "{:.3f}")
        mac("certLPAllRatio", ((ok.Jc - ok.LBbox) / (ok.Jc - ok.LB_lp).clip(lower=1e-12)).median(), "{:.0f}")
        rg = 100 * (ok.Jrex - ok.LB_lp) / ok.Jc                # gap of CL-r (exact) to J* certified by the LP
        mac("certLPAllRex", rg.mean(), "{:.4f}" if rg.max() < 0.01 else "{:.3f}")
        mac("certLPAllRexMax", rg.max(), "{:.4f}" if rg.max() < 0.01 else "{:.3f}")
        if exists("e15/cert_lp.csv"):
            o = ok.merge(rc("e15/cert_lp.csv")[["seed", "LB_lp"]], on="seed", suffixes=("", "_old")).dropna()
            mac("certLPAllAgree", float(((o.LB_lp - o.LB_lp_old) / o.LB_lp_old).abs().max()), "{:.0e}")
    a0_isa()
    e16_isa(eps)


def a0_isa():
    """Infiltration rate a0 (the parameter that sets eps): gaps to the clairvoyant schedule per room."""
    if not exists("e15/a0_sweep.csv"):
        return
    A = rc("e15/a0_sweep.csv")
    J = A.pivot_table(index=["a0", "seed"], columns="policy", values="J")
    rows = []
    tagA = {0.25: "Quarter", 0.5: "Half", 1.0: "One", 1.5: "OneHalf", 2.0: "Two"}
    cl = A[A.policy == "Clairvoyant"].set_index(["a0", "seed"])
    for a0 in sorted(A.a0.unique()):
        j = J.loc[a0]
        rng = (j["Off"] - j["Clairvoyant"]) / j["Clairvoyant"]
        e = A[A.a0 == a0].eps.iloc[0]
        r = {"a0": a0, "eps": e, "range": 100 * rng.mean()}
        for pol, k in [("Relay", "Relay"), ("THC", "THC"), ("CL-r", "RCL"), ("EMPC-FB-L", "FBL"), ("CL-r-exact", "RCLex")]:
            r[k] = (100 * (j[pol] - j["Clairvoyant"]) / j["Clairvoyant"]).mean()
            mac(f"aZ{k}{tagA[a0]}", r[k], "{:.4f}" if abs(r[k]) < 0.01 else "{:.3f}" if abs(r[k]) < 1 else "{:.2f}")
        c = cl.loc[a0]
        r["cert"] = (100 * (c.clair_internal - c.LB) / c.clair_internal).max()
        r["hz"] = (100 * (j["CL-r"] - j["EMPC-FB-L"]) / (j["Off"] - j["Clairvoyant"])).mean()
        mac(f"aZEps{tagA[a0]}", e, "{:.2f}"); mac(f"aZCert{tagA[a0]}", r["cert"], "{:.3f}")
        mac(f"aZRange{tagA[a0]}", r["range"], "{:.1f}")
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RES, "e15/a0_summary.csv"), index=False)
    f = lambda v: f"{v:.4f}" if abs(v) < 0.01 else f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"
    tex_table(os.path.join(GEN, "tab_a0.tex"),
              ["$a_0$ (h$^{-1}$)", "$\\varepsilon$", "Range", "Relay", "THC", "CL-r", "EMPC-$\\Lambda$", "CL-r, exact", "$\\beta$, max"],
              [[f"{r.a0:.2f}", f"{r.eps:.2f}", f"{r['range']:.1f}", f(r.Relay), f(r.THC), f(r.RCL), f(r.FBL), f(r.RCLex),
                f(r.cert)] for _, r in R.iterrows()], "rrrrrrrrr", label=lambda s: s)
    mac("aZEpsMin", R.eps.min(), "{:.2f}"); mac("aZEpsMax", R.eps.max(), "{:.2f}")
    mac("aZN", A.seed.nunique(), "{:d}")
    mac("aZRCLMax", R.RCL.max(), "{:.4f}"); mac("aZTHCMin", R.THC.min(), "{:.2f}"); mac("aZTHCMax", R.THC.max(), "{:.2f}")
    mac("aZRelayMin", R.Relay.min(), "{:.2f}"); mac("aZRelayMax", R.Relay.max(), "{:.2f}")
    mac("aZFBLMax", R.FBL.max(), "{:.4f}"); mac("aZCertMax", R.cert.max(), "{:.3f}")
    mac("aZHzMax", R.hz.max(), "{:.3f}")


def e16_isa(eps):
    """Direct test of Theorem 1 on its own sampled model (plant, model and cost on the one-minute grid)."""
    if not exists("e16/sampled.csv"):
        return
    S = rc("e16/sampled.csv")
    mac("smpN", S.seed.nunique(), "{:d}")
    sm = S[S.scale <= 1.0]
    mac("smpZeroEL", int(((sm["J_EL-exact"] - sm.Jc).abs() < 1e-9).sum()), "{:d}")
    mac("smpCasesSmall", len(sm), "{:d}")
    for pol, k in [("RL", "RL"), ("EL-exact", "EL"), ("EL-pers", "ELp"), ("EM-stat", "EMs")]:
        S["rel"] = 100 * (S["J_" + pol] - S.Jc) / (S.Joff - S.Jc)
        S["abs"] = 100 * (S["J_" + pol] - S.Jc) / S.Jc
        g = S.groupby("scale").rel.mean()
        for sc, tag in TAGS.items():
            if sc in g.index:
                v = g[sc]
                mac(f"smp{k}{tag}", v, "{:.5f}" if v < 0.001 else "{:.4f}" if v < 0.01 else "{:.3f}" if v < 1 else "{:.2f}")
        b, lo, hi = boot_slope(S.pivot(index="seed", columns="scale", values="abs"), eps,
                               sorted(x for x in S.scale.unique() if x <= 1.0))
        mac(f"smpSlope{k}", b, "{:.1f}"); mac(f"smpSlope{k}Lo", lo, "{:.1f}"); mac(f"smpSlope{k}Hi", hi, "{:.1f}")
    S["cg"] = 100 * (S.Jc - S.LB) / (S.Joff - S.LB)
    for sc, tag in [(0.25, "Quarter"), (1.0, "One")]:
        x = S[S.scale == sc]
        mac(f"smpCert{tag}", x.cg.mean(), "{:.4f}" if x.cg.mean() < 0.01 else "{:.3f}")
        mac(f"smpCert{tag}Max", x.cg.max(), "{:.4f}" if x.cg.max() < 0.01 else "{:.3f}")


def e13_isa():
    if exists("e13/rollout_check.json"):
        s = rj("e13/rollout_check.json")
        mac("rcN", s["0.8"]["n"], "{:d}"); mac("rcViol", s["0.8"]["n_violations"] + s["1.0"]["n_violations"], "{:d}")
        mac("rcMinImp", 100 * s["0.8"]["min_rel_improvement"], "{:.2f}")
        mac("rcMedImp", 100 * s["0.8"]["median_rel_improvement"], "{:.2f}")
        mac("rcMinImpHundred", 100 * s["1.0"]["min_rel_improvement"], "{:.2f}")
    if exists("e13/invariance.json"):
        s = rj("e13/invariance.json")
        mac("invCov", 100 * s["coverage"], "{:.0f}"); mac("invAgree", 100 * s["agreement_when_invariant"], "{:.0f}")
        mac("invPen", 100 * s["share_penalty_active"], "{:.1f}")


def solver_isa():
    if exists("e6/timing_scaling.csv"):            # speed-up of branch and bound (bound b) over full enumeration
        g = rc("e6/timing_scaling.csv").groupby(["factor", "value", "mode"]).time_us.median().unstack()
        mac("bbSpeedNuEight", g.loc[("Nu", 8), 0] / g.loc[("Nu", 8), 2], "{:.0f}")
        mac("bbSpeedDutySix", g.loc[("nu", 6), 0] / g.loc[("nu", 6), 2], "{:.1f}")
    if exists("e6/methods_summary.json"):
        s = rj("e6/methods_summary.json")
        mac("rrFirstDiff", 100 * (1 - s["rr"]["first_move_agreement"]), "{:.0f}")
    if exists("e6/milp_summary.json"):
        s = rj("e6/milp_summary.json")
        mac("milpN", s["n"], "{:d}"); mac("milpMedMs", s["median_t_milp_ms"], "{:.0f}"); mac("milpPctS", s["p95_t_milp_ms"] / 1000, "{:.1f}")
        mac("milpMaxGap", s["max_rel_gap"], "{:.1e}")
    if exists("e6/opcount_summary.json"):
        s = rj("e6/opcount_summary.json")
        mx = max(v["max_flops"] for k, v in s.items() if k in ("0", "1", "2"))
        mac("opMaxM", mx / 1e6, "{:.1f}"); mac("opMedAffM", s["2"]["median_flops"] / 1e6, "{:.1f}")
        mac("opMaxSec", mx / 1e7, "{:.2f}")


def fskill_isa():
    if exists("e14/skill.csv"):
        s = rc("e14/skill.csv")
        n = s[(s.scenario == "nominal") & (s.channel == "PM2.5 zone A")].set_index("lead_min")
        b = s[(s.scenario == "plan bias +20 min") & (s.channel == "PM2.5 zone A")].set_index("lead_min")
        mac("fskOne", 100 * n.skill_vs_persistence[1], "{:+.0f}")
        mac("fskFifteen", 100 * n.skill_vs_persistence[15], "{:.0f}"); mac("fskSixty", 100 * n.skill_vs_persistence[60], "{:.0f}")
        mac("fskBiasFifteen", 100 * b.skill_vs_persistence[15], "{:+.0f}"); mac("fskBiasSixty", 100 * b.skill_vs_persistence[60], "{:+.0f}")


def e7_isa():
    df = rc("e7/budget_formulation.csv")
    df["form"] = df.policy.str.split("|").str[0]
    if exists("e12/e7_rollout.csv"):
        r = rc("e12/e7_rollout.csv"); r["form"] = "budget state + forecast price"
        df = pd.concat([df, r], ignore_index=True)
    df["over"] = df.energy_Wh / df.budget_Wh - 1
    for f, k in [("budget-state", "BS"), ("per-horizon cap", "PH")]:
        d = df[df.form == f]
        mac(f"over{k}Max", 100 * d.over.max(), "{:+.1f}")
        mac(f"over{k}Share", 100 * (d.over > 1e-9).mean(), "{:.0f}")
    rows = []
    for (f, b), d in df.groupby(["form", "budget_Wh"], sort=False):
        rows.append([f.replace("budget-state", "budget state"), f"{b:.1f}", f"{d.energy_Wh.mean():.1f}",
                     f"{d.energy_Wh.max():.1f}", f"{100 * (d.over > 1e-9).mean():.0f}", f"{d.CE.mean():.2f}"])
    tex_table(os.path.join(GEN, "tab_e7.tex"), ["Formulation", "$E_B$", "Mean $E$", "Max $E$", "Over (\\%)", "CE"],
              rows, "lrrrrr")



def platform_isa():
    """Sensor characterisation table without the TVOC row (the monitor's TVOC channel could not be verified)."""
    d = rc("platform/calibration_check.csv")
    d = d[(d.subset == "all 11 h") & (d.pollutant != "TVOC")]
    rows = [[r.pollutant, f"{r.MAE:.2f}", f"{r.bias:+.2f}", f"{r['MAPE_%']:.1f}", f"{r['max_rel_err_%']:.0f}",
             f"{r.F10:.2f}", f"{r.r:.2f}"] for _, r in d.iterrows()]
    tex_table(os.path.join(GEN, "tab_cal.tex"), ["Pollutant", "MAE", "Bias", "MAPE (\\%)", "Max. rel. err. (\\%)",
              "$F_{\\pm10}$", "$r$"], rows, "lrrrrrr", label=lambda p: S.POLL_TEX.get(p, p))
    if exists("platform/outdoor_check.json"):        # measured outdoor PM2.5 (U.S. Embassy Dhaka)
        o = rj("platform/outdoor_check.json")
        mac("outN", o["working_days"], "{:d}"); mac("outOverlap", o["n_days_overlap"], "{:d}")
        mac("outR", o["r"], "{:.5f}"); mac("outMed", o["day_mean_median"], "{:.0f}")
        mac("outPfive", o["day_mean_p5"], "{:.0f}"); mac("outPninetyfive", o["day_mean_p95"], "{:.0f}")
        mac("outHours", o["relay_hours"], "{:d}")


def supp_tables():
    """Supplementary tables: tuning grids and optima of the baselines; forecast skill against persistence."""
    rows = []
    def best(path):
        g = rc(path); k = g.loc[g.J.idxmin()]
        return k["key"] if "key" in g.columns else g.J.idxmin(), float(g.J.min()), len(g)
    specs = [("Threshold + hysteresis, raw input", ["e11/grid_thc.csv", "e11/grid_thc_refine.csv"],
              r"$\theta_1\in\{10..60\}$, $\Delta\theta\in\{5..120\}$, band $\in\{0..20\}$, gating on/off"),
             ("Threshold + hysteresis, 60-s mean", ["e12/grid_thcavg.csv"],
              r"$\theta_1\in\{15..35\}$, $\Delta\theta\in\{20..80\}$, band $\in\{0,2.5,5\}$, gated"),
             ("Scheduled duty", ["e12/grid_sched.csv"], r"$u\in\{0.8,0.9,1\}$, pre-purge $\in\{0..60\}$\,min"),
             ("PI, sigma--delta", ["e12/grid_pisd.csv", "e12/grid_pisd2_0.csv"],
              r"setpoint $\in\{4..35\}$, $K_\mathrm{P}\in\{0.06..2\}$, $K_\mathrm{I}\in\{0..6\}$"),
             ("Timer + relay", ["e11/grid_timer.csv", "e11/grid_timer_refine.csv"], r"lead $\in\{20..90\}$, tail $\in\{0..120\}$\,min, $u\in\{0.8,0.9\}$"),
             ("Zone-tracking MPC", ["e11/grid_track.csv", "e11/grid_track_refine.csv"], r"setpoint $\in\{5..35\}$, tracking weight $\in\{0.003..10\}$")]
    for name, paths, grid in specs:
        gs = [rc(p) for p in paths if exists(p)]
        if not gs:
            continue
        g = pd.concat(gs)
        i = g.J.idxmin() if g.index.is_unique else g.reset_index(drop=True).J.idxmin()
        g = g.reset_index(drop=True)
        k = g.loc[g.J.idxmin()]
        key = k["key"] if "key" in g.columns else ""
        def fmt(a, b):
            if a in ("lead", "tail", "pre"):
                return f"{a} {60 * float(b):.0f} min"
            if isinstance(b, bool):
                return "gated" if b else "ungated"
            if a in ("u", "u_ff"):
                return f"duty {100 * float(b):.0f}\\,\\%"
            names = {"r25": "setpoint", "kp": "$K_\\mathrm{P}$", "ki": "$K_\\mathrm{I}$", "th1": "$\\theta_1$", "dth": "$\\Delta\\theta$",
                     "hyst": "band", "avg": "mean over", "q": "tracking weight", "u_base": "base duty"}
            a = names.get(a, a)
            return f"{a} {b:g}" if isinstance(b, (int, float)) else f"{a} {b}"
        try:
            kd = json.loads(key)
            keystr = ", ".join(fmt(a, b) for a, b in kd.items() if a not in ("r10", "gate") or isinstance(b, bool))
        except Exception:
            keystr = str(key)
        parts = keystr.split("$")
        keystr = "$".join(q if i % 2 else q.replace("_", r"\_") for i, q in enumerate(parts))   # escape outside math only
        rows.append([name, grid, len(g), keystr, f"{k.J:.4f}"])
    if rows:
        tex_table(os.path.join(GEN, "tab_grids.tex"), ["Baseline", "Grid", "Points", "Optimum", "$J$ (tuning)"],
                  [[r[0], r[1], str(r[2]), r[3], r[4]] for r in rows], "p{3.2cm}p{5.4cm}rp{4.2cm}r", label=lambda s: s)
    if exists("e14/skill.csv"):
        sk = rc("e14/skill.csv")
        rows = []
        for (sc, ch), d in sk.groupby(["scenario", "channel"], sort=False):
            for _, r in d.iterrows():
                rows.append([f"{sc}, {ch}".replace("PM2.5", "PM$_{2.5}$").replace("+20", "$+$20"), f"{int(r.lead_min)}",
                             f"{r.persistence:.1f}", f"{r.schedule:.1f}", f"{100 * r.skill_vs_persistence:+.0f}"])
        nom = sk[(sk.scenario == sk.scenario.iloc[0]) & (sk.lead_min >= 30)].skill_vs_persistence * 100
        bia = sk[(sk.scenario != sk.scenario.iloc[0]) & (sk.lead_min >= 60)].skill_vs_persistence * 100
        mac("skillNomMin", nom.min(), "{:.0f}"); mac("skillNomMax", nom.max(), "{:.0f}")
        mac("skillBiasMin", bia.min(), "{:.0f}"); mac("skillBiasMax", bia.max(), "{:.0f}")
        tex_table(os.path.join(GEN, "tab_skill.tex"), ["Scenario, channel", "Lead (min)", "RMSE persistence",
                  "RMSE schedule", r"Skill (\%)"], rows, "lrrrr", label=lambda s: s)

if __name__ == "__main__":
    for f in [S.e1, S.e1b, S.example, S.e2, S.e5, S.e6, S.e8, S.e9, S.e10, S.platform, S.e11, S.e6m,
              e1_isa, fronts_isa, e3_isa, robust_isa, mc_isa, e15_isa, e13_isa, solver_isa, fskill_isa, e7_isa, platform_isa, supp_tables]:
        try:
            f()
            print(f.__name__, "ok")
        except (FileNotFoundError, KeyError, ValueError) as ex:
            print(f.__name__, "skipped:", repr(ex))
    with open(os.path.join(GEN, "macros.tex"), "w", encoding="utf-8") as fh:
        fh.write("% generated by code/analysis/summarize_isa.py -- do not edit\n")
        for k, v in sorted(S.MAC.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{texnum(v)}}}\n")
    print(len(S.MAC), "macros written")

