"""Statistics, LaTeX tables and number macros for the manuscript, generated from results/.
Every number quoted in the manuscript comes from manuscript/generated/macros.tex."""
import os, sys, json, re
import numpy as np
import pandas as pd
from scipy import stats

QUICK = os.environ.get("QUICK") == "1"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results_quick" if QUICK else "results")
GEN = os.path.join(ROOT, "manuscript", "generated")
os.makedirs(GEN, exist_ok=True)
MAC = {}
rng = np.random.default_rng(0)
BSL = chr(92)


def mac(name, val, fmt="{:.2f}"):
    assert name.isalpha(), f"invalid LaTeX macro name: {name}"
    MAC[name] = fmt.format(val) if not isinstance(val, str) else val
    if not isinstance(val, str) and "+" in fmt:
        # unsigned variant for running text whose wording already carries the direction ("more", "less")
        MAC[name + "Abs"] = fmt.replace("+", "").format(abs(val))


def boot_ci(x, n=4000):
    # fixed resampling seed: the same data give the same interval wherever it is quoted (text, tables, supplement)
    x = np.asarray(x, dtype=float)
    idx = np.random.default_rng(2024).integers(0, len(x), (n, len(x)))
    m = x[idx].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def perm_test(d, n=20000):
    """Two-sided paired sign-flip permutation test of the mean difference (H0: zero mean, symmetric)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if not np.any(d != 0):
        return 1.0
    obs = abs(d.mean())
    prng = np.random.default_rng(12345)
    signs = prng.choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(null >= obs - 1e-15)) / (n + 1))


def paired(a, b):
    """Mean of a-b, bootstrap 95% CI of the mean, two-sided paired permutation p of the mean, share a>b.
    (Test and interval refer to the same quantity, the mean paired difference.)"""
    d = np.asarray(a) - np.asarray(b)
    lo, hi = boot_ci(d)
    p = perm_test(d)
    return d.mean(), lo, hi, p, float(np.mean(d > 0))


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def prel(p):
    """p-value with its relation sign, for running text: =0.072 or <0.001."""
    return "<0.001" if p < 0.001 else f"={p:.3f}"


def prel_le(p):
    return "<0.001" if p < 0.001 else BSL + f"le {p:.3f}"


def texnum(s):
    """Typeset numbers in a generated string: 4.0e-16 -> 4.0x10^-16, a signed zero loses its sign,
    and a hyphen before a digit becomes a true minus (valid in text and in math mode)."""
    s = re.sub(r"(\d(?:\.\d+)?)e([+-])0*(\d+)",
               lambda m: f"{BSL}ensuremath{{{m.group(1)}{BSL}times10^{{{'-' if m.group(2) == '-' else ''}{m.group(3)}}}}}", s)
    s = re.sub(r"(?<![\w.])[+-](0(?:\.0+)?)(?![0-9.])", r"\1", s)
    return re.sub(r"(?<![\w{^.-])-(?=\d)", lambda m: BSL + "ensuremath{-}", s)


def tex_table(path, header, rows, colspec, caption=None, label=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n" + " & ".join(header) + "\\\\\n\\midrule\n")
        for r in rows:
            if label is not None:
                c0 = label(r[0])
            else:
                c0 = re.sub(r"(?<!\\)%", r"\\%", r[0].replace("_", "\\_").replace("+-", "$\\pm$"))
                c0 = re.sub(r"(?<!\\)#", r"\\#", c0)
            r = [c0] + [texnum(c) for c in r[1:]]
            f.write(" & ".join(r) + "\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def scenario_label(c):
    """LaTeX label of a robustness configuration (same wording as fig_robust)."""
    c = (c.replace("lam0", "$a_0$").replace("k_dep", "$k_p$").replace("eta_PM", "$\\eta_\\mathrm{PM}$")
          .replace("Q x", "$\\bar Q\\times$").replace(" x", " $\\times$").replace("+-30%", "$\\pm$30\\,\\%")
          .replace("eCO2", "eCO$_2$").replace("#", "\\#"))
    c = re.sub(r"\s*(?<!\\)%", r"\\,\\%", c)        # e.g. "sensor drift +2 %/h": an unescaped % would comment out the row end
    return re.sub(r"\((\d+)-(\d+)\)", r"(\1--\2)", c)


POLL_TEX = {"PM2.5": "PM$_{2.5}$", "PM10": "PM$_{10}$", "NO2": "NO$_2$", "CO": "CO", "TVOC": "TVOC",
            "eCO2": "eCO$_2$"}


# ------------------------------------------------------------------ E1
def e1_frame():
    """E1 test-shift results, with the E11 baselines merged in (paired on the same seeds). The re-tuned PI
    and timer (wider grids, E11) replace the E1 runs; THC, tracking MPC and relax-and-round are added."""
    df = pd.read_csv(os.path.join(RES, "e1/main.csv"))
    p11 = os.path.join(RES, "e11/test.csv")
    if os.path.exists(p11):
        x = pd.read_csv(p11)
        x["policy"] = x.policy.replace({"PI*": "PI", "Timer*": "Timer+Relay-PM"})
        df = pd.concat([df[~df.policy.isin(["PI", "Timer+Relay-PM"])], x[df.columns.intersection(x.columns)]],
                       ignore_index=True)
    return df


def e1():
    df = e1_frame()
    T = json.load(open(os.path.join(RES, "tuning.json")))
    mac("rhoIso", T["rho_iso"], "{:.3f}")
    mac("nTest", df.seed.nunique(), "{:d}")
    order = ["Off", "Relay", "Relay-PM", "AO-80", "AO-100", "PI", "Timer+Relay-PM", "THC", "Costate-FB",
             "MPC-track", "MPC-RR", "MPC-FB", "MPC-FF", "MPC-oracle"]
    label = {"Off": "Fan off", "Relay": "Relay (baseline)", "Relay-PM": "Relay, PM channels only",
             "AO-80": "Always-on 80\\,\\%", "AO-100": "Always-on 100\\,\\%", "PI": "PI (tuned)",
             "Timer+Relay-PM": "Timer pre-activation + relay (tuned)", "THC": "Threshold + hysteresis (tuned)",
             "Costate-FB": "Costate feedback law",
             "MPC-track": "Zone-tracking MPC (tuned)", "MPC-RR": "Economic MPC, relax-and-round",
             "MPC-FB": "Economic MPC, feedback only", "MPC-FF": "Economic MPC, schedule forecast",
             "MPC-oracle": "Economic MPC, oracle"}
    piv = {k: df.pivot(index="seed", columns="policy", values=k) for k in
           ["captured", "Efrac", "energy_Wh", "CE", "TWA_PM25", "switches", "exceed_L3_min"]}
    rows = []
    ps, pes, keys = [], [], []
    for pol in order:
        if pol not in piv["CE"]:
            continue
        c = piv["captured"][pol]; e = piv["Efrac"][pol]
        r = [label[pol], f"{c.mean():.3f}", f"{e.mean():.3f}", f"{piv['CE'][pol].mean():.2f}",
             f"{piv['TWA_PM25'][pol].mean():.1f}", f"{piv['switches'][pol].mean():.0f}",
             f"{piv['exceed_L3_min'][pol].mean():.1f}"]
        if pol not in ("Off", "Relay", "AO-100"):
            dm, lo, hi, p, share = paired(piv["captured"][pol], piv["captured"]["Relay"])
            de, elo, ehi, pe, _ = paired(piv["Efrac"][pol], piv["Efrac"]["Relay"])
            r += [f"{dm:+.3f} [{lo:+.3f}, {hi:+.3f}]", f"{de:+.3f} [{elo:+.3f}, {ehi:+.3f}]"]
            ps += [p]; pes.append(pe); keys.append(pol)
        else:
            r += ["--", "--"]
        rows.append(r)
    tex_table(os.path.join(GEN, "tab_e1.tex"),
              ["Policy", "$\\bar\\kappa$", "$\\bar\\epsilon$", "CE", "TWA$_{2.5}$", "Sw.", "$t_{>\\bar c}$",
               "$\\Delta\\kappa$ vs relay [95\\% CI]", "$\\Delta\\epsilon$ vs relay [95\\% CI]"], rows, "lrrrrrrll")
    padj = holm(ps)
    mac("eOneMaxHolmP", prel_le(max(padj)), "{}")
    padj_e = holm(pes)
    nonsig = [k for k, q in zip(keys, padj_e) if q >= 0.05]
    mac("eOneEnNonsig", ", ".join(nonsig) if nonsig else "none", "{}")
    mac("exceedMaxAll", max(piv["exceed_L3_min"][k].mean() for k in order if k in piv["CE"] and k != "Off"), "{:.1f}")
    mac("exceedOff", piv["exceed_L3_min"]["Off"].mean(), "{:.1f}")
    for k in ["Relay", "AO-80", "AO-100", "Costate-FB", "MPC-FB", "MPC-FF", "MPC-oracle", "PI", "Timer+Relay-PM", "Off",
              "THC", "MPC-track", "MPC-RR"]:
        if k not in piv["CE"]:
            continue
        kk = k.replace("-", "").replace("+", "").replace("100", "Hundred").replace("80", "Eighty")
        mac(f"cap{kk}", piv["captured"][k].mean(), "{:.3f}")
        mac(f"ef{kk}", piv["Efrac"][k].mean(), "{:.3f}")
        mac(f"en{kk}", piv["energy_Wh"][k].mean(), "{:.1f}")
        mac(f"sw{kk}", piv["switches"][k].mean(), "{:.0f}")
        mac(f"twa{kk}", piv["TWA_PM25"][k].mean(), "{:.1f}")
        mac(f"ce{kk}", piv["CE"][k].mean(), "{:.2f}")
    # purifier leverage
    lev = (piv["CE"]["Off"] - piv["CE"]["AO-100"]) / piv["CE"]["Off"]
    mac("leverage", 100 * lev.mean(), "{:.1f}")
    mac("leverageLo", 100 * lev.quantile(0.05), "{:.1f}")
    mac("leverageHi", 100 * lev.quantile(0.95), "{:.1f}")
    twa_lev = piv["TWA_PM25"]["Off"] - piv["TWA_PM25"]["AO-100"]
    mac("twaSpan", twa_lev.mean(), "{:.1f}")
    # key paired comparisons
    pairs = [("MPC-FF", "Relay", "FFvsRelay"), ("Costate-FB", "Relay", "CFvsRelay"), ("MPC-FF", "AO-80", "FFvsAOEighty"),
             ("MPC-FF", "THC", "FFvsTHC"), ("Costate-FB", "THC", "CFvsTHC"), ("MPC-FF", "MPC-track", "FFvsTRK"),
             ("MPC-FF", "MPC-RR", "FFvsRR"), ("THC", "Relay", "THCvsRelay")]
    for a, b, nm in pairs:
        if a not in piv["CE"] or b not in piv["CE"]:
            continue
        dm, lo, hi, p, share = paired(piv["captured"][a], piv["captured"][b])
        mac(f"pCapP{nm}", prel(p), "{}")
        mac(f"dCap{nm}", dm, "{:+.3f}"); mac(f"dCap{nm}Lo", lo, "{:+.3f}"); mac(f"dCap{nm}Hi", hi, "{:+.3f}")
        mac(f"share{nm}", 100 * share, "{:.0f}")
        de, elo, ehi, pe, _ = paired(piv["energy_Wh"][a], piv["energy_Wh"][b])
        mac(f"dEn{nm}", 100 * de / piv["energy_Wh"][b].mean(), "{:+.1f}")
        dce, clo, chi, pc, _ = paired(piv["CE"][a], piv["CE"][b])
        mac(f"dCE{nm}", 100 * dce / piv["CE"][b].mean(), "{:+.2f}")
        mac(f"dCE{nm}Lo", 100 * clo / piv["CE"][b].mean(), "{:+.2f}")
        mac(f"dCE{nm}Hi", 100 * chi / piv["CE"][b].mean(), "{:+.2f}")
        dt, tlo, thi, pt, _ = paired(piv["TWA_PM25"][a], piv["TWA_PM25"][b])
        mac(f"dTWA{nm}", dt, "{:+.2f}"); mac(f"dTWA{nm}Lo", tlo, "{:+.2f}"); mac(f"dTWA{nm}Hi", thi, "{:+.2f}")
    # same objective J = CE + rho E: forecast value
    rho = T["rho_iso"]
    J = piv["CE"] + rho * piv["energy_Wh"]
    for a, b, nm in [("MPC-FF", "MPC-FB", "FFvsFB"), ("MPC-oracle", "MPC-FF", "ORvsFF"), ("MPC-FF", "Costate-FB", "FFvsCF"),
                     ("Costate-FB", "MPC-FF", "CFvsFF"), ("MPC-FF", "THC", "JFFvsTHC"), ("MPC-FF", "MPC-track", "JFFvsTRK"),
                     ("MPC-FF", "MPC-RR", "JFFvsRR"), ("MPC-RR", "MPC-FF", "JRRvsFF")]:
        if a not in J or b not in J:
            continue
        dm, lo, hi, p, share = paired(J[a], J[b])
        mac(f"dJ{nm}", 100 * dm / J[b].mean(), "{:+.3f}")
        mac(f"dJ{nm}Lo", 100 * lo / J[b].mean(), "{:+.3f}")
        mac(f"dJ{nm}Hi", 100 * hi / J[b].mean(), "{:+.3f}")
        mac(f"pJ{nm}", prel(p), "{}")
        dc, clo, chi, pc, _ = paired(piv["captured"][a], piv["captured"][b])
        mac(f"dCap{nm}", dc, "{:+.4f}"); mac(f"dCap{nm}Lo", clo, "{:+.4f}"); mac(f"dCap{nm}Hi", chi, "{:+.4f}")
        mac(f"pCap{nm}", prel(pc), "{}")
    # sample size for a physical paired crossover (MPC-FF vs relay), from per-shift effect sizes
    for k, nm in [("CE", "CE"), ("energy_Wh", "En"), ("captured", "Cap")]:
        d = piv[k]["MPC-FF"] - piv[k]["Relay"]
        dz = abs(d.mean()) / d.std(ddof=1)
        n_t = ((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) / dz) ** 2 + 0.5 * stats.norm.ppf(0.975) ** 2
        mac(f"dz{nm}", dz, "{:.2f}")
        mac(f"nPairs{nm}", int(np.ceil(n_t / 0.864)), "{:d}")


# ------------------------------------------------------------------ E11 stronger baselines
def e11():
    T2 = json.load(open(os.path.join(RES, "e11/tuning2.json")))
    th = T2["thc"]
    mac("thcThOne", th["th1"], "{:g}"); mac("thcThTwo", th["th1"] + th["dth"], "{:g}")
    mac("thcThThree", th["th1"] + 3 * th["dth"], "{:g}"); mac("thcHyst", th["hyst"], "{:g}")
    mac("thcGate", "with" if th["gate"] else "without", "{}")
    mac("piRset", T2["pi"]["r25"], "{:g}"); mac("piKp", T2["pi"]["kp"], "{:g}"); mac("piKi", T2["pi"]["ki"], "{:g}")
    mac("tmLead", 60 * T2["timer"]["lead"], "{:.0f}"); mac("tmTail", 60 * T2["timer"]["tail"], "{:.0f}")
    mac("tmDuty", 100 * T2["timer"]["u_ff"], "{:.0f}")
    mac("trkR", T2["track"]["r25"], "{:g}"); mac("trkQ", T2["track"]["q"], "{:g}")
    edges = [k for k in ["thc_edges", "pi_edges", "timer_edges", "track_edges"] for v in T2.get(k, {}).values() if v == "edge"]
    mac("nTuneEdges", len(edges), "{:d}")
    # fronts of the new baselines against the economic MPC front (same 60 seeds, as-built purifier)
    p = os.path.join(RES, "e11/fronts.csv")
    pf = os.path.join(RES, "e2/fronts_scale1.0.csv")
    if os.path.exists(p) and os.path.exists(pf):
        nf = pd.read_csv(p)
        nf["fam"] = nf.policy.str.split("|").str[0]
        mf = pd.read_csv(pf)
        mf["fam"] = mf.policy.str.split("|").str[0]
        t11 = pd.read_csv(os.path.join(RES, "e11/test.csv"))
        thc = t11[t11.policy == "THC"].set_index("seed")
        d_iso = []
        for sd, g in mf[mf.fam == "MPC-FF"].groupby("seed"):
            if sd in thc.index:
                d_iso.append(interp_front(g, thc.Efrac[sd]) - thc.captured[sd])
        mac("isoFFvsTHC", np.mean(d_iso), "{:+.3f}")
        lo, hi = boot_ci(d_iso); mac("isoFFvsTHCLo", lo, "{:+.3f}"); mac("isoFFvsTHCHi", hi, "{:+.3f}")
        # largest captured share the THC and tracking fronts reach at or below the relay's energy
        rel_e = mf[mf.policy == "Relay"].Efrac.mean()
        for fam, nm in [("THC", "THC"), ("TRK", "TRK")]:
            m = nf[nf.fam == fam].groupby("policy")[["captured", "Efrac"]].mean().sort_values("Efrac")
            mac(f"frontMax{nm}", np.interp(rel_e, m.Efrac.values, m.captured.values), "{:.3f}")
        m = mf[mf.fam == "MPC-FF"].groupby("policy")[["captured", "Efrac"]].mean().sort_values("Efrac")
        mac("frontMaxFF", np.interp(rel_e, m.Efrac.values, m.captured.values), "{:.3f}")


def e6m():
    p = os.path.join(RES, "e6/methods_summary.json")
    if os.path.exists(p):
        s = json.load(open(p))
        mac("rrSubopt", 100 * s["rr"]["share_suboptimal"], "{:.0f}")
        mac("rrGtOne", 100 * s["rr"]["share_gap_gt_1pct"], "{:.0f}")
        mac("rrGapPct", 100 * s["rr"]["p95_gap"], "{:.1f}")
        mac("rrGapMax", 100 * s["rr"]["max_gap"], "{:.1f}")
        mac("rrFirst", 100 * s["rr"]["first_move_agreement"], "{:.0f}")
        mac("rrInfeas", 100 * s["rr"]["round_infeasible_share_binding"], "{:.0f}")
        mac("constSubopt", 100 * s["const"]["share_suboptimal"], "{:.0f}")
        mac("constGtOne", 100 * s["const"]["share_gap_gt_1pct"], "{:.0f}")
        mac("constGapMax", 100 * s["const"]["max_gap"], "{:.0f}")
        mac("mN", s["n"], "{:d}"); mac("mNbind", s["n_binding"], "{:d}")
    p = os.path.join(RES, "e6/esp32_footprint.json")
    if os.path.exists(p):
        f = json.load(open(p))
        mac("espSolverKB", f["solver_text_bytes_Os"] / 1024, "{:.1f}")
        mac("espWsKB", f["workspace_bytes_xtensa"] / 1024, "{:.1f}")
        mac("espCostateB", f["costate_text_bytes_Os"], "{:d}")
        if "rollout_text_bytes_Os" in f:
            mac("espRolloutB", f"{f['rollout_text_bytes_Os']:,d}".replace(",", "{,}"))
            mac("espRolloutParB", f"{f['rollout_param_bytes']:,d}".replace(",", "{,}"))
        mac("espSketchKB", f["bench_sketch_flash_bytes"] / 1024, "{:.0f}")
        mac("espSketchPct", 100 * f["bench_sketch_flash_bytes"] / 1310720, "{:.0f}")
        mac("espRamKB", f["bench_sketch_static_ram_bytes"] / 1024, "{:.0f}")
        mac("espRamPct", 100 * f["bench_sketch_static_ram_bytes"] / 327680, "{:.0f}")
    p = os.path.join(RES, "e6/rollout_c_check.json")
    if os.path.exists(p):
        f = json.load(open(p))
        mac("rlCcases", f"{f['cases']:,d}".replace(",", "{,}")); mac("rlCmism", f["mismatches"], "{:d}")
        mac("rlCbudget", f"{f['budget_limited_cases']:,d}".replace(",", "{,}"))
        mac("rlMacsBack", f"{f['macs_backward_max']:,d}".replace(",", "{,}")); mac("rlMacsScore", f["macs_scoring"], "{:d}")


# ------------------------------------------------------------------ E2 iso-energy comparison
def interp_front(sub, x):
    """Front value at energy x: monotone cubic (PCHIP) interpolation of the swept points, which follows the
    concave front more closely than linear interpolation; linear if the points are not strictly increasing."""
    g = sub.groupby("Efrac", as_index=False).captured.mean().sort_values("Efrac")
    e, k = g.Efrac.values, g.captured.values
    x = np.clip(x, e[0], e[-1])
    if len(e) >= 3 and np.all(np.diff(e) > 0):
        from scipy.interpolate import PchipInterpolator
        return PchipInterpolator(e, k)(x)
    return np.interp(x, e, k)


def e2():
    for sc in ["1.0", "4.0"]:
        p = os.path.join(RES, f"e2/fronts_scale{sc}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df["fam"] = df.policy.str.split("|").str[0]
        tag = "A" if sc == "1.0" else "B"
        rel = df[df.policy == "Relay"].set_index("seed")
        for fam in ["MPC-FF", "MPC-FB", "MPC-oracle", "Costate-FB", "PI"]:
            d_iso, e_save = [], []
            for sd, g in df[df.fam == fam].groupby("seed"):
                d_iso.append(interp_front(g, rel.Efrac[sd]) - rel.captured[sd])
                gg = g.sort_values("captured")
                e_eq = np.interp(rel.captured[sd], gg.captured.values, gg.Efrac.values)
                e_save.append(1 - e_eq / rel.Efrac[sd])
            k = fam.replace("-", "")
            mac(f"iso{k}{tag}", np.mean(d_iso), "{:+.3f}")
            lo, hi = boot_ci(d_iso); mac(f"iso{k}{tag}Lo", lo, "{:+.3f}"); mac(f"iso{k}{tag}Hi", hi, "{:+.3f}")
            mac(f"esave{k}{tag}", 100 * np.mean(e_save), "{:.0f}")
            lo, hi = boot_ci(e_save); mac(f"esave{k}{tag}Lo", 100 * lo, "{:.0f}"); mac(f"esave{k}{tag}Hi", 100 * hi, "{:.0f}")
        for pol in ["Relay", "AO-80", "Timer+Relay-PM"]:
            k = pol.replace("-", "").replace("+", "").replace("80", "Eighty")
            mac(f"capF{k}{tag}", df[df.policy == pol].captured.mean(), "{:.3f}")
            mac(f"efF{k}{tag}", df[df.policy == pol].Efrac.mean(), "{:.3f}")
        lev = (df[df.policy == "Off"].set_index("seed").CE - df[df.policy == "AO-100"].set_index("seed").CE) / df[df.policy == "Off"].set_index("seed").CE
        mac(f"leverage{tag}", 100 * lev.mean(), "{:.1f}")


# ------------------------------------------------------------------ E3 budget
def e3():
    df = pd.read_csv(os.path.join(RES, "e3/budget_main.csv"))
    piv = df.pivot(index="seed", columns="policy", values="captured")
    pe = df.pivot(index="seed", columns="policy", values="Efrac")
    order = ["Relay", "Relay+cutoff", "Costate-FB/fixed", "Costate-FB/pacing", "MPC-FB/fixed", "MPC-FB/pacing",
             "MPC-FF/fixed", "MPC-FF/pacing", "MPC-FB/rollout", "Costate-FB/rollout", "MPC-FF/rollout", "MPC-oracle/rollout"]
    order = [o for o in order if o in piv]
    fb_only = [o for o in order if o.startswith(("Costate-FB/fixed", "Costate-FB/pacing", "MPC-FB/"))]
    best_fb = max(fb_only, key=lambda o: piv[o].mean())
    mac("bestFB", best_fb.replace("/", " / "), "{}")
    rows = []
    for pol in order:
        r = [pol.replace("/", " / "), f"{piv[pol].mean():.3f}", f"{pe[pol].mean():.3f}", f"{pe[pol].max():.3f}"]
        if pol != best_fb and pol != "Relay":
            dm, lo, hi, p, share = paired(piv[pol], piv[best_fb])
            r += [f"{dm:+.3f} [{lo:+.3f}, {hi:+.3f}]", pfmt(p)]
        else:
            r += ["--", "--"]
        rows.append(r)
    tex_table(os.path.join(GEN, "tab_e3.tex"), ["Policy", "$\\bar\\kappa$", "$\\bar\\epsilon$", "max $\\epsilon$",
              f"$\\Delta\\kappa$ vs {best_fb.replace('/', ' / ')} [95\\% CI]", "$p$"], rows, "lrrrlr")
    for a, b, nm in [("MPC-FF/rollout", best_fb, "FFrollVsBestFB"), ("MPC-FF/rollout", "MPC-FB/rollout", "FFrollVsFBroll"),
                     ("MPC-oracle/rollout", "MPC-FF/rollout", "ORrollVsFFroll"), ("MPC-FF/rollout", "Relay+cutoff", "FFrollVsRelayCut"),
                     ("Costate-FB/rollout", best_fb, "CFrollVsBestFB")]:
        if a not in piv or b not in piv:
            continue
        dm, lo, hi, p, share = paired(piv[a], piv[b])
        mac(f"b{nm}", dm, "{:+.3f}"); mac(f"b{nm}Lo", lo, "{:+.3f}"); mac(f"b{nm}Hi", hi, "{:+.3f}")
        mac(f"b{nm}P", prel(p), "{}"); mac(f"b{nm}Share", 100 * share, "{:.0f}")
        mac(f"b{nm}Rel", 100 * dm / piv[b].mean(), "{:+.0f}")
    for pol, k in [("MPC-FF/rollout", "FFroll"), ("MPC-oracle/rollout", "ORroll"), (best_fb, "BestFB"),
                   ("Relay+cutoff", "RelayCut"), ("Costate-FB/rollout", "CFroll"), ("MPC-FB/rollout", "FBroll")]:
        if pol in piv:
            mac(f"bcap{k}", piv[pol].mean(), "{:.3f}")
            mac(f"bef{k}", pe[pol].mean(), "{:.3f}")
            mac(f"befmax{k}", pe[pol].max(), "{:.3f}")
    # heterogeneity of the per-shift gain and what it correlates with
    sys.path.insert(0, os.path.join(ROOT, "code"))
    from aqmpc.scenarios import make_shift, ScenarioConfig, _rate
    gain = piv["MPC-FF/rollout"] - piv[best_fb]
    late, cout = [], []
    for s in gain.index:
        sh = make_shift(np.random.default_rng(int(s)), ScenarioConfig())
        m = np.array([(e.t1 - e.t0) * _rate[e.kind][0].sum() for e in sh.plan])
        mid = np.array([(e.t0 + e.t1) / 2 for e in sh.plan])
        late.append((m * (mid > 5.0)).sum() / max(m.sum(), 1e-9)); cout.append(sh.c_out[0, 0])
    q = np.percentile(gain, [10, 50, 90])
    mac("bGainQlo", q[0], "{:+.3f}"); mac("bGainMed", q[1], "{:+.3f}"); mac("bGainQhi", q[2], "{:+.3f}")
    r1 = stats.spearmanr(cout, gain); r2 = stats.spearmanr(late, gain)
    mac("bRhoOut", r1.statistic, "{:+.2f}"); mac("bRhoOutP", prel(r1.pvalue), "{}")
    mac("bRhoLate", r2.statistic, "{:+.2f}"); mac("bRhoLateP", prel(r2.pvalue), "{}")
    p2 = os.path.join(RES, "e3/budget_sweep.csv")
    if os.path.exists(p2):
        s = pd.read_csv(p2)
        for frac in sorted(s.budget_frac.unique()):
            ps_ = s[s.budget_frac == frac].pivot(index="seed", columns="policy", values="captured")
            fbs = [o for o in ps_ if o.startswith(("Costate-FB/fixed", "Costate-FB/pacing", "MPC-FB/"))]
            bf = max(fbs, key=lambda o: ps_[o].mean())
            d = ps_["MPC-FF/rollout"] - ps_[bf]
            words = {15: "Fifteen", 25: "TwentyFive", 35: "ThirtyFive", 45: "FortyFive"}
            mac(f"sweepGain{words[int(round(frac * 100))]}", d.mean(), "{:+.3f}")


# ------------------------------------------------------------------ E4 robustness
def e4():
    df = pd.read_csv(os.path.join(RES, "e4/robustness.csv"))
    rows = []
    gains_u, gains_b = [], []
    for cfg, d in df.groupby("config", sort=False):
        pc = d.pivot(index="seed", columns="policy", values="captured")
        pe = d.pivot(index="seed", columns="policy", values="Efrac")
        du = paired(pc["MPC-FF"], pc["Relay"])
        de = paired(pe["MPC-FF"], pe["Relay"])
        dff = paired(pc["MPC-FF"], pc["MPC-FB"])
        db = paired(pc["MPC-FF/rollout"], pc["MPC-FB/rollout"])
        dbf = paired(pc["MPC-FF/rollout"], pc["Costate-FB/fixed"])
        gains_u.append(du[0]); gains_b.append(dbf[0])
        rows.append([cfg, f"{pc['Relay'].mean():.2f}/{pe['Relay'].mean():.2f}",
                     f"{pc['MPC-FF'].mean():.2f}/{pe['MPC-FF'].mean():.2f}",
                     f"{dff[0]:+.4f}", f"{pc['MPC-FF/rollout'].mean():.2f}", f"{dbf[0]:+.3f} [{dbf[1]:+.2f}, {dbf[2]:+.2f}]",
                     f"{db[0]:+.3f}"])
    tex_table(os.path.join(GEN, "tab_e4.tex"), ["Truth / scenario", "Relay $\\kappa/\\epsilon$", "MPC-FF $\\kappa/\\epsilon$",
              "FF$-$FB", "FF-roll. $\\kappa$", "FF-roll.$-$CF-fixed [95\\% CI]", "FF-roll.$-$FB-roll."], rows, "lllrrlr",
              label=scenario_label)
    mac("robMinBudgetGain", min(gains_b), "{:+.3f}"); mac("robMaxBudgetGain", max(gains_b), "{:+.3f}")
    mac("robMinRelayGain", min(gains_u), "{:+.3f}"); mac("robMaxRelayGain", max(gains_u), "{:+.3f}")
    # forecast value at free energy, and what drives the budget gain across configurations
    ffb, unspent, rolluse, over_exp = [], [], [], None
    for cfg, d in df.groupby("config", sort=False):
        pc = d.pivot(index="seed", columns="policy", values="captured")
        pe = d.pivot(index="seed", columns="policy", values="Efrac")
        ffb.append(abs((pc["MPC-FF"] - pc["MPC-FB"]).mean()))
        unspent.append(1 - pe["Costate-FB/fixed"].mean() / 0.25)
        rolluse.append(pe["MPC-FF/rollout"].mean() / 0.25)
        if cfg == "fan exponent 2.5":
            over_exp = pe["MPC-FF/rollout"].mean() / 0.25 - 1
    mac("robFFFBmax", max(ffb), "{:.4f}")
    r = stats.spearmanr(unspent, gains_b)
    mac("robRhoUnspent", r.statistic, "{:.2f}"); mac("robRhoUnspentP", prel(r.pvalue), "{}")
    nominal_like = [u for c, u in zip(df.config.unique(), rolluse) if c != "fan exponent 2.5"]
    mac("robRollUseMin", 100 * min(nominal_like), "{:.0f}"); mac("robRollUseMax", 100 * max(nominal_like), "{:.0f}")
    mac("robFixedUseMin", 100 * (1 - max(unspent)), "{:.0f}"); mac("robFixedUseMax", 100 * (1 - min(unspent)), "{:.0f}")
    if over_exp is not None:
        mac("robExpOver", 100 * over_exp, "{:.0f}")
    mac("robNcfg", len(rows), "{:d}")
    mac("robNposBudget", int(np.sum(np.array(gains_b) > 0)), "{:d}")


# ------------------------------------------------------------------ E5 weights
def e5():
    df = pd.read_csv(os.path.join(RES, "e5/weights.csv"))
    nom = df[df.design == "nominal"]
    piv = nom.pivot_table(index=["weights", "seed"], columns="policy", values="CE")
    cap = lambda pol: ((piv["Off"] - piv[pol]) / (piv["Off"] - piv["AO-100"])).groupby("weights").mean()
    d = cap("MPC-FF") - cap("Relay")
    mac("wN", len(d), "{:d}")
    mac("wGainMin", d.min(), "{:+.3f}"); mac("wGainMed", d.median(), "{:+.3f}"); mac("wGainMax", d.max(), "{:+.3f}")
    mac("wNpos", int((d > 0).sum()), "{:d}")
    dcf = cap("Costate-FB") - cap("Relay")
    mac("wCFGainMin", dcf.min(), "{:+.3f}")
    # redesign: matched-weight MPC vs nominal-design MPC under the same evaluation weights
    mt = df[df.design == "matched"].set_index(["weights", "seed"]).CE
    gain = []
    for w in df[df.design == "matched"].weights.unique():
        a = mt.loc[w]; b = piv.loc[w]["MPC-FF"]; off = piv.loc[w]["Off"]; ao = piv.loc[w]["AO-100"]
        gain.append((((off - a) / (off - ao)) - ((off - b) / (off - ao))).mean())
    mac("wRedesignMax", max(gain), "{:+.3f}"); mac("wRedesignMin", min(gain), "{:+.3f}")
    em = df[df.design == "matched"].groupby("weights").energy_Wh.mean()
    en = df[(df.design == "nominal") & (df.policy == "MPC-FF")].energy_Wh.mean()
    mac("wRedesignEmin", em.min(), "{:.1f}"); mac("wRedesignEmax", em.max(), "{:.1f}"); mac("wNominalE", en, "{:.1f}")


# ------------------------------------------------------------------ E6
def e6():
    p = os.path.join(RES, "e6/exactness_summary.json")
    if os.path.exists(p):
        s = json.load(open(p))
        mac("exN", s["instances"], "{:d}"); mac("exGap", s["max_rel_gap"], "{:.1e}")
        mac("exLB", s["lb_checks"], "{:d}"); mac("exLBviol", s["lb_violations"], "{:d}")
        mac("exNodesMed", s["median_nodes_bb"], "{:.0f}"); mac("exNodesPctl", s["p95_nodes_bb"], "{:.0f}")
        mac("exFull", s["full_tree_nodes"], "{:d}")
    p = os.path.join(RES, "e6/timing_summary.json")
    if os.path.exists(p):
        s = json.load(open(p))
        t = s["deployed_median_time_us"]; t95 = s["deployed_p95_time_us"]; nd = s["deployed_median_nodes"]
        for m, k in [("0", "Ex"), ("1", "BF"), ("2", "BA")]:
            mac(f"tMed{k}", t[m] / 1000, "{:.2f}"); mac(f"tPctl{k}", t95[m] / 1000, "{:.2f}")
            mac(f"nodesMed{k}", nd[m], "{:.0f}")
        mac("cRelErr", s["deployed_max_rel_err_vs_python"], "{:.1e}")
        mac("cAgree", 100 * s["deployed_first_move_agreement"], "{:.0f}")
        mac("wsBytes", s["workspace_bytes_deployed"] / 1024, "{:.1f}")
        mac("modelBytes", s["model_bytes_deployed"] / 1024, "{:.1f}")
    p = os.path.join(RES, "e6/timing_scaling.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        g = d[(d.factor == "Nu") & (d["mode"] == 0)].groupby("value").time_us.median()
        g2 = d[(d.factor == "Nu") & (d["mode"] == 2)].groupby("value").time_us.median()
        mac("tNuEightEx", g.get(8, np.nan) / 1000, "{:.1f}"); mac("tNuEightBA", g2.get(8, np.nan) / 1000, "{:.1f}")
        g = d[(d.factor == "nz") & (d["mode"] == 2)].groupby("value").time_us.median()
        mac("tNzEightBA", g.get(8, np.nan) / 1000, "{:.1f}")


# ------------------------------------------------------------------ E7, E8
def e7():
    df = pd.read_csv(os.path.join(RES, "e7/budget_formulation.csv"))
    df["form"] = df.policy.str.split("|").str[0]
    df["over"] = df.energy_Wh / df.budget_Wh - 1
    for f, k in [("budget-state", "BS"), ("per-horizon cap", "PH")]:
        d = df[df.form == f]
        mac(f"over{k}Max", 100 * d.over.max(), "{:+.1f}")
        mac(f"over{k}Share", 100 * (d.over > 1e-9).mean(), "{:.0f}")
    rows = []
    for (f, b), d in df.groupby(["form", "budget_Wh"]):
        rows.append([f.replace("budget-state", "budget state"), f"{b:.1f}", f"{d.energy_Wh.mean():.1f}", f"{d.energy_Wh.max():.1f}",
                     f"{100 * (d.over > 1e-9).mean():.0f}", f"{d.CE.mean():.2f}"])
    tex_table(os.path.join(GEN, "tab_e7.tex"), ["Formulation", "$E_B$", "Mean $E$", "Max $E$",
              "Over (\\%)", "CE"], rows, "lrrrrr")


def e8():
    df = pd.read_csv(os.path.join(RES, "e8/horizon.csv"))
    T = json.load(open(os.path.join(RES, "tuning.json")))
    df = df[~df.policy.isin(["Off", "AO-100"])]
    rows = []
    for pol, d in df.groupby("policy"):
        N, Nu, term = [x.split("=")[1] for x in pol.split("|")]
        rows.append((int(N), int(Nu), int(term), d.captured.mean(), d.Efrac.mean(), (d.CE + T["rho_iso"] * d.energy_Wh).mean()))
    rows.sort()
    tex_table(os.path.join(GEN, "tab_e8.tex"), ["$N$", "$N_u$", "Terminal cost", "$\\bar\\kappa$", "$\\bar\\varphi$", "mean $J$"],
              [[str(a), str(b), "yes" if c else "no", f"{d:.3f}", f"{e:.3f}", f"{f:.3f}"] for a, b, c, d, e, f in rows], "rrlrrr")
    js = {(a, b, c): f for a, b, c, d, e, f in rows}
    if (30, 6, 1) in js and (10, 6, 1) in js:
        jt = [f for (a, b, c), f in js.items() if c == 1]
        mac("hJrange", 100 * (max(jt) - min(jt)) / js[(30, 6, 1)], "{:.2f}")
    noterm = [f for (a, b, c), f in js.items() if c == 0]
    term = [f for (a, b, c), f in js.items() if c == 1 and b == 6]
    if noterm and term:
        mac("hNoTermPenalty", 100 * (js.get((10, 6, 0), np.nan) - js.get((10, 6, 1), np.nan)) / js.get((10, 6, 1), np.nan), "{:.1f}")


# ------------------------------------------------------------------ E9
def e9():
    s = pd.read_csv(os.path.join(RES, "e9/forecast_summary.csv"))
    sel = s[(s.channel == "PM2.5 zone A") & (s.lead_min == 15)]
    for (cfg, fc), r in sel.set_index(["config", "forecaster"]).iterrows():
        k = ("Nom" if cfg.startswith("nominal") else "Mis") + ("Inv" if "inverted" in fc else "True")
        mac(f"fRmseT{k}", r.rmse_vs_true / r.true_sd, "{:.2f}")
        mac(f"fRmseI{k}", r.rmse_vs_inv / r.true_sd, "{:.2f}")
        mac(f"fInvBias{k}", 100 * r.inv_bias / r.true_mean, "{:+.0f}")
        mac(f"fBias{k}", 100 * r.bias_vs_true / r.true_mean, "{:+.0f}")
    cl = pd.read_csv(os.path.join(RES, "e9/closed_loop.csv"))
    cl = cl[cl.policy == "MPC-FF/rollout"]
    for cfg in cl.config.unique():
        k = ("Nom" if cfg.startswith("nominal") else "Mis") + ("Inv" if "inverted" in cfg else "True")
        mac(f"fCap{k}", cl[cl.config == cfg].captured.mean(), "{:.3f}")


def e1b():
    p = os.path.join(RES, "e1/relay_occ.csv")
    if os.path.exists(p):
        r = pd.read_csv(p)
        main = pd.read_csv(os.path.join(RES, "e1/main.csv"))
        rel = main[main.policy == "Relay"].set_index("seed")
        ro = r.set_index("seed")
        mac("capRelayOcc", ro.captured.mean(), "{:.3f}"); mac("efRelayOcc", ro.Efrac.mean(), "{:.3f}")
        mac("enSaveRelayOcc", 100 * (1 - ro.energy_Wh.mean() / rel.energy_Wh.mean()), "{:.1f}")
        mac("dCapRelayOcc", (ro.captured - rel.captured.loc[ro.index]).mean(), "{:+.3f}")
    p = os.path.join(RES, "e1/mechanism.csv")
    if os.path.exists(p):
        g = pd.read_csv(p).groupby("policy").mean(numeric_only=True)
        mac("mechRelayUnocc", 100 * g.E_unocc_share["Relay"], "{:.1f}")
        mac("mechMPCUnocc", 100 * g.E_unocc_share["MPC-FF"], "{:.1f}")
        mac("mechRelayEighty", 100 * g.frac80["Relay"], "{:.0f}")
        mac("mechRelayNinety", 100 * g.frac90["Relay"], "{:.0f}")
        mac("mechMPCEighty", 100 * g.frac80["MPC-FF"], "{:.0f}")
        mac("mechMPCHundred", 100 * g.frac100["MPC-FF"], "{:.0f}")
        mac("mechRelayRun", g.runtime_h["Relay"], "{:.1f}")
        mac("mechMPCRun", g.runtime_h["MPC-FF"], "{:.1f}")


def example():
    p = os.path.join(RES, "e1/response_example.json")
    if os.path.exists(p):
        d = json.load(open(p))
        for k, nm in [("Relay", "Relay"), ("Costate-FB", "CF"), ("Rollout-CL", "RCL"), ("MPC-FF", "FF")]:
            if k not in d:
                continue
            mac(f"ex{nm}CE", d[k]["CE"], "{:.2f}"); mac(f"ex{nm}E", d[k]["energy_Wh"], "{:.1f}")
            mac(f"ex{nm}Sw", d[k]["switches"], "{:d}")


def e10():
    df = pd.read_csv(os.path.join(RES, "e10/identification.csv"))
    g = df.groupby(["case", "pairs"]).rel_err
    mae = g.apply(lambda e: np.mean(np.abs(e)))
    bias = g.mean()
    mac("idMaeNomOne", 100 * mae[("nominal", 1)], "{:.0f}")
    mac("idMaeNomThree", 100 * mae[("nominal", 3)], "{:.0f}")
    mac("idMaeQlowThree", 100 * mae[("Q x0.6", 3)], "{:.0f}")
    mac("idMaeQhighThree", 100 * mae[("Q x1.25", 3)], "{:.0f}")
    mac("idBiasLamThree", 100 * bias[("lam0 x2", 3)], "{:+.0f}")
    mac("idBiasMixThree", 100 * bias[("interzonal x0.25", 3)], "{:+.0f}")


def platform():
    p = os.path.join(ROOT, "results", "platform/calibration_check.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d = d[d.subset == "all 11 h"]
    rows = [[r.pollutant, f"{r.MAE:.2f}", f"{r.bias:+.2f}", f"{r['MAPE_%']:.1f}", f"{r['max_rel_err_%']:.0f}",
             f"{r.F10:.2f}", f"{r.r:.2f}"] for _, r in d.iterrows()]
    tex_table(os.path.join(GEN, "tab_cal.tex"), ["Pollutant", "MAE", "Bias", "MAPE (\\%)", "Max. rel. err. (\\%)",
              "$F_{\\pm10}$", "$r$"], rows, "lrrrrrr", label=lambda p: POLL_TEX.get(p, p))
    for _, r in d.iterrows():
        mac(f"Fcal{r.pollutant.replace('.', '').replace('2', 'Two').replace('5','Five').replace('10','Ten')}", r.F10, "{:.2f}")


if __name__ == "__main__":
    for f in [e1, e1b, example, e2, e3, e4, e5, e6, e7, e8, e9, e10, platform, e11, e6m]:
        try:
            f()
            print(f.__name__, "ok")
        except (FileNotFoundError, KeyError) as ex:
            print(f.__name__, "skipped:", ex)
    with open(os.path.join(GEN, "macros.tex"), "w", encoding="utf-8") as fh:
        fh.write("% generated by code/analysis/summarize.py -- do not edit\n")
        for k, v in sorted(MAC.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{texnum(v)}}}\n")
    print(len(MAC), "macros written")
