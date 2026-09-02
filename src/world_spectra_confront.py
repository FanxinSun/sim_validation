#!/usr/bin/env python3
"""world_spectra_confront.py — generator-level confrontation of Pythia tunes with
the world pp-200 identified spectra (reshape campaign, 2026-09-02).

Inputs : P5/angantyr/<tune>_world.txt  (gen_world.cc output)
         external_data/world_pp200_spectra/ins886590/Table{1,2,3}.yaml  (PHENIX, |eta|<0.35,
             E d3sig/dp3 [mb/GeV^2] -> per inelastic event via sigma_inel = 42 mb)
         external_data/world_pp200_spectra/ins628232/fig2{a,b,c}.yaml    (STAR, |y|<0.5, per NSD event)
Outputs: <checkout>/plots/world_spectra_confront_<tag>.png, <checkout>/ledgers/world_spectra_confront_<tag>.txt
Usage  : python3 world_spectra_confront.py <tag> <tune1> <tune2> ...   (run from anywhere)
"""
import sys, os, math, yaml
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); VDIR = os.path.realpath(os.path.join(HERE, ".."))
REPO = "/home/rog/sPHENIX/3D_ClusterFindingML"; GEN = os.path.join(REPO, "P5/angantyr")  # fresh gen_world outputs; archived runs live in <checkout>/ledgers/world_scan
DATA = os.path.join(REPO, "external_data/world_pp200_spectra"); SIG_INEL = 42.0
tag = sys.argv[1] if len(sys.argv) > 1 else "v7pre"; tunes = sys.argv[2:] or ["ours", "monash", "mdc2"]
DPT = 0.05; NB = 100; PTC = (np.arange(NB) + 0.5) * DPT

def load_gen(name):
    g = {"spec": {}}
    fp = os.path.join(GEN, f"{name}_world.txt")
    if not os.path.exists(fp): fp = os.path.join(VDIR, "ledgers", "world_scan", f"{name}_world.txt")
    for line in open(fp):
        p = line.split()
        if line.startswith("SUMMARY"):
            g["nev"] = int(p[2]); g["nnsd"] = int(p[4]); g["dnde"] = float(p[6]); g["eps"] = float(p[8])
            g["classN"] = [float(x) for x in p[10:14]]; g["meanpt_ch"] = float(p[15])
        elif line.startswith("SPEC"):
            g["spec"][(p[1], p[2])] = {"meanpt": float(p[4]), "yield": float(p[6]), "y": np.array([float(x) for x in p[7:7 + NB]])}
    return g

def rebin(yfine, lo, hi):
    """bin-averaged invariant yield over [lo,hi] from fine invariant yields: counts-weighted."""
    m = (PTC >= lo) & (PTC < hi)
    if not m.any(): return np.nan
    return float((yfine[m] * PTC[m]).sum() * DPT / (0.5 * (lo + hi) * (hi - lo)))

def phenix(table, idx):
    d = yaml.safe_load(open(os.path.join(DATA, "ins886590", f"Table{table}.yaml")))
    xs = d["independent_variables"][0]["values"]; dv = d["dependent_variables"][idx]
    lo = np.array([x["low"] for x in xs]); hi = np.array([x["high"] for x in xs])
    y = np.array([v["value"] for v in dv["values"]]) / SIG_INEL
    err = np.array([math.sqrt(sum(float(e.get("symerror", 0)) ** 2 for e in v.get("errors", []))) for v in dv["values"]]) / SIG_INEL
    return lo, hi, y, err, dv["header"]["name"]

def star(fig, tagname):
    d = yaml.safe_load(open(os.path.join(DATA, "ins628232", f"{fig}.yaml")))
    xc = np.array([x["value"] for x in d["independent_variables"][0]["values"]])
    edges = np.concatenate([[xc[0] - (xc[1] - xc[0]) / 2], (xc[1:] + xc[:-1]) / 2, [xc[-1] + (xc[-1] - xc[-2]) / 2]])
    dv = [v for v in d["dependent_variables"] if tagname in v["header"]["name"] and "p+p" in v["header"]["name"]][0]
    y = np.array([float(v["value"]) if v["value"] not in ("-", "") else np.nan for v in dv["values"]])
    err = np.array([math.sqrt(sum(float(e.get("symerror", 0)) ** 2 for e in v.get("errors", []))) for v in dv["values"]])
    return edges[:-1], edges[1:], y, err, dv["header"]["name"]

def trunc_meanpt(lo, hi, y):
    c = 0.5 * (lo + hi); w = y * c * (hi - lo); m = np.isfinite(y) & (y > 0)
    return float((w[m] * c[m]).sum() / w[m].sum())

def integral(lo, hi, y, a, b):
    c = 0.5 * (lo + hi); m = (c >= a) & (c < b) & np.isfinite(y)
    return float((2 * math.pi * y[m] * c[m] * (hi[m] - lo[m])).sum())

datasets = {  # (label, loader, [(species-key-plus, species-key-minus, args)])
    "PHENIX |y|<0.35 inel": ("sel0", [("pi", phenix, (1, 0), (1, 1)), ("K", phenix, (2, 0), (2, 1)), ("p", phenix, (4, 0), (4, 1))]),
    "STAR |y|<0.5 NSD": ("sel1", [("pi", star, ("fig2a", "\\pi^{+}"), ("fig2a", "\\pi^{-}")), ("K", star, ("fig2b", "K^{+}"), ("fig2b", "K^{-}")), ("p", star, ("fig2c", "p,"), ("fig2c", "\\bar{p}"))]),
}
genkey = {"pi": ("pip", "pim"), "K": ("kp", "km"), "p": ("p", "pbar")}
G = {t: load_gen(t) for t in tunes}
out = [f"# world_spectra_confront {tag}: tunes {tunes} | PHENIX per inelastic (sigma_inel {SIG_INEL} mb, +9.7% norm. unc. not shown; p/pbar = Table 4 feed-down-corrected), STAR per NSD | gen p/pbar hyperon-feed-down excluded"]
for t in tunes:
    g = G[t]; out.append(f"TUNE {t}: dNch/deta(|eta|<0.5) {g['dnde']:.3f} | MBD eps {g['eps']:.3f} | <pT>ch {g['meanpt_ch']:.3f} | classN {' '.join(f'{x:.2f}' for x in g['classN'])} | NSD frac {g['nnsd']/g['nev']:.3f}")
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex="col"); colors = {"ours": "#A8218E", "monash": "#1F4E9C", "mdc2": "#2E7D4F", "w1": "#B8471B"}
for r, (dname, (sel, specs)) in enumerate(datasets.items()):
    for c, (sp, loader, argp, argm) in enumerate(specs):
        ax = axes[r][c]
        lo1, hi1, yp, ep, _ = loader(*argp); lo2, hi2, ym, em, _ = loader(*argm)
        ydat = 0.5 * (yp + ym); edat = 0.5 * np.sqrt(ep ** 2 + em ** 2); c_ = 0.5 * (lo1 + hi1)
        ax.fill_between(c_, 1 - edat / ydat, 1 + edat / ydat, color="0.85", label="data stat+sys" if (r == 0 and c == 0) else None)
        ax.axhline(1, color="k", lw=0.8)
        out.append(f"DATA {dname} {sp}: <pT>trunc {trunc_meanpt(lo1, hi1, ydat):.3f} over {lo1[0]:.2f}-{hi1[-1]:.2f} | yield(0.5-2.0) {integral(lo1, hi1, ydat, 0.5, 2.0):.4f}")
        for t in tunes:
            gp, gm = genkey[sp]; ygen = 0.5 * (np.array([rebin(G[t]["spec"][(gp, sel)]["y"], a, b) for a, b in zip(lo1, hi1)]) + np.array([rebin(G[t]["spec"][(gm, sel)]["y"], a, b) for a, b in zip(lo1, hi1)]))
            ratio = ygen / ydat; ok = np.isfinite(ratio)
            ax.plot(c_[ok], ratio[ok], "-o", ms=3, color=colors.get(t, None), label=t if (r == 0 and c == 0) else None)
            n = ok.sum(); half = n // 2
            out.append(f"  {t:7s} {dname} {sp}: mean ratio {np.nanmean(ratio):.3f} | low-half {np.nanmean(ratio[ok][:half]):.3f} high-half {np.nanmean(ratio[ok][half:]):.3f} | <pT>trunc gen {trunc_meanpt(lo1, hi1, ygen):.3f} vs data {trunc_meanpt(lo1, hi1, ydat):.3f} | yield(0.5-2.0) gen/data {integral(lo1, hi1, ygen, 0.5, 2.0)/integral(lo1, hi1, ydat, 0.5, 2.0):.3f}")
        ax.set_ylim(0.3, 2.2); ax.set_title(f"{sp}  —  {dname}", fontsize=10); ax.set_ylabel("Pythia / data")
        if r == 1: ax.set_xlabel("pT [GeV/c]")
axes[0][0].legend(fontsize=8, loc="upper left")
fig.suptitle(f"pp 200 GeV identified spectra: generator tunes vs PHENIX (PRC 83 064903) and STAR (PLB 616)  [{tag}]", fontsize=11)
fig.tight_layout(); os.makedirs(os.path.join(VDIR, "plots"), exist_ok=True); os.makedirs(os.path.join(VDIR, "ledgers"), exist_ok=True)
png = os.path.join(VDIR, "plots", f"world_spectra_confront_{tag}.png"); fig.savefig(png, dpi=130)
# species ratios in a common range from the tabulated points
for dname, (sel, specs) in datasets.items():
    vals = {}
    for sp, loader, argp, argm in specs:
        lo1, hi1, yp, ep, _ = loader(*argp); lo2, hi2, ym, em, _ = loader(*argm); vals[sp] = integral(lo1, hi1, 0.5 * (yp + ym), 0.5, 1.7)
        for t in tunes:
            gp, gm = genkey[sp]; yg = 0.5 * (np.array([rebin(G[t]["spec"][(gp, sel)]["y"], a, b) for a, b in zip(lo1, hi1)]) + np.array([rebin(G[t]["spec"][(gm, sel)]["y"], a, b) for a, b in zip(lo1, hi1)]))
            vals[(t, sp)] = integral(lo1, hi1, yg, 0.5, 1.7)
    out.append(f"RATIOS {dname} (0.5-1.7 GeV/c): data K/pi {vals['K']/vals['pi']:.3f} p/pi {vals['p']/vals['pi']:.3f} | " + " | ".join(f"{t} K/pi {vals[(t,'K')]/vals[(t,'pi')]:.3f} p/pi {vals[(t,'p')]/vals[(t,'pi')]:.3f}" for t in tunes))
led = os.path.join(VDIR, "ledgers", f"world_spectra_confront_{tag}.txt"); open(led, "w").write("\n".join(out) + "\n")
print("\n".join(out)); print("wrote", png, led)
