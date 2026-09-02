#!/usr/bin/env python3
"""world_tune_pick.py — rank reshape-scan points against PHENIX pp-200 identified spectra + world
multiplicity (reshape campaign, 2026-09-02). chi2 = sum over PHENIX bins (pi 0.3-3.0, K 0.4-2.0,
p 0.5-2.5; data stat+sys (+) 5% gen-stat floor, in quadrature) + ((dNch/deta - 2.33)/0.12)^2.
Usage: python3 world_tune_pick.py <tag> <label-glob-prefix>   e.g.  v7scan scan_
Reads P5/angantyr/<label>_world.txt; writes <checkout>/ledgers/world_tune_pick_<tag>.txt
"""
import sys, os, glob, math, yaml
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); VDIR = os.path.realpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("wsc", os.path.join(HERE, "world_spectra_confront.py"))
# reuse loaders without executing the script body: re-implement the small pieces here
REPO = "/home/rog/sPHENIX/3D_ClusterFindingML"; GEN = os.path.join(REPO, "P5/angantyr"); DATA = os.path.join(REPO, "external_data/world_pp200_spectra")
SIG = 42.0; DPT = 0.05; NB = 100; PTC = (np.arange(NB) + 0.5) * DPT
DNDE_T, DNDE_E = 2.33, 0.12
tag = sys.argv[1] if len(sys.argv) > 1 else "v7scan"; prefix = sys.argv[2] if len(sys.argv) > 2 else "scan_"

def load_gen(path):
    g = {"spec": {}}
    for line in open(path):
        p = line.split()
        if line.startswith("SUMMARY"): g["dnde"] = float(p[6]); g["eps"] = float(p[8]); g["meanpt"] = float(p[15]); g["classN"] = [float(x) for x in p[10:14]]
        elif line.startswith("SPEC"): g["spec"][(p[1], p[2])] = np.array([float(x) for x in p[7:7 + NB]])
    return g
def rebin(y, lo, hi):
    m = (PTC >= lo) & (PTC < hi); return float((y[m] * PTC[m]).sum() * DPT / (0.5 * (lo + hi) * (hi - lo))) if m.any() else np.nan
def phenix(table, idx):
    d = yaml.safe_load(open(os.path.join(DATA, "ins886590", f"Table{table}.yaml")))
    xs = d["independent_variables"][0]["values"]; dv = d["dependent_variables"][idx]
    lo = np.array([x["low"] for x in xs]); hi = np.array([x["high"] for x in xs])
    y = np.array([v["value"] for v in dv["values"]]) / SIG
    e = np.array([math.sqrt(sum(float(q.get("symerror", 0)) ** 2 for q in v.get("errors", []))) for v in dv["values"]]) / SIG
    return lo, hi, y, e
species = {"pi": ((1, 0), (1, 1), ("pip", "pim"), 3.0), "K": ((2, 0), (2, 1), ("kp", "km"), 2.0), "p": ((4, 0), (4, 1), ("p", "pbar"), 2.5)}
D = {}
for sp, (ap, am, gk, ptmax) in species.items():
    lo, hi, yp, ep = phenix(*ap); _, _, ym, em = phenix(*am)
    m = hi <= ptmax + 1e-9; D[sp] = (lo[m], hi[m], 0.5 * (yp + ym)[m], 0.5 * np.sqrt(ep ** 2 + em ** 2)[m])
rows = []
for f in sorted(glob.glob(os.path.join(GEN, f"{prefix}*_world.txt"))):
    lab = os.path.basename(f)[:-10]; g = load_gen(f); chi = {}; nb = 0
    for sp, (ap, am, gk, ptmax) in species.items():
        lo, hi, yd, ed = D[sp]
        yg = 0.5 * (np.array([rebin(g["spec"][(gk[0], "sel0")], a, b) for a, b in zip(lo, hi)]) + np.array([rebin(g["spec"][(gk[1], "sel0")], a, b) for a, b in zip(lo, hi)]))
        err = np.sqrt(ed ** 2 + (0.05 * yd) ** 2); chi[sp] = float((((yg - yd) / err) ** 2).sum()); nb += len(yd)
        r = yg / yd; ok = np.isfinite(r); h = ok.sum() // 2
        chi[sp + "_r"] = float(np.nanmean(r)); chi[sp + "_lo"] = float(np.nanmean(r[ok][:h])); chi[sp + "_hi"] = float(np.nanmean(r[ok][h:]))
    chi_d = ((g["dnde"] - DNDE_T) / DNDE_E) ** 2; tot = chi["pi"] + chi["K"] + chi["p"] + chi_d
    rows.append((tot, lab, g, chi, chi_d, nb))
rows.sort(key=lambda r: r[0])
out = [f"# world_tune_pick {tag}: chi2 vs PHENIX pi/K/p (|y|<0.35, per inelastic, 5% gen floor) + dNch/deta target {DNDE_T}+-{DNDE_E}; {len(rows)} points"]
out.append(f"{'rank':>4} {'label':38s} {'chi2tot':>8} {'chi2pi':>7} {'chi2K':>7} {'chi2p':>7} {'chi2dn':>7} {'dnde':>6} {'<r>pi':>6} {'<r>K':>6} {'<r>p':>6} {'pi lo/hi':>11} {'K lo/hi':>11} {'eps':>6} {'<pT>':>6}")
for i, (tot, lab, g, chi, chd, nb) in enumerate(rows):
    out.append(f"{i+1:4d} {lab:38s} {tot:8.1f} {chi['pi']:7.1f} {chi['K']:7.1f} {chi['p']:7.1f} {chd:7.1f} {g['dnde']:6.3f} {chi['pi_r']:6.3f} {chi['K_r']:6.3f} {chi['p_r']:6.3f} {chi['pi_lo']:5.2f}/{chi['pi_hi']:4.2f} {chi['K_lo']:5.2f}/{chi['K_hi']:4.2f} {g['eps']:6.3f} {g['meanpt']:6.3f}")
os.makedirs(os.path.join(VDIR, "ledgers"), exist_ok=True)
open(os.path.join(VDIR, "ledgers", f"world_tune_pick_{tag}.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
