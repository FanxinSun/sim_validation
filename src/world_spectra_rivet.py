#!/usr/bin/env python3
"""world_spectra_rivet.py — CANONICAL (Rivet) confrontation of the reshape-campaign
Pythia tunes with the world pp-200 identified spectra.

Companion to world_spectra_confront.py, which does the same comparison with hand-written
fiducial definitions.  Here the definitions come from the experiments' own Rivet
routines and the normalisation from the generator's own cross section, so the two can be
differenced and the size of each convention read off.

Inputs : P5/angantyr/yoda/<tune>.yoda          (rivet output, see P5/angantyr/run_rivet.sh)
         P5/angantyr/rivet/PHENIX_2011_I886590.yoda   (refs built from HEPData ins886590)
         $RIVET_PREFIX/share/Rivet/STAR_*.yoda.gz     (refs shipped with Rivet)
         <checkout>/ledgers/world_spectra_confront_v7tune.txt  (the hand-rolled numbers)
Outputs: <checkout>/ledgers/rivet/world_spectra_rivet_<tag>.txt
         <checkout>/plots/rivet_world_spectra_{phenix,star,conventions}.png
Usage  : . P5/angantyr/rivet_env.sh && python3 world_spectra_rivet.py [tag]
"""
import os
import re
import sys
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yoda
except ImportError:
    sys.exit("yoda not importable -- source P5/angantyr/rivet_env.sh first")

HERE = os.path.dirname(os.path.abspath(__file__))
VDIR = os.path.realpath(os.path.join(HERE, ".."))
REPO = "/home/rog/sPHENIX/3D_ClusterFindingML"
YDIR = os.path.join(REPO, "P5", "angantyr", "yoda")
OURREF = os.path.join(REPO, "P5", "angantyr", "rivet", "PHENIX_2011_I886590.yoda")
RIVETSHARE = "/home/rog/sw/rivet-4.1.3/share/Rivet"
TAG = sys.argv[1] if len(sys.argv) > 1 else "2026-09"

TUNES = ["ours", "monash", "mdc2", "mdc2_nolim", "w1"]
SEEDS = {"ours": 20261001, "monash": 20261002, "mdc2": 20261003,
         "mdc2_nolim": 20261005, "w1": 20261004}
SETTINGS = {
    "ours":       "Monash + MultipartonInteractions:pT0Ref = 1.85",
    "monash":     "Monash + MultipartonInteractions:pT0Ref = 2.28",
    "mdc2":       "readFile island_post/official_pp_mb_mdc2.cfg  (sPHENIX HF TG MDC2 = Detroit)",
    "mdc2_nolim": "as mdc2 but ParticleDecays:limitTau0 = off  (control for the cfg's tau0Max = 1e-7 mm)",
    "w1":         "Monash + pT0Ref 2.55, StringPT:sigma 0.36, StringFlav:probStoUD 0.27, probQQtoQ 0.11",
}
PHX = "/PHENIX_2011_I886590"
PHX_ETA = "/PHENIX_2011_I886590:acc=eta"
DY_PHENIX = 0.7                    # |y| < 0.35
COLORS = {"ours": "#A8218E", "monash": "#1F4E9C", "mdc2": "#2E7D4F",
          "mdc2_nolim": "#7BAE8B", "w1": "#B8471B"}

# Species -> (invariant-histogram key pairs, counts-twin key pairs).  Several pairs are
# concatenated into one bin set: PHENIX Tables 3 and 4 carry two overlapping pT series
# (d03/d04 fine, d09/d10 coarse) whose union is exactly the 34 tabulated points the
# hand-rolled confrontation ran over, so the two mean ratios are directly differenceable.
PHENIX_SPECIES = {
    "pi": ([("d01-x01-y01", "d01-x01-y02")], [("n_pip", "n_pim")]),
    "K":  ([("d02-x01-y01", "d02-x01-y02")], [("n_kp",  "n_km")]),
    "p":  ([("d04-x01-y01", "d04-x01-y02"), ("d10-x01-y01", "d10-x01-y02")],
           [("n_p_fd", "n_pbar_fd"), ("n_p_fd_c", "n_pbar_fd_c")]),        # feed-down corrected
    "p_inc": ([("d03-x01-y01", "d03-x01-y02"), ("d09-x01-y01", "d09-x01-y02")],
              [("n_p_inc", "n_pbar_inc"), ("n_p_inc_c", "n_pbar_inc_c")]),  # inclusive
}
# STAR_2005_I628232 (PLB 616) -- the STAR measurement the hand-rolled confrontation used.
STAR05_SPECIES = {
    "pi":   ([("d01-x01-y01", "d01-x01-y02")], [("n_pip", "n_pim")]),
    "K":    ([("d02-x01-y01", "d02-x01-y02")], [("n_kp",  "n_km")]),
    "p":    ([("d03-x01-y01", "d03-x01-y02")], [("n_p",   "n_pbar")]),   # inclusive, as published
    "p_fd": ([("fd_p",        "fd_pbar")],     [("n_p_fd", "n_pbar_fd")]),
}
STAR05_REF = {"pi": [("d01-x01-y01", "d01-x01-y02")], "K": [("d02-x01-y01", "d02-x01-y02")],
              "p": [("d03-x01-y01", "d03-x01-y02")], "p_fd": [("d03-x01-y01", "d03-x01-y02")]}
# STAR_2006_I709170 (PLB 637): bonus third reference. Pions and protons only, no kaons.
STAR06_SPECIES = {"pi": ([("d02-x01-y01", "d07-x01-y01")], None),
                  "p":  ([("d12-x01-y01", "d17-x01-y01")], None)}
# pT windows of the hand-rolled STAR comparison (ins628232 = PLB 616), for a like-for-like
# restriction of the wider STAR_2006_I709170 range.
STAR_HAND_WINDOW = {"pi": (0.35, 2.75), "p": (0.42, 3.88)}


# ---------------------------------------------------------------- yoda helpers
def vals(obj):
    """(xmid, xwidth, value, err) arrays for a binned YODA object, NaN where unfilled."""
    bs = list(obj.bins())
    x = np.array([b.xMid() for b in bs], dtype=float)
    w = np.array([b.xWidth() for b in bs], dtype=float)
    v = np.array([(b.val() if b.val() is not None else np.nan) for b in bs], dtype=float)
    e = np.array([(b.totalErrAvg() if b.val() is not None else np.nan) for b in bs], dtype=float)
    return x, w, v, e


def mean_err(err):
    """MC statistical uncertainty on the plain mean over bins (bins independent)."""
    ok = np.isfinite(err)
    return float(np.sqrt(np.nansum(err[ok]**2)) / ok.sum()) if ok.sum() else np.nan


def mean_ratio(ratio):
    """mean, low-half, high-half over the finite bins -- world_spectra_confront.py's split."""
    ok = np.isfinite(ratio)
    r = ratio[ok]
    if r.size == 0:
        return (np.nan, np.nan, np.nan, 0)
    half = r.size // 2
    return (float(np.mean(r)), float(np.mean(r[:half])), float(np.mean(r[half:])), int(r.size))


def species_ratio(mc, ref, keypairs, refpairs=None, window=None, centre=False, dy=DY_PHENIX):
    """Charge-averaged MC/data ratio per bin, the convention of world_spectra_confront.py.

    keypairs is a list of (plus, minus) histogram names; their bins are concatenated in
    order, which is how the two overlapping PHENIX proton series recover the tabulated
    34-point set.  centre=True reads the counts twins instead and forms
    N/(2 pi pT_CENTRE dpT dy) -- the bin-centre convention the hand-rolled code uses;
    otherwise the routine's own bin-averaged invariant quantity is used.
    """
    refpairs = refpairs or keypairs
    X, W, RAT, ERR = [], [], [], []
    for (kp, km), (rkp, rkm) in zip(keypairs, refpairs):
        gp = vals(mc[kp]); gm = vals(mc[km])
        rp = vals(ref[rkp]); rm = vals(ref[rkm])
        x, w = rp[0], rp[1]
        ygen = 0.5 * (gp[2] + gm[2])
        egen = 0.5 * np.sqrt(gp[3]**2 + gm[3]**2)      # MC statistical, charge-averaged
        if centre:
            ygen = ygen / (2 * math.pi * x * dy)
            egen = egen / (2 * math.pi * x * dy)
        ydat = 0.5 * (rp[2] + rm[2])
        X.append(x); W.append(w); RAT.append(ygen / ydat); ERR.append(egen / ydat)
    x = np.concatenate(X); w = np.concatenate(W)
    ratio = np.concatenate(RAT); err = np.concatenate(ERR)
    if window is not None:
        keep = (x >= window[0]) & (x <= window[1])
        ratio = np.where(keep, ratio, np.nan)
        err = np.where(keep, err, np.nan)
    return x, w, ratio, err


def hand_rolled(path):
    """Parse ledgers/world_spectra_confront_v7tune.txt -> {(tune, exp, species): (m, lo, hi)}."""
    out, dnde = {}, {}
    pat = re.compile(r"^\s+(\S+)\s+(PHENIX|STAR)[^:]*\s(pi|K|p): mean ratio ([\d.]+) \| "
                     r"low-half ([\d.]+) high-half ([\d.]+)")
    for line in open(path):
        m = pat.match(line)
        if m:
            out[(m.group(1), m.group(2), m.group(3))] = tuple(float(m.group(i)) for i in (4, 5, 6))
        m2 = re.match(r"^TUNE (\S+): dNch/deta\(\|eta\|<0.5\) ([\d.]+)", line)
        if m2:
            dnde[m2.group(1)] = float(m2.group(2))
    return out, dnde


# ---------------------------------------------------------------- main
def main():
    ref_phx = yoda.read(OURREF)
    ref_phx = {k.split("/")[-1]: v for k, v in ref_phx.items()}
    ref_s05 = yoda.read(os.path.join(REPO, "P5", "angantyr", "rivet", "STAR_2005_I628232.yoda"))
    ref_s05 = {k.split("/")[-1]: v for k, v in ref_s05.items()}
    ref_s06 = yoda.read(os.path.join(RIVETSHARE, "STAR_2006_I709170.yoda.gz"))
    ref_s06 = {k.split("/")[-1]: v for k, v in ref_s06.items() if k.startswith("/REF/")}
    ref_s08 = yoda.read(os.path.join(RIVETSHARE, "STAR_2008_I793126.yoda.gz"))
    ref_s08 = {k.split("/")[-1]: v for k, v in ref_s08.items() if k.startswith("/REF/")}
    HAND, HAND_DNDE = hand_rolled(os.path.join(VDIR, "ledgers", "world_spectra_confront_v7tune.txt"))

    D = {}
    for t in TUNES:
        d = yoda.read(os.path.join(YDIR, f"{t}.yoda"))
        D[t] = {k: v for k, v in d.items() if not k.startswith("/RAW")}

    R = {}          # (tune, key) -> (x, w, ratio)
    S = {}          # scalar summary numbers
    for t in TUNES:
        d = D[t]
        mc_y   = {k.split("/")[-1]: v for k, v in d.items() if k.startswith(PHX + "/")}
        mc_eta = {k.split("/")[-1]: v for k, v in d.items() if k.startswith(PHX_ETA + "/")}
        for sp, (inv, cnt) in PHENIX_SPECIES.items():
            R[(t, "PHENIX", sp, "y")]      = species_ratio(mc_y, ref_phx, inv)
            R[(t, "PHENIX", sp, "cen")]    = species_ratio(mc_y, ref_phx, cnt, refpairs=inv, centre=True)
            R[(t, "PHENIX", sp, "eta")]    = species_ratio(mc_eta, ref_phx, inv)
        mc_s05 = {k.split("/")[-1]: v for k, v in d.items() if k.startswith("/STAR_2005_I628232/")}
        for sp, (inv, cnt) in STAR05_SPECIES.items():
            R[(t, "STAR05", sp, "y")] = species_ratio(mc_s05, ref_s05, inv,
                                                      refpairs=STAR05_REF[sp], dy=1.0)
            R[(t, "STAR05", sp, "cen")] = species_ratio(mc_s05, ref_s05, cnt,
                                                        refpairs=STAR05_REF[sp],
                                                        centre=True, dy=1.0)
        mc_s06 = {k.split("/")[-1]: v for k, v in d.items() if k.startswith("/STAR_2006_I709170/")}
        for sp, (inv, _) in STAR06_SPECIES.items():
            R[(t, "STAR06", sp, "full")] = species_ratio(mc_s06, ref_s06, inv)
            R[(t, "STAR06", sp, "win")]  = species_ratio(mc_s06, ref_s06, inv,
                                                         window=STAR_HAND_WINDOW[sp])
        # STAR_2008: shape-only. Mean of the unit-normalised multiplicity distribution.
        xm, wm, vm, _ = vals(d["/STAR_2008_I793126/d01-x01-y01"])
        xr, wr, vr, _ = vals(ref_s08["d01-x01-y01"])
        okm, okr = np.isfinite(vm), np.isfinite(vr)
        S[(t, "s08_meanNch_mc")]   = float(np.sum(xm[okm] * vm[okm] * wm[okm]) / np.sum(vm[okm] * wm[okm]))
        S[(t, "s08_meanNch_data")] = float(np.sum(xr[okr] * vr[okr] * wr[okr]) / np.sum(vr[okr] * wr[okr]))
        # MC_PP200_MULT (generator level)
        S[(t, "dnde_inel")]   = float(d["/MC_PP200_MULT/nch05_inel"].val())
        S[(t, "dnde_nsd")]    = float(d["/MC_PP200_MULT/nch05_nsd"].val())
        S[(t, "dndept_inel")] = float(d["/MC_PP200_MULT/nch05pt_inel"].val())
        S[(t, "nsd_frac")]    = float(d["/MC_PP200_MULT/n_nsd"].val() / d["/MC_PP200_MULT/n_inel"].val())

    # ------------------------------------------------------------ the table
    o = []
    A = o.append
    A(f"# world_spectra_rivet {TAG} — CANONICAL confrontation of the reshape-campaign tunes")
    A(f"# with the experiments' own Rivet analysis definitions.  Companion to, and checked")
    A(f"# against, the hand-rolled ledgers/world_spectra_confront_v7tune.txt.")
    A(f"#")
    A(f"# Rivet 4.1.3 (YODA 2.1.3, HepMC3 3.03.01) — /home/rog/sw/rivet-4.1.3, built 2026-07-30.")
    A(f"# Generator: Pythia 8.317 (P5/angantyr/install), SoftQCD:inelastic, pp 200 GeV,")
    A(f"#   written as HepMC3 ASCII by P5/angantyr/gen_hepmc.cc via Pythia8ToHepMC3.")
    A(f"#   300000 events per sample; sigmaGen = 41.9734 mb for every tune (SoftQCD:inelastic")
    A(f"#   total cross section does not depend on pT0Ref) and it is what normalises the")
    A(f"#   spectra — no 42 mb assumption enters the canonical column.")
    for t in TUNES:
        A(f"#   {t:11s} seed {SEEDS[t]}  {SETTINGS[t]}")
    A(f"#")
    A(f"# Routines:")
    A(f"#   PHENIX_2011_I886590   local, P5/angantyr/rivet/ (PRC 83 064903; refs from HEPData")
    A(f"#                         ins886590 Tables 1-8).  Written for this job; the community")
    A(f"#                         routine (cnattras/RIVETAnalyses @ 1b79221b) is unusable —")
    A(f"#                         see the .info for the four defects.  UNVALIDATED.")
    A(f"#   STAR_2005_I628232     local, P5/angantyr/rivet/ (PLB 616 (2005) 8 / nucl-ex/0309012;")
    A(f"#                         refs from HEPData ins628232 fig2a/b/c, p+p NSD series).  This")
    A(f"#                         IS the STAR measurement the hand-rolled confrontation used, so")
    A(f"#                         section C differences conventions and nothing else.  STAR's own")
    A(f"#                         BBC-coincidence NSD selection, |y|<0.5.  UNVALIDATED.")
    A(f"#   STAR_2006_I709170     official, shipped with Rivet.  VALIDATED.  BONUS third")
    A(f"#                         reference only: it is PLB 637 (2006) 161 / nucl-ex/0601033,")
    A(f"#                         a DIFFERENT paper from ins628232, with pi to 10 and p to")
    A(f"#                         7 GeV/c and NO kaon observable.  Its value here is the high-pT")
    A(f"#                         reach.  Its rows are NOT a convention comparison.")
    A(f"#                         ERRATUM: the handover identified I709170 as PLB 616; it is not.")
    A(f"#   STAR_2008_I793126     official, shipped with Rivet.  UNVALIDATED and SHAPE-ONLY:")
    A(f"#                         its finalize() normalises the multiplicity distribution and")
    A(f"#                         each pT spectrum to a hard-coded data integral, and it folds")
    A(f"#                         STAR track/vertex efficiencies into the MC.  No absolute yield.")
    A(f"#   MC_PP200_MULT         local, auxiliary, NO reference data: generator-level dNch/deta")
    A(f"#                         from the same event loop, for the multiplicity leg.")
    A(f"#")
    A(f"# Columns per (tune, experiment, species): mean MC/data ratio over the data bins and")
    A(f"# its low-half / high-half split by bin index — the convention of")
    A(f"# world_spectra_confront.py, so the numbers are directly differenceable.")
    A("")

    A("== A: PHENIX PRC 83 064903, |y|<0.35, invariant cross section, canonical ==")
    A("   stat = MC statistical uncertainty on the mean ratio (300k events).  A d(mean)")
    A("   larger than 0.05 is only meaningful if it also exceeds this; see the flag column.")
    A(f"{'tune':11s} {'sp':3s} {'mean':>7s} {'stat':>6s} {'low':>7s} {'high':>7s} {'nbins':>6s}   "
      f"{'hand':>7s} {'hand.lo':>7s} {'hand.hi':>7s}   {'d(mean)':>8s} {'d/stat':>7s}  flag")
    for t in TUNES:
        for sp in ("pi", "K", "p"):
            m, lo, hi, n = mean_ratio(R[(t, "PHENIX", sp, "y")][2])
            se = mean_err(R[(t, "PHENIX", sp, "y")][3])
            h = HAND.get((t, "PHENIX", sp))
            if h:
                d = m - h[0]
                nsig = d / se if se > 0 else np.nan
                flag = "FLAG" if (abs(d) > 0.05 and abs(nsig) > 2) else (
                       "stat" if abs(d) > 0.05 else "ok")
                A(f"{t:11s} {sp:3s} {m:7.3f} {se:6.3f} {lo:7.3f} {hi:7.3f} {n:6d}   "
                  f"{h[0]:7.3f} {h[1]:7.3f} {h[2]:7.3f}   {d:+8.3f} {nsig:+7.1f}  {flag}")
            else:
                A(f"{t:11s} {sp:3s} {m:7.3f} {se:6.3f} {lo:7.3f} {hi:7.3f} {n:6d}   "
                  f"{'-':>7s} {'-':>7s} {'-':>7s}   {'-':>8s} {'-':>7s}  (no hand-rolled counterpart)")
    A("   flag: FLAG = differs by more than 0.05 AND by more than 2 MC sigma;")
    A("         stat = differs by more than 0.05 but is within 2 MC sigma, i.e. the 0.05")
    A("                threshold is below this row's statistical resolution;")
    A("         ok   = agrees within 0.05.")
    A("")

    A("== B: the same PHENIX comparison under each single convention change ==")
    A("   y      canonical, |y|<0.35, bin-averaged invariant cross section (column A)")
    A("   eta    |eta|<0.35 instead of |y|<0.35            (rapidity convention)")
    A("   cen    N_bin/(2 pi pT_CENTRE dpT dy)             (binning convention; hand-rolled uses this)")
    A("   inc    p/pbar inclusive instead of feed-down corrected  (feed-down convention, p only)")
    A(f"{'tune':11s} {'sp':3s} {'y':>7s} {'eta':>7s} {'cen':>7s} {'inc':>7s}   "
      f"{'d.rap':>7s} {'d.bin':>7s} {'d.feed':>7s}")
    for t in TUNES:
        for sp in ("pi", "K", "p"):
            my = mean_ratio(R[(t, "PHENIX", sp, "y")][2])[0]
            me = mean_ratio(R[(t, "PHENIX", sp, "eta")][2])[0]
            mc = mean_ratio(R[(t, "PHENIX", sp, "cen")][2])[0]
            mi = mean_ratio(R[(t, "PHENIX", "p_inc", "y")][2])[0] if sp == "p" else np.nan
            A(f"{t:11s} {sp:3s} {my:7.3f} {me:7.3f} {mc:7.3f} "
              f"{(f'{mi:7.3f}' if sp == 'p' else '      -')}   "
              f"{me-my:+7.3f} {mc-my:+7.3f} "
              f"{(f'{mi-my:+7.3f}' if sp == 'p' else '      -')}")
    A("")
    A("   Normalisation convention: the hand-rolled column divides the PHENIX cross sections")
    A("   by sigma_inel = 42.0 mb; the canonical column uses the generator's own")
    A("   sigmaGen = 41.9734 mb.  Ratio 42.0/41.9734 = 1.00063, i.e. +0.06% — negligible.")
    A("")

    A("== C: STAR PLB 616 (2005) 8, |y|<0.5, BBC-coincidence NSD, per-NSD-event yields ==")
    A("   Same data as the hand-rolled STAR column, so d(mean) is a convention difference.")
    A("   p    = inclusive p/pbar, as published (the paper states no feed-down treatment)")
    A("   p_fd = the same with hyperon-ancestor protons removed (the hand-rolled definition)")
    A(f"{'tune':11s} {'sp':5s} {'mean':>7s} {'stat':>6s} {'low':>7s} {'high':>7s} {'nbins':>6s}   "
      f"{'hand':>7s} {'hand.lo':>7s} {'hand.hi':>7s}   {'d(mean)':>8s} {'d/stat':>7s} {'cen':>7s}  flag")
    for t in TUNES:
        for sp in ("pi", "K", "p", "p_fd"):
            m, lo, hi, n = mean_ratio(R[(t, "STAR05", sp, "y")][2])
            se = mean_err(R[(t, "STAR05", sp, "y")][3])
            mcv = mean_ratio(R[(t, "STAR05", sp, "cen")][2])[0]
            h = HAND.get((t, "STAR", "p" if sp == "p_fd" else sp))
            if h:
                d = m - h[0]
                nsig = d / se if se > 0 else np.nan
                flag = "FLAG" if (abs(d) > 0.05 and abs(nsig) > 2) else (
                       "stat" if abs(d) > 0.05 else ("ok*" if abs(nsig) > 2 else "ok"))
                A(f"{t:11s} {sp:5s} {m:7.3f} {se:6.3f} {lo:7.3f} {hi:7.3f} {n:6d}   "
                  f"{h[0]:7.3f} {h[1]:7.3f} {h[2]:7.3f}   {d:+8.3f} {nsig:+7.1f} {mcv:7.3f}  {flag}")
            else:
                A(f"{t:11s} {sp:5s} {m:7.3f} {se:6.3f} {lo:7.3f} {hi:7.3f} {n:6d}   "
                  f"{'-':>7s} {'-':>7s} {'-':>7s}   {'-':>8s} {'-':>7s} {mcv:7.3f}  (no hand-rolled counterpart)")
    A("   ok* = within the 0.05 threshold but more than 2 MC sigma from the hand-rolled")
    A("         value: a real, small, systematic offset (the NSD definition).")
    A("   The hand-rolled STAR column used the hyperon-excluded proton definition, so the")
    A("   p_fd row is its like-for-like counterpart; the p row shows the size of that choice.")
    A("")

    A("== C2 (bonus): STAR_2006_I709170 (PLB 637) — a DIFFERENT measurement, high-pT reach ==")
    A(f"{'tune':11s} {'sp':3s} {'mean':>7s} {'low':>7s} {'high':>7s} {'nbins':>6s} | "
      f"{'win.mean':>8s} {'win.lo':>7s} {'win.hi':>7s} {'n':>4s}")
    for t in TUNES:
        for sp in ("pi", "p"):
            m, lo, hi, n = mean_ratio(R[(t, "STAR06", sp, "full")][2])
            wm, wlo, whi, wn = mean_ratio(R[(t, "STAR06", sp, "win")][2])
            A(f"{t:11s} {sp:3s} {m:7.3f} {lo:7.3f} {hi:7.3f} {n:6d} | "
              f"{wm:8.3f} {wlo:7.3f} {whi:7.3f} {wn:4d}")
    A("   win.* restricts to the pT window of PLB 616 for orientation only; the rows are not")
    A("   a canonical-vs-hand-rolled comparison.  No kaon row exists in this routine.")
    A("")

    A("== D: multiplicity ==")
    A("   STAR_2008_I793126 d01 is the |eta|<0.5, pT>0.2 GeV/c charged multiplicity")
    A("   distribution, unit-normalised by the routine, with STAR track+vertex efficiencies")
    A("   folded into the MC.  Its MEAN survives the normalisation and is compared here.")
    A("   MC_PP200_MULT is generator level (no efficiency, no data), for the world-value leg.")
    A(f"{'tune':11s} {'<Nch>MC':>8s} {'<Nch>dat':>8s} {'ratio':>7s} | "
      f"{'dNch/deta':>9s} {'(NSD)':>7s} {'pT>0.2':>7s} {'NSDfrac':>8s} | {'hand dNch/deta':>14s}")
    for t in TUNES:
        hd = HAND_DNDE.get(t)
        hds = f"{hd:14.3f}" if hd else f"{'-':>14s}"
        A(f"{t:11s} {S[(t,'s08_meanNch_mc')]:8.3f} {S[(t,'s08_meanNch_data')]:8.3f} "
          f"{S[(t,'s08_meanNch_mc')]/S[(t,'s08_meanNch_data')]:7.3f} | "
          f"{S[(t,'dnde_inel')]:9.3f} {S[(t,'dnde_nsd')]:7.3f} {S[(t,'dndept_inel')]:7.3f} "
          f"{S[(t,'nsd_frac')]:8.3f} | {hds}")
    A("   NSDfrac is the STAR BBC coincidence fraction (>=1 charged in -5.0<eta<-3.3 AND in")
    A("   3.3<eta<5.0); the hand-rolled code called an event NSD by Pythia process code")
    A("   (not 103/104), which gave 0.778-0.779 for every tune.  The NSD DEFINITION is thus")
    A("   itself a convention worth about 0.12-0.17 in accepted fraction.")
    A("   World anchor for A4: dNch/deta(|eta|<0.5) = 2.2-2.4 (the band pre-registered in the")
    A("   handover; the pipeline session supplies PHOBOS 2.29 +- 0.08 inelastic and PHENIX")
    A("   2.38 +- 0.17, NOT independently verified in this session).  No Rivet routine on this")
    A("   box carries a pp 200 GeV dN/deta reference, so this leg is a GENERATOR-LEVEL number")
    A("   against published values, not a Rivet confrontation.")
    A("")

    A("== E: mdc2 decay-handling control ==")
    A("   island_post/official_pp_mb_mdc2.cfg sets ParticleDecays:limitTau0 = on with")
    A("   tau0Max = 1e-7 mm, so Pythia leaves K0S, Lambda (and pi0) undecayed in the mdc2")
    A("   sample.  mdc2_nolim is the same tune with limitTau0 = off.  Difference = the part")
    A("   of the mdc2 signature that is decay handling rather than tune.  Note that in the")
    A("   mdc2 sample the inclusive and feed-down-corrected proton rows of section C are")
    A("   IDENTICAL (0.561 both), which is the direct proof that no hyperon decays occur.")
    A(f"{'observable':34s} {'mdc2':>9s} {'mdc2_nolim':>11s} {'delta':>9s}")
    for sp in ("pi", "K", "p"):
        a = mean_ratio(R[("mdc2", "PHENIX", sp, "y")][2])[0]
        b = mean_ratio(R[("mdc2_nolim", "PHENIX", sp, "y")][2])[0]
        A(f"{'PHENIX mean ratio ' + sp:34s} {a:9.3f} {b:11.3f} {b-a:+9.3f}")
    for sp in ("pi", "K", "p", "p_fd"):
        a = mean_ratio(R[("mdc2", "STAR05", sp, "y")][2])[0]
        b = mean_ratio(R[("mdc2_nolim", "STAR05", sp, "y")][2])[0]
        A(f"{'STAR05 (PLB 616) mean ratio ' + sp:34s} {a:9.3f} {b:11.3f} {b-a:+9.3f}")
    for sp in ("pi", "p"):
        a = mean_ratio(R[("mdc2", "STAR06", sp, "win")][2])[0]
        b = mean_ratio(R[("mdc2_nolim", "STAR06", sp, "win")][2])[0]
        A(f"{'STAR06 mean ratio (window) ' + sp:34s} {a:9.3f} {b:11.3f} {b-a:+9.3f}")
    for k, lbl in (("dnde_inel", "dNch/deta(|eta|<0.5) inel"),
                   ("dnde_nsd", "dNch/deta(|eta|<0.5) NSD"),
                   ("s08_meanNch_mc", "<Nch> STAR_2008 (folded)")):
        a, b = S[("mdc2", k)], S[("mdc2_nolim", k)]
        A(f"{lbl:34s} {a:9.3f} {b:11.3f} {b-a:+9.3f}")
    A("")

    out = os.path.join(VDIR, "ledgers", "rivet", f"world_spectra_rivet_{TAG}.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(o) + "\n")
    print("\n".join(o))

    # ------------------------------------------------------------ figures
    os.makedirs(os.path.join(VDIR, "plots"), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for c, sp in enumerate(("pi", "K", "p")):
        ax = axes[c]
        ax.axhline(1, color="k", lw=0.8)
        for t in TUNES:
            x, w, r, _e = R[(t, "PHENIX", sp, "y")]
            ok = np.isfinite(r)
            ax.plot(x[ok], r[ok], "-o", ms=3, color=COLORS[t], label=t if c == 0 else None)
        ax.set_ylim(0.3, 3.0)
        ax.set_title(f"{sp} — PHENIX PRC 83 064903 (Rivet, |y|<0.35)", fontsize=10)
        ax.set_xlabel("$p_T$ [GeV/$c$]")
        ax.set_ylabel("Pythia / data")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("pp 200 GeV identified spectra, canonical Rivet confrontation — PHENIX", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(VDIR, "plots", "rivet_world_spectra_phenix.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for c, sp in enumerate(("pi", "K", "p_fd")):
        ax = axes[c]
        ax.axhline(1, color="k", lw=0.8)
        for t in TUNES:
            x, w, r, _e = R[(t, "STAR05", sp, "y")]
            ok = np.isfinite(r)
            ax.plot(x[ok], r[ok], "-o", ms=3, color=COLORS[t], label=t if c == 0 else None)
        ax.set_ylim(0.0, 2.5)
        ax.set_title(f"{sp} — STAR PLB 616 (Rivet, |y|<0.5, BBC NSD)", fontsize=10)
        ax.set_xlabel("$p_T$ [GeV/$c$]"); ax.set_ylabel("Pythia / data")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("pp 200 GeV identified spectra, canonical Rivet confrontation — STAR PLB 616",
                 fontsize=11)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(VDIR, "plots", "rivet_world_spectra_star.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.2))
    xr, wr, vr, _ = vals(ref_s08["d01-x01-y01"])
    ax.step(xr, vr, where="mid", color="k", lw=1.2, label="STAR data")
    for t in TUNES:
        xm, wm, vm, _ = vals(D[t]["/STAR_2008_I793126/d01-x01-y01"])
        ax.step(xm, vm, where="mid", color=COLORS[t], lw=1.0, label=t)
    ax.set_yscale("log"); ax.set_xlabel("$N_{ch}$ ($|\\eta|<0.5$, $p_T>0.2$ GeV/$c$)")
    ax.set_ylabel("normalised $P(N_{ch})$")
    ax.set_title("STAR_2008_I793126 (shape only)", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(VDIR, "plots", "rivet_world_spectra_multiplicity.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    base = [t for t in TUNES if t != "mdc2_nolim"]
    for c, sp in enumerate(("pi", "K", "p")):
        ax = axes[c]
        xs = np.arange(len(base))
        can = [mean_ratio(R[(t, "PHENIX", sp, "y")][2])[0] for t in base]
        han = [HAND.get((t, "PHENIX", sp), (np.nan,))[0] for t in base]
        cen = [mean_ratio(R[(t, "PHENIX", sp, "cen")][2])[0] for t in base]
        eta = [mean_ratio(R[(t, "PHENIX", sp, "eta")][2])[0] for t in base]
        ax.plot(xs, han, "s--", color="0.4", label="hand-rolled" if c == 0 else None)
        ax.plot(xs, can, "o-", color="#A8218E", label="canonical |y|<0.35" if c == 0 else None)
        ax.plot(xs, cen, "^:", color="#1F4E9C", label="canonical, bin-centre" if c == 0 else None)
        ax.plot(xs, eta, "v:", color="#2E7D4F", label="canonical |eta|<0.35" if c == 0 else None)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xticks(xs); ax.set_xticklabels(base, rotation=20, fontsize=8)
        ax.set_title(f"{sp} — mean ratio vs PHENIX", fontsize=10)
        ax.set_ylabel("mean Pythia / data")
    axes[0].legend(fontsize=8)
    fig.suptitle("Canonical vs hand-rolled mean ratios, and the size of each convention", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(VDIR, "plots", "rivet_world_spectra_conventions.png"), dpi=130)
    plt.close(fig)

    print("\nwrote", out)
    for n in ("phenix", "star", "multiplicity", "conventions"):
        print("wrote", os.path.join(VDIR, "plots", f"rivet_world_spectra_{n}.png"))


if __name__ == "__main__":
    main()
