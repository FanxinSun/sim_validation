# Figure notes

Caveats and supersessions for figures in `plots/`, for readers of the figure
folder itself. Frozen figures are never edited or overwritten — when one is
found to be wrong, it stays as the era record and the correction is delivered
under a new tag, with the reason recorded here.

## residual_fingerprint, v6.1 baseline — panel 6, all-99 leg (2026-09-05)

`residual_fingerprint_v61.png` (2026-08-24 era record) and
`residual_fingerprint_v61r.png` (its byte-identical 2026-09-05 replot) stay
byte-frozen as era records and carry a **known defect in panel 6**: the R1, R2,
R3 and `adc<30` rows were counted with cut strings that lacked the laser veto
while being divided by the vetoed event count, so their real legs include real
event 44 (the GL1 diffuse laser flash) and the bars understate the residual —
the `total` bar is correct, because it alone was built from `CANON::TPC_CUT`.
For any reading of panel 6 these two figures are **superseded by
`residual_fingerprint_v61rfix.png`**, where `adc<30` reads −34.4%, matching the
acceptance battery's `lowadc_per_event` target for the all-99 reference.
The complete-61 figures — `residual_fingerprint_v61c.png`, `_v61rc.png` and
`_v61rcfix.png` — were **never affected**, because their real file contains only
the 61 complete non-laser events by construction; `_v61rcfix.png` is
byte-identical to `_v61rc.png`, which is the control that demonstrates it.
**The current all-99 fingerprint of the v6.1 baseline is
`residual_fingerprint_v61rfix.png`.**

Panels 1–5 are unaffected in every one of these figures: they were always built
from `CANON::TPC_CUT` / `CANON::TPC_CUT_NOLASER`. A pixel diff of `_v61rfix`
against `_v61r` finds zero changed pixels in panels 1–5 and all 4382 changed
pixels inside panel 6. No ledger row is affected either — `ledgers/residuals_v61.txt`
takes its low-adc numbers from the acceptance battery, which was always consistent.

The veto is `CANON::LASER_VETO` (`include/canon.h:24`, `"event!=44"`); the fixed
rows append it rather than spelling the condition out inline.

Commits: **69a1da9** (v61r replot, which uncovered the discrepancy) and
**1430887** (the fix and the `_v61rfix` / `_v61rcfix` renders).
