// circlefit.h — THE circle fitter shared by every sim-validation meter (2026-09-11).
//
// Why one header: until 2026-09-11 the fitter lived as nine near-verbatim copies
// (ms_nofinder, missed_tracks, ms_split, ms_real, truth_circle, ms_fieldcmp,
// ms_barscan in src/; fieldmeter, cmcheck, pilot_compmeter in island_post/) —
// Kasa algebraic start + a FIXED number of undamped Gauss-Newton steps with no
// convergence test.  CODEX (CODEX_REAL_FIT_VALIDATION.md) and the Real-Data-Probe
// thread (island_post/fitpred_probe.C) showed 2.1% of real and 3.5% of sim 4-layer
// windows unconverged, with 73% of the sim windows above 3 mm being numerical.
//
// What it does (reference: namespace FPP in island_post/fitpred_probe.C, CODEX-
// verified to three decimals): in centred+scaled coordinates the curve is
// parametrised by (theta, d, kappa) — normal n = (cos th, sin th), passing through
// d*n, curvature kappa; kappa = 0 IS the straight line, so the parameter space
// contains the line and stiff tracks cannot drive the fit into a singular step.
// Exact signed distance
//   f = (k V - 2 U) / (1 + sqrt(1 - 2 k U + k^2 V)),  U = n.p - d,  V = |p|^2 - 2 d n.p + d^2,
// analytic Jacobian, Levenberg-Marquardt to |grad| < 1e-12 or step < 1e-13
// (<= 200 iterations).  Starts: the OLD estimator (kept verbatim below as
// kasaGN, or a seed the caller supplies), the Kasa circle, and the PCA line
// (the exact least-squares line, the kappa = 0 corner).  The best of all
// candidates INCLUDING the old estimator's own result is returned, so on identical
// input points the new fitter can never return a larger RMS than the old one.
//
// Interface (unchanged for every consumer): Fit {a, b, R, rms, n, ok} in the
// caller's units (cm); rms = sqrt(mean squared radial residual) over ALL points.
// A near-line optimum (|kappa| < 1e-12 in scaled units) is returned as a = b = 0,
// R = 1e99, rms = the line's rms, ok = true: bar gates of the form 45 <= R < 2e4
// reject it exactly as they reject straight tracks today; window fits (no R gate)
// keep the line's rms; residuals computed as hypot(x-a, y-b) - R come out as
// -1e99 and are rejected by any |res| cut, never NaN.
#ifndef SIMVAL_CIRCLEFIT_H
#define SIMVAL_CIRCLEFIT_H

#include <vector>
#include <cmath>
#include <algorithm>

namespace CircleFit
{
struct Fit
{
  double a = 0, b = 0, R = 0, rms = 0; int n = 0; bool ok = false;
  // 2026-09-11 (CODEX O06): ok means "finite objective, usable rms/(a,b,R)"; it is NOT a
  // convergence certificate.  status tells how the returned optimum was reached:
  //   0 not fitted | 1 gradient < 1e-12 | 2 no accepted LM step (stalled) |
  //   3 gain/step below tolerance | 4 iteration cap | 5 old estimator's own result kept
  // isLine: the optimum is the straight line (|kappa| < 1e-12); a = b = 0, R = 1e99 then
  // do NOT represent the curve — use rms (exact) or refit; radius gates reject it.
  int status = 0; bool isLine = false;
};

// ---- the pre-2026-09-11 estimator, verbatim (Kasa + nGN undamped GN steps) ----
// Kept (a) as one of the starts and (b) as the non-regression floor: its own
// result is always a candidate.  minN = the caller's historical minimum.
inline Fit kasaGN(const std::vector<double> &X, const std::vector<double> &Y, int nGN = 6, int minN = 5)
{
  Fit F; F.n = (int) X.size();
  if (F.n < minN || F.n < 3) return F;
  double Sx = 0, Sy = 0, Sxx = 0, Syy = 0, Sxy = 0, Sxz = 0, Syz = 0, Sz = 0;
  for (size_t i = 0; i < X.size(); ++i)
  {
    double x = X[i], y = Y[i], z = x * x + y * y;
    Sx += x; Sy += y; Sxx += x * x; Syy += y * y; Sxy += x * y;
    Sxz += x * z; Syz += y * z; Sz += z;
  }
  double n = F.n;
  double det = Sxx * (Syy * n - Sy * Sy) - Sxy * (Sxy * n - Sy * Sx) + Sx * (Sxy * Sy - Syy * Sx);
  if (std::fabs(det) < 1e-9) return F;
  double A = (Sxz * (Syy * n - Sy * Sy) - Sxy * (Syz * n - Sy * Sz) + Sx * (Syz * Sy - Syy * Sz)) / det;
  double B = (Sxx * (Syz * n - Sy * Sz) - Sxz * (Sxy * n - Sy * Sx) + Sx * (Sxy * Sz - Syz * Sx)) / det;
  double C = (Sxx * (Syy * Sz - Syz * Sy) - Sxy * (Sxy * Sz - Syz * Sx) + Sxz * (Sxy * Sy - Syy * Sx)) / det;
  F.a = A / 2; F.b = B / 2;
  double r2 = C + F.a * F.a + F.b * F.b;
  if (r2 <= 0) return F;
  F.R = std::sqrt(r2);
  for (int it = 0; it < nGN; ++it)
  {
    double M[3][3] = {{0}}, v[3] = {0};
    for (size_t i = 0; i < X.size(); ++i)
    {
      double dx = X[i] - F.a, dy = Y[i] - F.b, rho = std::hypot(dx, dy);
      if (rho < 1e-9) continue;
      double res = rho - F.R, J[3] = {-dx / rho, -dy / rho, -1.};
      for (int p = 0; p < 3; ++p)
      {
        v[p] -= J[p] * res;
        for (int q = 0; q < 3; ++q) M[p][q] += J[p] * J[q];
      }
    }
    double d = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
             - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
             + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    if (std::fabs(d) < 1e-12) break;
    double d0 = (v[0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
               - M[0][1] * (v[1] * M[2][2] - M[1][2] * v[2])
               + M[0][2] * (v[1] * M[2][1] - M[1][1] * v[2])) / d;
    double d1 = (M[0][0] * (v[1] * M[2][2] - M[1][2] * v[2])
               - v[0] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
               + M[0][2] * (M[1][0] * v[2] - v[1] * M[2][0])) / d;
    double d2 = (M[0][0] * (M[1][1] * v[2] - v[1] * M[2][1])
               - M[0][1] * (M[1][0] * v[2] - v[1] * M[2][0])
               + v[0] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])) / d;
    F.a += d0; F.b += d1; F.R += d2;
  }
  double s2 = 0;
  for (size_t i = 0; i < X.size(); ++i)
  {
    double res = std::hypot(X[i] - F.a, Y[i] - F.b) - F.R;
    s2 += res * res;
  }
  F.rms = std::sqrt(s2 / n);
  F.ok = true;
  return F;
}

// ---- converged fitter -----------------------------------------------------------
namespace detail
{
struct Pt { double x, y; };
struct Curve   // in centred/scaled coordinates
{
  double th = 0, d = 0, k = 0, mse = 1e99; bool ok = false; int status = 0;
  double f(double u, double v, double *J = nullptr) const
  {
    double nx = std::cos(th), ny = std::sin(th), q = nx * u + ny * v, qt = -ny * u + nx * v;
    double U = q - d, V = u * u + v * v - 2 * d * q + d * d;
    double S = std::sqrt(std::max(0., 1 - 2 * k * U + k * k * V)), N = k * V - 2 * U, den = 1 + S;
    if (J)
    {
      double Sg = std::max(S, 1e-12);
      double dN[3] = {-2 * qt * (1 + k * d), 2 - 2 * k * U, V};
      double dS[3] = {-k * qt * (1 + k * d) / Sg, k * (1 - k * U) / Sg, (k * V - U) / Sg};
      for (int p = 0; p < 3; ++p) J[p] = dN[p] / den - N * dS[p] / (den * den);
    }
    return N / den;
  }
};

inline bool solve3(double M[3][3], double b[3], double x[3])
{
  double A[3][4];
  for (int i = 0; i < 3; ++i) { for (int j = 0; j < 3; ++j) A[i][j] = M[i][j]; A[i][3] = b[i]; }
  for (int j = 0; j < 3; ++j)
  {
    int p = j;
    for (int i = j + 1; i < 3; ++i) if (std::fabs(A[i][j]) > std::fabs(A[p][j])) p = i;
    if (std::fabs(A[p][j]) < 1e-30) return false;
    for (int c = 0; c < 4; ++c) std::swap(A[j][c], A[p][c]);
    double piv = A[j][j];
    for (int c = j; c < 4; ++c) A[j][c] /= piv;
    for (int i = 0; i < 3; ++i)
      if (i != j) { double f = A[i][j]; for (int c = j; c < 4; ++c) A[i][c] -= f * A[j][c]; }
  }
  for (int i = 0; i < 3; ++i) x[i] = A[i][3];
  return true;
}

inline double mseOf(const std::vector<Pt> &P, const Curve &F, double M[3][3] = nullptr, double g[3] = nullptr)
{
  if (M) for (int i = 0; i < 3; ++i) { g[i] = 0; for (int j = 0; j < 3; ++j) M[i][j] = 0; }
  double ss = 0;
  for (auto &p : P)
  {
    double J[3];
    double r = F.f(p.x, p.y, M ? J : nullptr);
    if (!std::isfinite(r)) return 1e99;
    ss += r * r;
    if (M) for (int i = 0; i < 3; ++i) { g[i] += r * J[i]; for (int j = 0; j < 3; ++j) M[i][j] += J[i] * J[j]; }
  }
  double n = P.size();
  if (M) for (int i = 0; i < 3; ++i) { g[i] /= n; for (int j = 0; j < 3; ++j) M[i][j] /= n; }
  return ss / n;
}

inline Curve refine(const std::vector<Pt> &P, Curve F, int maxit = 200)
{
  double lam = 1e-3;
  F.mse = mseOf(P, F); F.status = 4;   // 4 = ran to the iteration cap unless a break below says otherwise
  for (int it = 0; it < maxit; ++it)
  {
    double M[3][3], g[3];
    F.mse = mseOf(P, F, M, g);
    double gmax = std::max({std::fabs(g[0]), std::fabs(g[1]), std::fabs(g[2])});
    if (gmax < 1e-12) { F.status = 1; break; }
    bool acc = false; double gain = 0, step = 0;
    for (int t = 0; t < 20; ++t)
    {
      double A[3][3], b[3], x[3];
      for (int i = 0; i < 3; ++i) { b[i] = -g[i]; for (int j = 0; j < 3; ++j) A[i][j] = M[i][j]; A[i][i] += lam * std::max(M[i][i], 1e-9); }
      if (!solve3(A, b, x)) { lam *= 10; continue; }
      Curve T = F; T.th += x[0]; T.d += x[1]; T.k += x[2]; T.mse = mseOf(P, T);
      if (T.mse < F.mse)
      {
        gain = F.mse - T.mse; step = std::max({std::fabs(x[0]), std::fabs(x[1]), std::fabs(x[2])});
        F.th = T.th; F.d = T.d; F.k = T.k; F.mse = T.mse;
        lam = std::max(lam * 0.3, 1e-15); acc = true; break;
      }
      lam *= 10;
    }
    if (!acc) { F.status = 2; break; }
    if (gain < 1e-15 * (1 + F.mse) || step < 1e-13) { F.status = 3; break; }
  }
  F.ok = std::isfinite(F.mse) && F.mse < 1e98;
  return F;
}
}  // namespace detail

// Converged fit.  minN = the caller's historical minimum point count (returns
// ok = false below it, exactly as before).  seed = the caller's OLD estimator
// result, if it differs from kasaGN(6); when null, kasaGN(6, minN) is used.
inline Fit fitCircle(const std::vector<double> &X, const std::vector<double> &Y, int minN = 5, const Fit *seed = nullptr)
{
  using namespace detail;
  Fit old = seed ? *seed : kasaGN(X, Y, 6, minN);
  Fit F; F.n = (int) X.size();
  if (F.n < minN || F.n < 3) return F;
  // centre + scale
  double mx = 0, my = 0;
  for (size_t i = 0; i < X.size(); ++i) { mx += X[i]; my += Y[i]; }
  mx /= F.n; my /= F.n;
  double xx = 0, yy = 0, xy = 0;
  for (size_t i = 0; i < X.size(); ++i) { double x = X[i] - mx, y = Y[i] - my; xx += x * x; yy += y * y; xy += x * y; }
  double s = std::sqrt((xx + yy) / F.n);
  if (s < 1e-10) return old;   // all points coincide: nothing to fit
  std::vector<Pt> P(X.size());
  for (size_t i = 0; i < X.size(); ++i) P[i] = {(X[i] - mx) / s, (Y[i] - my) / s};

  Curve best; best.mse = 1e99; best.ok = false;
  bool bestIsOld = false;
  auto consider = [&](const Curve &F0, bool isOld)
  {
    Curve R = refine(P, F0);
    if (R.ok && R.mse < best.mse) { best = R; bestIsOld = false; }
    if (isOld)   // the old result ITSELF, unrefined, is a candidate (non-regression floor)
    {
      Curve O = F0; O.mse = mseOf(P, O); O.ok = std::isfinite(O.mse) && O.mse < 1e98; O.status = 5;
      if (O.ok && O.mse <= best.mse) { best = O; bestIsOld = true; }
    }
  };
  // start 1: the old estimator (seed)
  if (old.ok && old.R > 0 && std::isfinite(old.rms))
  {
    Curve C; double a = (old.a - mx) / s, b = (old.b - my) / s, R = old.R / s;
    C.th = std::atan2(b, a); C.d = std::hypot(a, b) - R; C.k = 1 / R;
    consider(C, true);
  }
  // start 2: the Kasa circle in scaled coordinates
  {
    double M[3][3] = {{0}}, v[3] = {0}, sol[3];
    for (auto &p : P)
    {
      double q[3] = {p.x, p.y, 1}, z = p.x * p.x + p.y * p.y;
      for (int i = 0; i < 3; ++i) { v[i] += q[i] * z; for (int j = 0; j < 3; ++j) M[i][j] += q[i] * q[j]; }
    }
    if (solve3(M, v, sol))
    {
      double a = sol[0] / 2, b = sol[1] / 2, r2 = sol[2] + a * a + b * b;
      if (r2 > 0) { Curve C; double R = std::sqrt(r2); C.th = std::atan2(b, a); C.d = std::hypot(a, b) - R; C.k = 1 / R; consider(C, false); }
    }
  }
  // start 3: the PCA line — ALWAYS (2026-09-11 check: restricting it to n <= 30 left
  // 16 of 134,661 real 4-layer windows and 96 sim windows short of the multi-start
  // optimum by > 0.1 mm; the extra LM costs ~10 us per fit)
  {
    Curve C; C.th = 0.5 * std::atan2(2 * xy, xx - yy) + M_PI / 2; C.d = 0; C.k = 0;
    consider(C, false);
  }
  if (!best.ok) return old;
  if (bestIsOld) { Fit O = old; O.status = 5; return O; }   // bit-identical to the pre-2026-09-11 result
  F.status = best.status;
  // map back to (a, b, R) in the caller's units
  if (std::fabs(best.k) < 1e-12)
  {
    F.a = 0; F.b = 0; F.R = 1e99; F.isLine = true;    // near-line optimum (see header comment)
  }
  else
  {
    double c = best.d + 1 / best.k;
    F.a = mx + s * c * std::cos(best.th);
    F.b = my + s * c * std::sin(best.th);
    F.R = s / std::fabs(best.k);
  }
  F.rms = s * std::sqrt(std::max(0., best.mse));
  F.ok = true;
  return F;
}
}  // namespace CircleFit

#endif
