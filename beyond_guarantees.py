r"""
Beyond Guarantees: When Reliable ML Systems Fail
=================================================
Author : Mariam Mohamed Elelidy

This is the capstone of the Mathematical Reliability for ML series.
The six preceding artifacts built a stack of reliability tools:
  1. Split conformal prediction         — coverage guarantee
  2. Assumption stress harness          — assumption sensitivity
  3. Influence & stability analysis     — training data robustness
  4. Calibration decomposition          — probability reliability
  5. Selective prediction               — abstention under uncertainty
  6. Covariate shift detector           — deployment monitoring

This artifact asks the harder question each tool leaves unanswered:

  "What happens when every tool in the stack passes — and the system
   still fails in a way that matters?"

Four failure modes are demonstrated empirically:

  1. WHEN METRICS LIE         — coverage and RMSE are satisfied by a
                                useless model (mean predictor passes checks)
  2. WHEN DECISIONS FAIL      — 90% global coverage hides a subgroup
                                reliability gap that multiplies decision cost
  3. WHEN COVERAGE IS USELESS — coverage → 1.0 as width → ∞; the metric
                                rewards inflation, not information
  4. WHEN MONITORING MISSES   — label shift leaves MMD ≈ 0 while coverage
                                collapses from 0.910 to 0.108 silently

Each section produces an empirical result — a number, not a claim.

Design philosophy
-----------------
Reliability is not a property of a metric. It is a property of a system
under a specific deployment condition. This artifact operationalises that
distinction: for each failure mode, it shows (a) which metric passes,
(b) what actually fails, and (c) the minimum additional check required.

Usage
-----
    python beyond_guarantees.py          # defaults
    python beyond_guarantees.py --n 4000 --n-perm 1000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.stats import spearmanr


# ────────────────────────────────────────────────────────────────────────────
# Shared primitives
# ────────────────────────────────────────────────────────────────────────────

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def conformal_q(abs_res: np.ndarray, alpha: float = 0.10) -> float:
    n = len(abs_res)
    k = min(max(int(np.ceil((n + 1) * (1 - alpha))), 1), n)
    return float(np.sort(abs_res)[k - 1])


def coverage_and_width(
    y: np.ndarray,
    pred: np.ndarray,
    q: float,
) -> tuple[float, float]:
    lo, hi = pred - q, pred + q
    return float(((y >= lo) & (y <= hi)).mean()), 2 * q


def rbf_mmd2(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Unbiased U-statistic MMD²."""
    def K(A, B):
        sq = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-sq / (2 * sigma ** 2))
    Kxx, Kyy, Kxy = K(X, X), K(Y, Y), K(X, Y)
    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0.0); np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum()/(n*(n-1)) + Kyy.sum()/(m*(m-1)) - 2*Kxy.mean())


def mmd_null_threshold(
    X: np.ndarray, Y: np.ndarray,
    sigma: float, n_perm: int = 400, alpha: float = 0.05, seed: int = 99,
) -> float:
    rng = np.random.default_rng(seed)
    pool = np.vstack([X, Y]); n = len(X)
    nulls = [
        rbf_mmd2(pool[p := rng.permutation(len(pool))][:n], pool[p][n:], sigma)
        for _ in range(n_perm)
    ]
    return float(np.percentile(nulls, 100 * (1 - alpha)))


def r_squared(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


# ────────────────────────────────────────────────────────────────────────────
# Data
# ────────────────────────────────────────────────────────────────────────────

def make_dataset(n: int, d: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    X      = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y      = X @ w_true + 0.6 * rng.normal(size=n)
    idx    = rng.permutation(n)
    n_tr   = int(0.6 * n); n_cal = int(0.2 * n)
    splits = {
        "tr":  idx[:n_tr],
        "cal": idx[n_tr : n_tr + n_cal],
        "te":  idx[n_tr + n_cal :],
    }
    return X, y, w_true, splits


# ────────────────────────────────────────────────────────────────────────────
# Section 1 — When Metrics Lie
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    name:        str
    coverage:    float
    width:       float
    rmse:        float
    r2:          float
    efficiency:  float  # coverage / width


def section_metrics_lie(
    X_tr, y_tr, X_cal, y_cal, X_te, y_te,
    alpha: float = 0.10,
) -> list[MetricResult]:
    """Three models: all pass standard checks. Only one is useful.

    Model A — ridge (real)   : narrow intervals, high R², efficient
    Model B — mean predictor : wide intervals, R²≈0, passes coverage check
    Model C — heavy ridge    : nearly constant predictions, wide intervals
    """
    results = []

    # A: useful ridge
    w_A    = ridge_fit(X_tr, y_tr)
    pred_A = X_te @ w_A
    q_A    = conformal_q(np.abs(y_cal - X_cal @ w_A), alpha)
    cov_A, w = coverage_and_width(y_te, pred_A, q_A)
    results.append(MetricResult("Ridge (useful)", cov_A, w,
        float(np.sqrt(np.mean((pred_A - y_te)**2))),
        r_squared(y_te, pred_A), cov_A / w))

    # B: mean predictor
    mean_v = float(y_tr.mean())
    pred_B = np.full(len(y_te), mean_v)
    q_B    = conformal_q(np.abs(y_cal - mean_v), alpha)
    cov_B, w = coverage_and_width(y_te, pred_B, q_B)
    results.append(MetricResult("Mean predictor", cov_B, w,
        float(np.sqrt(np.mean((pred_B - y_te)**2))),
        r_squared(y_te, pred_B), cov_B / w))

    # C: heavy regularisation
    w_C    = ridge_fit(X_tr, y_tr, lam=1e4)
    pred_C = X_te @ w_C
    q_C    = conformal_q(np.abs(y_cal - X_cal @ w_C), alpha)
    cov_C, w = coverage_and_width(y_te, pred_C, q_C)
    results.append(MetricResult("Heavy ridge", cov_C, w,
        float(np.sqrt(np.mean((pred_C - y_te)**2))),
        r_squared(y_te, pred_C), cov_C / w))

    return results


# ────────────────────────────────────────────────────────────────────────────
# Section 2 — When Decisions Fail
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class DecisionAudit:
    global_coverage:    float
    high_stake_cov:     float
    low_stake_cov:      float
    coverage_gap:       float
    cost_actual:        float
    cost_if_uniform:    float
    cost_inflation:     float
    q:                  float
    err_high:           float
    err_low:            float


def section_decisions_fail(
    X_tr, y_tr, X_cal, y_cal, X_te, y_te,
    alpha: float = 0.10,
    cost_high_miss: float = 10.0,
    cost_low_miss:  float = 1.0,
    stake_pct:      float = 75.0,
) -> DecisionAudit:
    """90% global coverage masks subgroup reliability failure.

    High-stake cases = top (100-stake_pct)% of |y| — extreme outcomes.
    These are disproportionately harder to cover, inflating decision cost.
    """
    w      = ridge_fit(X_tr, y_tr)
    q      = conformal_q(np.abs(y_cal - X_cal @ w), alpha)
    pred   = X_te @ w
    lo, hi = pred - q, pred + q
    covered = (y_te >= lo) & (y_te <= hi)

    high   = np.abs(y_te) >= np.percentile(np.abs(y_te), stake_pct)
    low    = ~high
    h_frac = float(high.mean()); l_frac = float(low.mean())

    cov_h = float(covered[high].mean())
    cov_l = float(covered[low].mean())
    gcov  = float(covered.mean())

    cost_actual  = (1 - cov_h) * cost_high_miss * h_frac + (1 - cov_l) * cost_low_miss * l_frac
    cost_uniform = (1 - gcov)  * cost_high_miss * h_frac + (1 - gcov)  * cost_low_miss * l_frac

    return DecisionAudit(
        global_coverage=gcov,
        high_stake_cov=cov_h, low_stake_cov=cov_l,
        coverage_gap=cov_l - cov_h,
        cost_actual=cost_actual,
        cost_if_uniform=cost_uniform,
        cost_inflation=cost_actual / max(cost_uniform, 1e-9),
        q=q,
        err_high=float(np.abs(y_te[high] - pred[high]).mean()),
        err_low =float(np.abs(y_te[low]  - pred[low]).mean()),
    )


# ────────────────────────────────────────────────────────────────────────────
# Section 3 — When Coverage is Useless
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CoveragePoint:
    alpha:       float
    coverage:    float
    width:       float
    efficiency:  float   # coverage / width
    actionable:  bool    # width < clinical threshold


def section_coverage_useless(
    X_tr, y_tr, X_cal, y_cal, X_te, y_te,
    clinical_threshold: float = 3.0,
) -> list[CoveragePoint]:
    """Coverage inflates toward 1.0 as width → ∞.

    The metric 'coverage' rewards inflation: a model that outputs
    [-∞, +∞] achieves 100% coverage. Efficiency = coverage/width is
    the minimum additional constraint that separates useful from trivial.
    """
    w       = ridge_fit(X_tr, y_tr)
    pred_te = X_te @ w
    pred_cal= X_cal @ w
    abs_res = np.abs(y_cal - pred_cal)

    alphas = [0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.001]
    pts = []
    for a in alphas:
        q_   = conformal_q(abs_res, alpha=a)
        cov_ = float(((y_te >= pred_te - q_) & (y_te <= pred_te + q_)).mean())
        w_   = 2 * q_
        pts.append(CoveragePoint(
            alpha=a, coverage=cov_, width=w_,
            efficiency=cov_ / max(w_, 1e-9),
            actionable=w_ < clinical_threshold,
        ))
    return pts


# ────────────────────────────────────────────────────────────────────────────
# Section 4 — When Monitoring Misses Failures
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class MonitorPoint:
    drift:     float
    mmd2:      float
    coverage:  float
    alarm:     bool
    monitor_says: str

    @property
    def gap(self) -> str:
        if not self.alarm and self.coverage < 0.85:
            return "BLIND SPOT"
        return "ok"


def section_monitoring_misses(
    X_tr, y_tr, X_cal, y_cal, X_te, y_te,
    w_true: np.ndarray,
    alpha: float = 0.10,
    n_perm: int = 400,
    drifts: list | None = None,
) -> tuple[list[MonitorPoint], float, float]:
    """Label shift: P(X) unchanged → MMD ≈ 0 but coverage collapses.

    The feature-space monitor (MMD) sees only P(X). If the data-generating
    process changes (w* shifts) while X stays in-distribution, the monitor
    reports no alarm while the conformal guarantee degrades silently.
    """
    if drifts is None:
        drifts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Sigma and threshold calibrated on original data
    dists = np.sqrt(np.sum((X_tr[:100, None] - X_tr[None, :100]) ** 2, axis=-1))
    sigma = float(np.median(dists[dists > 0])) * 0.3
    thresh = mmd_null_threshold(X_tr[:150], X_te[:100], sigma, n_perm, 0.05)

    # q calibrated on original relationship
    w_A = ridge_fit(X_tr, y_tr)
    q   = conformal_q(np.abs(y_cal - X_cal @ w_A), alpha)

    results = []
    for drift in drifts:
        rng = np.random.default_rng(42 + int(drift * 100))
        X_new = rng.normal(size=(len(X_te), X_tr.shape[1]))  # SAME P(X)
        w_dr  = w_true + drift * rng.normal(size=len(w_true))
        y_new = X_new @ w_dr + 0.6 * rng.normal(size=len(X_te))

        mmd2  = rbf_mmd2(X_tr[:150], X_new[:100], sigma)
        alarm = mmd2 > thresh
        pred  = X_new @ w_A
        cov   = float(((y_new >= pred - q) & (y_new <= pred + q)).mean())

        results.append(MonitorPoint(
            drift=drift, mmd2=mmd2, coverage=cov,
            alarm=alarm,
            monitor_says="SHIFT DETECTED" if alarm else "ALL CLEAR",
        ))

    return results, sigma, thresh


# ────────────────────────────────────────────────────────────────────────────
# Terminal report
# ────────────────────────────────────────────────────────────────────────────

def _bar(v: float, lo: float = 0.0, hi: float = 1.0, w: int = 20) -> str:
    x = max(0.0, min(1.0, (v - lo) / max(hi - lo, 1e-9)))
    k = int(round(x * w))
    return "█" * k + "░" * (w - k)


def print_section(title: str) -> None:
    sep = "═" * 78
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)


def print_report(
    s1: list[MetricResult],
    s2: DecisionAudit,
    s3: list[CoveragePoint],
    s4: tuple[list[MonitorPoint], float, float],
    alpha: float,
) -> None:
    target = 1 - alpha

    # ── Section 1 ──────────────────────────────────────────────────────────
    print_section("FAILURE MODE 1 — WHEN METRICS LIE")
    print()
    print("  Three models. All pass standard reliability checks.")
    print("  Which one should you deploy?\n")
    print(f"  {'model':<20}  {'coverage':>9}  {'width':>8}  "
          f"{'RMSE':>8}  {'R²':>8}  {'cov/width':>10}  {'decision?':>10}")
    print("  " + "─" * 76)
    for m in s1:
        passes = "PASS" if m.coverage >= target else "fail"
        useful = "DEPLOY" if m.r2 > 0.5 else "REJECT"
        print(f"  {m.name:<20}  {m.coverage:>9.4f}  {m.width:>8.4f}  "
              f"{m.rmse:>8.4f}  {m.r2:>8.4f}  {m.efficiency:>10.5f}  {useful:>10}")
    print()
    ridge  = s1[0]; mean_m = s1[1]
    print(f"  Mean predictor coverage check: {mean_m.coverage:.4f}  (target {target:.2f})")
    print(f"  Mean predictor R²            : {mean_m.r2:.6f}  (zero information content)")
    print(f"  Coverage efficiency ratio    : {ridge.efficiency/mean_m.efficiency:.2f}× "
          f"(ridge vs mean predictor)")
    print(f"\n  ⚠  Coverage alone cannot distinguish a useful model from a trivial one.")
    print(f"     Minimum additional check: R² > 0 AND coverage efficiency > threshold")

    # ── Section 2 ──────────────────────────────────────────────────────────
    print_section("FAILURE MODE 2 — WHEN DECISIONS FAIL")
    print()
    print("  Global coverage = 90%. High-stake cases disproportionately missed.\n")
    print(f"  Global coverage    : {s2.global_coverage:.4f}  (passes {target:.2f} check)")
    print(f"  High-stake coverage: {s2.high_stake_cov:.4f}  (top 25% |y|)")
    print(f"  Low-stake coverage : {s2.low_stake_cov:.4f}")
    print(f"  Coverage gap       : {s2.coverage_gap:+.4f}  (low − high)")
    print(f"  q                  : {s2.q:.4f}  (same for both subgroups)")
    print()
    print(f"  Asymmetric cost (miss high = 10×, miss low = 1×):")
    print(f"    Actual cost (subgroup gap)    : {s2.cost_actual:.4f}")
    print(f"    Cost if coverage were uniform  : {s2.cost_if_uniform:.4f}")
    print(f"    Cost inflation                 : {s2.cost_inflation:.2f}×")
    print()
    print(f"  ⚠  Global coverage is a population metric. Decisions are individual.")
    print(f"     Minimum additional check: subgroup coverage audit by stake level")

    # ── Section 3 ──────────────────────────────────────────────────────────
    print_section("FAILURE MODE 3 — WHEN COVERAGE IS USELESS")
    print()
    print("  Coverage increases monotonically toward 1.0 as width → ∞.")
    print("  The metric rewards inflation, not information.\n")
    print(f"  {'alpha':>7}  {'coverage':>9}  {'width':>9}  "
          f"{'efficiency':>11}  {'actionable':>11}  bar")
    print("  " + "─" * 68)
    for pt in s3:
        flag = "← trivial" if not pt.actionable else ""
        print(f"  {pt.alpha:>7.3f}  {pt.coverage:>9.4f}  {pt.width:>9.4f}  "
              f"{pt.efficiency:>11.6f}  {'YES' if pt.actionable else 'NO ':>11}  "
              f"{_bar(pt.efficiency, 0.0, 0.65, 14)}  {flag}")
    print()
    min_eff = min(pt.efficiency for pt in s3)
    max_eff = max(pt.efficiency for pt in s3)
    print(f"  Efficiency at α=0.50 (narrowest): {max_eff:.6f}")
    print(f"  Efficiency at α=0.001 (widest)  : {min_eff:.6f}")
    print(f"  Efficiency loss from over-covering: {(1-min_eff/max_eff)*100:.1f}%")
    print(f"\n  ⚠  Coverage → 1.0 is not a reliability achievement.")
    print(f"     Minimum additional check: coverage efficiency = coverage / width")
    print(f"     Clinical threshold: width < 3.0 → actionable (else: informed abstention)")

    # ── Section 4 ──────────────────────────────────────────────────────────
    pts, sigma, thresh = s4
    print_section("FAILURE MODE 4 — WHEN MONITORING MISSES FAILURES")
    print()
    print("  Label shift: P(X) unchanged → MMD ≈ 0. P(Y|X) changes → coverage collapses.")
    print("  The feature-space monitor reports ALL CLEAR while reliability fails silently.\n")
    print(f"  Sigma = {sigma:.4f}  │  MMD null threshold (α=0.05) = {thresh:.6f}\n")
    print(f"  {'drift':>7}  {'MMD²(X)':>12}  {'coverage':>9}  "
          f"{'alarm':>7}  {'monitor':>15}  {'reality':>12}")
    print("  " + "─" * 72)
    for pt in pts:
        reality = "ok" if pt.coverage >= 0.85 else "FAILING"
        flag = "  ← BLIND SPOT" if pt.gap == "BLIND SPOT" else ""
        print(f"  {pt.drift:>7.1f}  {pt.mmd2:>12.6f}  {pt.coverage:>9.4f}  "
              f"{'ALARM' if pt.alarm else 'ok':>7}  "
              f"{pt.monitor_says:>15}  {reality:>12}{flag}")
    print()
    blind_spots = [pt for pt in pts if pt.gap == "BLIND SPOT"]
    if blind_spots:
        print(f"  Blind spots: {len(blind_spots)} drift levels where monitor says")
        print(f"  ALL CLEAR but coverage is below 0.85:")
        for pt in blind_spots:
            print(f"    drift={pt.drift:.1f}  coverage={pt.coverage:.4f}  MMD²={pt.mmd2:.6f}")
    print()
    no_drift = pts[0]
    max_drift = pts[-1]
    print(f"  Coverage at drift=0: {no_drift.coverage:.4f}")
    print(f"  Coverage at drift={max_drift.drift:.1f}: {max_drift.coverage:.4f}")
    print(f"  Total coverage drop: {no_drift.coverage - max_drift.coverage:+.4f}")
    print(f"  MMD alarms fired: {sum(1 for p in pts if p.alarm)}/{len(pts)}")
    print(f"\n  ⚠  Feature-space monitoring cannot detect label shift.")
    print(f"     Minimum additional check: ground-truth label monitoring on a")
    print(f"     held-out labeled probe set deployed alongside the model")


def print_synthesis() -> None:
    sep = "─" * 78
    print(f"\n{'═'*78}")
    print("  SYNTHESIS: THE MINIMUM RELIABILITY CONTRACT")
    print(f"{'═'*78}")
    print()
    checks = [
        ("Coverage >= 1-α",          "necessary", "not sufficient (mean predictor passes)"),
        ("RMSE / accuracy",           "necessary", "not sufficient (wide model passes coverage)"),
        ("R² > 0",                    "necessary", "not sufficient (subgroup failure hidden)"),
        ("Coverage efficiency",       "necessary", "not sufficient (stake-level gaps hidden)"),
        ("Subgroup coverage audit",   "necessary", "not sufficient (label shift invisible)"),
        ("Feature-space MMD alarm",   "necessary", "not sufficient (label shift invisible)"),
        ("Label probe monitoring",    "necessary", "minimum complete deployment check"),
    ]
    print(f"  {'Check':<35}  {'Status':<12}  {'Why not sufficient'}") 
    print("  " + sep)
    for check, status, why in checks:
        print(f"  {check:<35}  {status:<12}  {why}")
    print()
    print("  No single check is sufficient. The reliability contract requires all of them.")
    print("  The preceding six artifacts each provide one check.")
    print("  This artifact provides the evidence that none is sufficient alone.")


def print_tensor_summary(
    s1: list[MetricResult],
    s2: DecisionAudit,
    s3: list[CoveragePoint],
    s4_pts: list[MonitorPoint],
) -> None:
    print(f"\n{'═'*78}")
    print("  FINAL TENSORS")
    print(f"{'═'*78}")
    print()
    print("  Section 1 — [coverage, width, RMSE, R², efficiency]")
    mat1 = np.array([[m.coverage, m.width, m.rmse, m.r2, m.efficiency] for m in s1])
    print(mat1.round(4))

    print()
    print("  Section 3 — [alpha, coverage, width, efficiency, actionable(0/1)]")
    mat3 = np.array([[p.alpha, p.coverage, p.width, p.efficiency, float(p.actionable)]
                     for p in s3])
    print(mat3.round(5))

    print()
    print("  Section 4 — [drift, MMD², coverage, alarm(0/1)]")
    mat4 = np.array([[p.drift, p.mmd2, p.coverage, float(p.alarm)] for p in s4_pts])
    print(mat4.round(5))


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Beyond guarantees: when reliable ML systems fail"
    )
    p.add_argument("--n",      type=int,   default=2000)
    p.add_argument("--d",      type=int,   default=8)
    p.add_argument("--alpha",  type=float, default=0.10)
    p.add_argument("--seed",   type=int,   default=42)
    p.add_argument("--n-perm", type=int,   default=400, dest="n_perm")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Beyond Guarantees: When Reliable ML Systems Fail")
    print(f"n={args.n}  d={args.d}  α={args.alpha}  seed={args.seed}")
    print("─" * 60)

    X, y, w_true, splits = make_dataset(args.n, args.d, args.seed)
    tr, cal, te = splits["tr"], splits["cal"], splits["te"]
    X_tr, y_tr   = X[tr],  y[tr]
    X_cal, y_cal = X[cal], y[cal]
    X_te,  y_te  = X[te],  y[te]

    print("Running Section 1: When Metrics Lie …")
    s1 = section_metrics_lie(X_tr, y_tr, X_cal, y_cal, X_te, y_te, args.alpha)

    print("Running Section 2: When Decisions Fail …")
    s2 = section_decisions_fail(X_tr, y_tr, X_cal, y_cal, X_te, y_te, args.alpha)

    print("Running Section 3: When Coverage is Useless …")
    s3 = section_coverage_useless(X_tr, y_tr, X_cal, y_cal, X_te, y_te)

    print(f"Running Section 4: When Monitoring Misses Failures ({args.n_perm} perms) …")
    s4 = section_monitoring_misses(
        X_tr, y_tr, X_cal, y_cal, X_te, y_te,
        w_true=w_true, alpha=args.alpha, n_perm=args.n_perm,
    )

    print_report(s1, s2, s3, s4, args.alpha)
    print_synthesis()
    print_tensor_summary(s1, s2, s3, s4[0])


if __name__ == "__main__":
    main()
