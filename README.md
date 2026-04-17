# Beyond Guarantees: When Reliable ML Systems Fail

> *Reliability is not a property of a metric. It is a property of a system under a specific deployment condition. Every check answers one question. Deployment reliability requires asking all of them.*

---

## What this is

The capstone of the Mathematical Reliability for ML series. Not a summary — an adversarial stress test of the entire reliability stack built across the preceding six artifacts.

Each of the six preceding repos provides one necessary reliability check. This artifact provides empirical evidence that none is sufficient alone — and identifies the one check that cannot be bypassed by any of the four demonstrated failure modes.

**Series:**
[Mathematical Reliability](https://github.com/mariam-elelidy/Mathematical-Reliability-for-ML-Predictions) · [Assumption Stress Harness](https://github.com/mariam-elelidy/Assumption-Stress-Harness) · [Influence & Stability](https://github.com/mariam-elelidy/Influence-Stability-Analysis-for-ML-Predictions) · [Calibration](https://github.com/mariam-elelidy/Calibration-as-a-Measurable-Reliability-Constraint) · [Selective Prediction](https://github.com/mariam-elelidy/Selective-Prediction-Under-Uncertainty) · [Shift Detector](https://github.com/mariam-elelidy/Covariate-Shift-Detector)

---

## The four failure modes

| # | Name | Passes all prior checks? | What fails |
|---|---|---|---|
| 1 | **Metrics lie** | Coverage ✓, RMSE (borderline) | Mean predictor R² = −0.000, passes coverage |
| 2 | **Decisions fail** | Coverage ✓, R² ✓ | High-stake coverage 0.870 vs 0.913 low-stake, 1.23× cost inflation |
| 3 | **Coverage is useless** | Coverage ✓, efficiency ✗ | Coverage 0.9975 at width 3.61 — 56% less efficient than α=0.50 |
| 4 | **Monitoring misses** | MMD alarm: 0/7 | Coverage 0.910 → 0.108. Every alarm: ALL CLEAR. |

---

## Core finding: The blind spot

Failure Mode 4 is the result that nothing in the series predicted:

- Test features: same $\mathcal{N}(0, I)$ as training ($P(X)$ unchanged)
- True weights $w^*$: shifted by $\delta \cdot \varepsilon$
- MMD alarm: **0 fires across all drift levels**
- Coverage: 0.910 → 0.108 (-0.803)

The covariate shift detector (Repo 6) is correctly built. It correctly detects covariate shift. This scenario — label shift — is outside its design specification. Feature-space monitoring cannot see $P(Y|X)$. The model continues reporting 90% confidence on intervals that cover 10.8% of outcomes.

---

## Quick start

```bash
pip install numpy scipy

python beyond_guarantees.py              # defaults: n=2000, d=8, α=0.10
python beyond_guarantees.py --n 4000 --n-perm 1000
```

---

## The minimum reliability contract

| Check | Sufficient alone? | What it misses |
|---|---|---|
| Coverage ≥ 1−α | No | Mean predictor passes |
| RMSE | No | Wide model still passes coverage |
| R² > 0 | No | Subgroup gap invisible |
| Coverage efficiency | No | Stake-level heterogeneity |
| Subgroup coverage audit | No | Label shift invisible |
| Feature-space MMD alarm | No | Label shift invisible |
| **Labeled probe monitoring** | **Minimum complete** | — |

---

## What each section demonstrates

**Section 1 — When Metrics Lie:**  
Three models, all passing standard checks. Only one is useful (R²=0.941). Mean predictor R²=−0.000, efficiency=0.106 vs 0.443. Efficiency ratio: 4.19×.

**Section 2 — When Decisions Fail:**  
Global coverage 0.9025 (passes). High-stake coverage 0.8700. Asymmetric cost inflation 1.23× — paying 23% more expected cost than uniform-coverage would deliver.

**Section 3 — When Coverage is Useless:**  
α=0.001 → coverage=0.9975, width=3.61, efficiency=0.276 (56% less efficient than α=0.50). The trivial bound $(-\infty, +\infty)$ achieves coverage=1.000, efficiency=0.

**Section 4 — When Monitoring Misses:**  
7 drift levels. 0 alarms. Coverage: 0.910 → 0.108. All silent.

---

## Repository layout

```
├── README.md              ← this file
├── beyond_guarantees.py   ← implementation
├── output.txt             ← annotated run output
└── writeup.md             ← full technical writeup
```

---

## References

- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*.
- Gretton, A., et al. (2012). A kernel two-sample test. *JMLR*, 13, 723–773.
- Rabanser, S., et al. (2019). Failing loudly: detecting dataset shift. *NeurIPS*.
- Geifman, Y., & El-Yaniv, R. (2017). Selective prediction in deep neural networks. *NeurIPS*.
