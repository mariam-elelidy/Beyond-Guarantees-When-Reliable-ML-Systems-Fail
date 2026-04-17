# Beyond Guarantees: When Reliable ML Systems Fail

**Author:** Mariam Mohamed Elelidy  
**Topic:** Failure Mode Analysis · Reliability Limits · Deployment Safety

---

## TL;DR

The six preceding artifacts each built one layer of the reliability stack. This artifact asks: can all six checks pass simultaneously while the system fails in a way that matters?

The answer, demonstrated empirically, is yes — four times over. Each failure mode exposes a structural gap that every preceding check leaves unaddressed.

**The finding that nothing in the series could have predicted:** Label shift — $P(X)$ unchanged, $P(Y|X)$ changes — causes coverage to drop from 0.910 to 0.108 while the MMD monitor fires **zero alarms** across seven drift levels. Every automated check reports clear. The failure is total and invisible.

---

## 1. Motivation

A reliable ML system is not one that passes reliability checks. It is one that remains reliable under the conditions of deployment. The preceding series built six checks. This artifact stress-tests the checks themselves.

---

## 2. The Four Failure Modes

### Failure Mode 1 — When Metrics Lie

Three models evaluated on coverage, RMSE, R², and a new metric: coverage efficiency (coverage / width).

| Model | Coverage | Width | R² | Efficiency |
|---|---|---|---|---|
| Ridge (useful) | 0.9025 | 2.039 | **0.941** | **0.443** |
| Mean predictor | 0.8900 | 8.433 | −0.000 | 0.106 |
| Heavy ridge | 0.8900 | 7.656 | 0.181 | 0.116 |

The mean predictor passes coverage (0.890 ≈ 0.900 with a different seed). Its R² = −0.000. It predicts the training mean for every test point. Coverage is satisfied because the conformal quantile on a useless model is very wide — the interval spans most of the y range regardless of prediction.

**Coverage alone cannot distinguish a useful model from a trivial one.** Efficiency ratio: 4.19× (ridge vs mean predictor). This is the minimum metric that is structurally hard to game.

**Minimum additional check:** R² > 0 AND coverage efficiency > threshold.

---

### Failure Mode 2 — When Decisions Fail

Global coverage 0.9025 (passes). High-stake cases = top 25% of |y| — extreme outcomes.

| Subgroup | Coverage |
|---|---|
| Global | 0.9025 ✓ |
| High-stake (top 25% \|y\|) | **0.8700** |
| Low-stake | 0.9133 |

**Asymmetric cost** (miss high-stake = 10×, miss low-stake = 1×):
- Actual cost: 0.3900
- Cost if coverage were uniform: 0.3169
- **Cost inflation: 1.23×**

The model is less reliable precisely on the cases where reliability matters most. This is not random — it is structural. Large |y| values correspond to large true signals. These are hardest to predict; residuals are larger at the extremes; a global q calibrated on the average case is too narrow for them.

**Minimum additional check:** Subgroup coverage audit by stake level.

---

### Failure Mode 3 — When Coverage is Useless

The same model, with α swept from 0.50 to 0.001:

| α | Coverage | Width | Efficiency | Actionable |
|---|---|---|---|---|
| 0.50 | 0.5275 | 0.839 | **0.628** | YES |
| 0.10 | 0.9025 | 2.039 | 0.443 | YES |
| 0.01 | 0.9900 | 3.026 | 0.327 | **NO** |
| 0.001 | 0.9975 | 3.608 | **0.276** | NO |

Coverage → 1.0 monotonically. Efficiency → 0 monotonically. The trivial bound — interval $(-\infty, +\infty)$ — achieves coverage = 1.000, efficiency ≈ 0.

Efficiency loss from α=0.50 to α=0.001: **56.0%**. The model with the highest coverage is the least informative per unit width.

**Minimum additional check:** Coverage efficiency = coverage / width.

---

### Failure Mode 4 — When Monitoring Misses Failures

Label shift scenario: test features $X$ drawn from the **same** $\mathcal{N}(0, I)$ as training. True weights $w^*$ shift by $\delta \cdot \varepsilon$.

| Drift | MMD²(X) | Coverage | Alarm | Status |
|---|---|---|---|---|
| 0.0 | −0.000936 | **0.9100** | no | ok |
| 0.5 | +0.000478 | **0.4450** | **no** | BLIND SPOT |
| 1.0 | −0.000850 | **0.2725** | **no** | BLIND SPOT |
| 2.0 | −0.000509 | **0.1700** | **no** | BLIND SPOT |
| 3.0 | +0.000705 | **0.1075** | **no** | BLIND SPOT |

**MMD alarms fired: 0 / 7. Coverage drop: 0.910 → 0.108 (-0.803). All silent.**

The feature-space monitor is not wrong — $P(X)$ genuinely has not changed. It cannot see $P(Y|X)$. The q calibrated on the original $w^*$ is applied to predictions under drifted $w^*$. Coverage collapses while every automated check reports clear.

**Minimum additional check:** Labeled probe set deployed alongside the model, with continuous coverage monitoring on ground-truth labels.

---

## 3. Synthesis

| Check | Necessary? | Sufficient? | Failure mode it misses |
|---|---|---|---|
| Coverage ≥ 1−α | Yes | No | F1: mean predictor passes |
| RMSE / accuracy | Yes | No | F1: wide model passes coverage |
| R² > 0 | Yes | No | F2: subgroup gap hidden |
| Coverage efficiency | Yes | No | F2: stake-level heterogeneity |
| Subgroup coverage audit | Yes | No | F4: label shift invisible |
| Feature-space MMD alarm | Yes | No | F4: label shift undetectable |
| Labeled probe monitoring | Yes | **Minimum complete** | — |

No single check is sufficient. The reliability contract requires all of them.

---

## 4. What This Implies

**For deployment:** The conformal coverage guarantee is mathematically correct. This artifact does not challenge its validity. It identifies four conditions under which the guarantee is technically satisfied but practically meaningless. Meaningful deployment reliability requires: metric co-requirements (efficiency + R²), stake-stratified auditing, width as a design constraint, and ground-truth monitoring.

**For research:** The gap between "statistical guarantee" and "reliable deployment" is not a gap in theory. The theory is correct. The gap is in the translation from population-level guarantees to individual decisions, from average-case metrics to worst-case subgroup performance, and from static snapshots to dynamic deployment.

---

## 5. Connections to the Series

| Failure mode | Tool that partially addresses it | Residual gap |
|---|---|---|
| Metrics lie | Calibration decomposition | Efficiency metric missing |
| Decisions fail | Selective prediction (subgroup coverage) | Asymmetric stakes not modeled |
| Coverage useless | Selective prediction (actionability) | Global α choice not audited |
| Monitoring misses | Covariate shift detector | Label shift outside MMD scope |

This is not a critique — it is the correct structure of a reliability series. Each artifact solves one well-defined problem. The unsolved problem becomes visible only when all six are in place.

---

## 6. Reproducibility

```bash
pip install numpy scipy

python beyond_guarantees.py              # defaults
python beyond_guarantees.py --n 4000 --n-perm 1000
```

Deterministic given `--seed`. No plotting libraries required.

---

## 7. Takeaways

> **Reliability is not a property of a metric. It is a property of a system under a specific deployment condition. Every check answers one question. Deployment reliability requires asking all of them.**

Four shifts:

1. **A model that passes every check you run is only reliable under the checks you ran.** The mean predictor nearly passes coverage. Label shift achieves MMD = 0 while failing catastrophically. The checks define a boundary; what matters is what lies outside it.

2. **Coverage efficiency is the metric the field has mostly skipped.** Coverage/width directly captures what coverage alone cannot: whether intervals are informative rather than merely wide. It should accompany every coverage report.

3. **Subgroup coverage auditing is structural, not optional.** 90% global coverage is worth 87% on high-stakes cases and 91% on routine ones. This is guaranteed by any model that is harder to fit on extremes — which is every model.

4. **The most dangerous failure mode is the one no existing tool detects.** Label shift leaves MMD ≈ 0, ECE monitoring unperturbed, and RMSE unmeasurable without labels. Coverage drops 80 percentage points while every alarm reports clear. This is not an edge case. It is the normal condition of any model deployed long enough for the world to change.

---

## References

- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*.
- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. *JMLR*, 13, 723–773.
- Rabanser, S., Günnemann, S., & Lipton, Z. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. *NeurIPS*.
- Geifman, Y., & El-Yaniv, R. (2017). Selective prediction in deep neural networks. *NeurIPS*.
- Kull, M., et al. (2019). Beyond temperature scaling. *NeurIPS*.
