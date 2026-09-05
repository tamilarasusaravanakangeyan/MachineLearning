# Linear Regression — Deep Dive

This page builds on the [house-price example](README.MD#quick-example-intuition) in the main supervised learning notes. There, the data sat perfectly on a line (`price ≈ $200 per sq ft`). Real data never does — so here we add a bit of scatter and use it to explain every metric you'd actually look at after fitting a line, and how to read them together to judge whether a model's output can be trusted.

---

## Worked example

Five houses, `x` = size in units of 500 sq ft, `y` = price in $100k:

| x (500 sq ft units) | y (price, $100k) |
|---|---|
| 1 | 2.0 |
| 2 | 3.0 |
| 3 | 3.5 |
| 4 | 4.5 |
| 5 | 5.0 |

---

## 1. Fitting the line — where SSE comes from

Linear regression assumes `y = β₀ + β₁x + ε` and picks the `β₀, β₁` that **minimize the sum of squared errors (SSE)** between predictions and actual values:

```
SSE = Σ (yᵢ − ŷᵢ)²
```

For this data, the least-squares fit is:

```
ŷ = 1.35 + 0.75x
```

| x | y (actual) | ŷ (predicted) | residual | residual² |
|---|---|---|---|---|
| 1 | 2.0 | 2.10 | −0.10 | 0.010 |
| 2 | 3.0 | 2.85 | +0.15 | 0.0225 |
| 3 | 3.5 | 3.60 | −0.10 | 0.010 |
| 4 | 4.5 | 4.35 | +0.15 | 0.0225 |
| 5 | 5.0 | 5.10 | −0.10 | 0.010 |

**SSE = Σ residual² = 0.075**

**Layman version:** SSE is "how wrong was the line, added up, after making every error positive by squaring it." It's always ≥ 0, has no upper limit, and its units are whatever `y` is measured in, squared — so on its own it's hard to judge as "good" or "bad" without more context. That's what the next few metrics fix.

> **Why squares, not absolute values?** Squaring penalizes large misses more than small ones, and it makes the minimization solvable in closed form (a derivative you can set to zero) — that's the actual reason "least squares" is the default fitting method.

---

## 2. Everyday error metrics — MSE, RMSE, MAE

SSE is a running total, so it grows just by adding more data points, even if the fit is equally good. These three metrics turn it into something you can actually reason about per prediction.

```
MSE  = SSE / n        = 0.075 / 5   = 0.015              ("average squared error")
RMSE = √MSE            = √0.015     ≈ 0.1225 ($100k)  →  ≈ $12,250
MAE  = Σ|residual| / n = 0.6 / 5     = 0.12  ($100k)   →  = $12,000
```

**Layman versions:**
- **MSE** — average squared error. Rarely reported directly because squared dollars aren't meaningful to a human; mainly used as a stepping stone to RMSE.
- **RMSE** — "on average, how far off is the model, in the same units as the actual price?" Here: **about $12,250 off per house**. Because it's still built from squares, one very bad prediction pulls it up more than several small ones.
- **MAE** — "on average, how far off is the model, treating every miss equally?" Here: **$12,000 off per house**. Doesn't overweight big misses the way RMSE does, so RMSE ≈ MAE (as here) tells you errors are fairly uniform in size, while RMSE ≫ MAE would flag a few large outlier errors.

> **Watch out:** `MSE` shows up with two different denominators depending on context. ML libraries like scikit-learn divide by `n` (a plain descriptive average, used above). Statistics/inference — the standard errors, t-tests, and intervals in the sections below — divide by `n − 2` (an *unbiased estimator* of the true error variance, since fitting the line already "used up" 2 degrees of freedom). Same name, slightly different number — don't be surprised when they don't match exactly.

---

## 3. R² and Adjusted R² — how much of the variance is explained

Two more sums, both relative to the mean `ȳ = 3.6`:

- **SST** (total variation in y, ignoring x): `Σ(yᵢ − ȳ)² = 5.70`
- **SSR** (variation explained by the model): `SST − SSE = 5.625`

```
R² = 1 − SSE/SST = 1 − 0.075/5.70 ≈ 0.987
```

**Layman version:** "Of all the ways prices differ across these houses, how much is explained just by knowing the size?" Here: **98.7%**. The rest (1.3%) is left to whatever else affects price that size alone doesn't capture (condition, location, luck).

R² is bounded `[0, 1]` for this kind of model and is *scale-free* — that's why it's reported far more often than raw SSE (which depends on the units of `y`).

**Adjusted R²** corrects a flaw: plain R² can only go up (or stay flat) as you add more predictors, even useless ones, because more predictors always give the fit more room to bend toward the data.

```
Adjusted R² = 1 − (1 − R²)·(n − 1)/(n − p − 1)     [p = number of predictors]
            = 1 − (1 − 0.9868)·(4/3) ≈ 0.982
```

With only one predictor the penalty is small (0.987 → 0.982). The gap grows the more predictors you add — if adjusted R² stays flat or drops while plain R² rises, the new predictor isn't actually earning its place.

> **Caveat:** Always compare models using adjusted R² (or better, performance on held-out data), never plain R², once you have more than one predictor.

---

## 4. Statistical significance — is the slope real, or noise?

Fit quality (R²) and coefficient reliability are **separate questions**. A model can fit well on a tiny sample and still have a slope estimate that would bounce around wildly on a different sample. Significance testing asks: *could `β₁ = 0.75` have appeared by chance if size had no real effect on price (null hypothesis β₁ = 0)?*

```
MSE      = SSE / (n − 2)              = 0.075 / 3   = 0.025   (n−2: two params estimated)
SE(β₁)   = √(MSE / Sxx)               = √(0.025/10) = 0.05
t        = β₁ / SE(β₁)                = 0.75 / 0.05 = 15
```

With `df = n − 2 = 3`, a t-statistic of 15 gives a p-value well under 0.001 — the slope is statistically significant.

**Layman versions:**
- **Standard error (SE)** — "if I repeated this experiment with a different sample of houses, how much would my estimated slope typically wobble?" Small SE = a stable, trustworthy estimate.
- **t-statistic** — "how many standard errors is the slope away from zero?" Bigger (in absolute value) = stronger evidence the effect is real, not noise. As a rough rule of thumb, `|t| > 2` is usually "significant" for reasonably sized samples.
- **p-value** — "if size truly had *no* effect on price, how likely would I be to see a slope this large just by chance?" Small p (conventionally < 0.05) = unlikely to be chance, so we trust the effect is real.

---

## 5. F-statistic — is the whole model useful?

With one predictor this is a formality (it will always agree with the slope's t-test), but it becomes essential once you have several predictors and want to know if they *collectively* explain anything, before checking each one individually.

```
MSR (mean square from regression) = SSR / p       = 5.625 / 1  = 5.625
MSE (from section 4)                              =             0.025
F                                  = MSR / MSE     = 5.625/0.025 = 225
```

**Layman version:** "Compared to just guessing the average price for every house, does using size at all meaningfully reduce error?" A large F with a small p-value (here, p < 0.001) says yes.

> **Neat check:** for a single predictor, `F = t²` always holds — here `15² = 225`, matching exactly. That's because with one predictor, "is the model useful at all" and "is this one slope non-zero" are the same question asked two ways.

---

## 6. Confidence interval vs. prediction interval

The part most often mixed up:

| | Confidence interval (CI) | Prediction interval (PI) |
|---|---|---|
| **Question** | What's the *average* price of all houses at this size? | What will *one specific new* house at this size sell for? |
| **Uncertainty source** | Only uncertainty in where the fitted line sits | Line uncertainty **+** natural scatter of individual houses around the line |
| **Width** | Narrower | Always wider |

At `x₀ = 3` (mean size, `ŷ = 3.6` → $360k), using `t(0.025, df=3) ≈ 3.182`:

```
CI (mean response):
  SE = √(MSE · (1/n))              = √(0.025 · 0.2)   = 0.0707
  interval = 3.6 ± 3.182·0.0707    = 3.6 ± 0.225       → ($337.5k, $382.5k)

PI (new observation):
  SE = √(MSE · (1 + 1/n))          = √(0.025 · 1.2)    = 0.1732
  interval = 3.6 ± 3.182·0.1732    = 3.6 ± 0.551       → ($305k, $415k)
```

Same center ($360k), very different widths:
- The CI says: "I'm confident the *average* price of 1,500 sq ft houses falls in a $45k band."
- The PI says: "but any *one* house could land in a $110k band" — individual scatter doesn't wash out the way it does for an average.

---

## 7. Residual diagnostics — is linear regression even valid here?

Everything above (SE, t, p-values, CI, PI) is only trustworthy if certain assumptions about the residuals hold. These aren't single numbers so much as checks you run on a residual plot (residuals on the y-axis, fitted values or `x` on the x-axis):

| Assumption | Plain-English check | This dataset |
|---|---|---|
| **Linearity** | Residuals shouldn't show a curve/pattern when plotted against x — a pattern means the *true* relationship isn't a straight line | −0.10, +0.15, −0.10, +0.15, −0.10 — no obvious curve |
| **Homoscedasticity** (constant variance) | Residual size shouldn't grow or shrink as x increases — a "funnel" shape breaks the SE/t/p-value math | Residuals stay in a similar 0.10–0.15 range throughout |
| **Normality** | Residuals should be roughly bell-shaped — needed for t-tests/intervals to be accurate in small samples (checked with a Q-Q plot) | Can't meaningfully assess with only 5 points |
| **Independence** | One residual shouldn't predict the next — matters mainly for time-ordered data (checked with the **Durbin-Watson statistic**) | Houses aren't sequential, so not a concern here |

> **Caution:** With just 5 data points these checks are illustrative only. Real diagnostics need enough data to actually see a pattern — a rough rule of thumb is several dozen observations minimum before residual plots become meaningful.

---

## 8. Multicollinearity (VIF) — only relevant with multiple predictors

Not applicable to this one-predictor example, but critical the moment you add a second predictor (e.g. bedrooms, from the main [Supervised Learning](README.MD) house-price table).

**Layman version:** if two predictors move together (bigger houses tend to have more bedrooms), the model struggles to tell which one is actually driving price. Coefficients and their p-values become unstable and can flip sign or lose significance, even though the model's overall predictions (R², RMSE) still look fine.

```
VIFᵢ = 1 / (1 − Rᵢ²)     [Rᵢ² = variance of predictor i explained by the *other* predictors]
```

**Rule of thumb:** VIF > 5–10 for a predictor signals problematic overlap with the others — consider dropping one of the correlated predictors or combining them.

---

## Summary

Same house-price dataset throughout, so the numbers below are the actual values computed in each section above — not placeholders.

| Concept | Formula | Layman meaning | Real-world example (this dataset) |
|---|---|---|---|
| **SSE** | `Σ(yᵢ − ŷᵢ)²` | Total squared error — what least squares minimizes | `SSE = 0.075`. Always ≥ 0; a worse-fitting line pushes this higher, with no upper limit. |
| **MSE / RMSE** | `SSE/n`, `√MSE` | "On average, how far off is a prediction, in real units?" | RMSE ≈ **$12,250** per house |
| **MAE** | `Σ\|residual\|/n` | Same idea, without over-weighting big misses | **$12,000** per house |
| **R²** | `1 − SSE/SST` | "What % of price differences does size explain?" | **98.7%** explained by size |
| **Adjusted R²** | `1 − (1−R²)(n−1)/(n−p−1)` | R², penalized for predictor count | **98.2%** — barely drops with 1 predictor |
| **Standard error (slope)** | `√(MSE / Sxx)` | "How much would this slope wobble on a different sample?" | `SE(β₁) = 0.05` — a stable estimate |
| **t-statistic** | `β₁ / SE(β₁)` | "How many standard errors from zero is the effect?" | `t = 15` — very far from zero |
| **p-value** | from t, df=n−2 | "Could this be pure chance?" | `p < 0.001` — essentially no |
| **F-statistic** | `MSR / MSE` | "Does the model as a whole beat just guessing the average?" | `F = 225` (= t², since 1 predictor) |
| **Confidence interval** | `ŷ₀ ± t·√(MSE(1/n + (x₀−x̄)²/Sxx))` | Uncertainty in the *mean* prediction | At 1,500 sqft: **($337.5k, $382.5k)** |
| **Prediction interval** | `ŷ₀ ± t·√(MSE(1 + 1/n + (x₀−x̄)²/Sxx))` | Uncertainty in *one new* prediction | At 1,500 sqft: **($305k, $415k)** — wider |
| **Residual diagnostics** | plot-based, not a formula | "Can I even trust the numbers above?" | No obvious curve or funnel in residuals — but n=5 too small to be conclusive |
| **VIF** | `1/(1−Rᵢ²)` | "Are my predictors secretly measuring the same thing?" | N/A here (1 predictor); relevant once size + bedrooms are both used |

---

## How to read a real model's output

Statistical software (R's `lm()`, Python's `statsmodels`) prints something like this — here, populated with this page's numbers:

```
                 coef      std err      t        P>|t|     [0.025    0.975]
const (β₀)      1.3500     0.1658     8.143     0.0040      0.822     1.878
size_x (β₁)     0.7500     0.0500    15.000     0.0007      0.591     0.909

R-squared:            0.987
Adj. R-squared:       0.982
F-statistic:          225.0
Prob (F-statistic):   0.00069
```

**Reading order — don't jump straight to the coefficients:**

1. **Check `Prob (F-statistic)` first.** If this is large (say > 0.05), the model as a whole isn't demonstrated to explain anything — stop here, nothing below is trustworthy yet. Here: `0.00069`, so the model clears this bar.
2. **Check R² / Adjusted R².** How much of the variance does the model actually explain? Here: 98.7% / 98.2% — a strong fit. A significant F-statistic with a *low* R² just means the effect is real but small.
3. **For each row, check `P>|t|`.** This tells you which specific predictors have a real effect. Here both `const` and `size_x` are significant (p = 0.004, p = 0.0007).
4. **Read the `coef` value for practical size and sign**, only for rows that passed step 3. `size_x = 0.75` (in these units) → roughly **$150 per sq ft**, and the positive sign confirms bigger houses cost more, as expected.
5. **Use `[0.025  0.975]`** (the coefficient's 95% CI) to see the plausible *range* of that effect, not just the point estimate — e.g. the true price-per-500-sqft effect plausibly sits between 0.591 and 0.909.
6. **Run residual diagnostics** (section 7) before trusting any of the above on a real dataset — significance tests assume the residual assumptions hold.
7. **Only then use the model to predict**, picking a **confidence interval** if you want the uncertainty in an *average*, or a **prediction interval** if you want the uncertainty in *one new* house.

← Back to [Supervised Learning](README.MD)
