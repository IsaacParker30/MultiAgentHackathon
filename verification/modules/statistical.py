from verification.registry import register

register(
    name="statistical_significance",
    description="Validate statistical significance of results and comparisons",
    priority=70,
    applicability_hint="Apply when comparing methods/models statistically, running hypothesis tests, or reporting p-values and confidence intervals.",
    prompt_snippet="""\
Statistical significance:
- Check p-values are in [0, 1]
- Verify confidence intervals are valid (lower < point estimate < upper)
- If comparing methods, ensure sample sizes are adequate
- Flag suspiciously small p-values (< 1e-10) that might indicate a bug
- Check that standard errors decrease with sqrt(n) as expected""",
    code_example="""\
import numpy as np
issues = []
if p_value < 0 or p_value > 1:
    issues.append(f"p-value={p_value} outside [0,1]")
if ci_lower > ci_upper:
    issues.append(f"CI inverted: [{ci_lower}, {ci_upper}]")
if p_value < 1e-10 and sample_size < 100:
    issues.append(f"Suspiciously small p-value={p_value:.2e} with n={sample_size}")
if issues:
    print("FAIL: " + "; ".join(issues))
else:
    print(f"PASS: Statistical results valid (p={p_value:.4f}, CI=[{ci_lower:.4f}, {ci_upper:.4f}])")""",
)

register(
    name="distribution_validation",
    description="Check that data distributions match expected properties",
    priority=60,
    applicability_hint="Apply when generated/predicted data should follow a known distribution (normal residuals, uniform sampling, Poisson counts, etc.).",
    prompt_snippet="""\
Distribution validation:
- Residuals should be approximately normal for linear models (check skewness, kurtosis)
- Check for unexpected multimodality in unimodal distributions
- Verify mean and variance match expectations
- Flag heavy tails (kurtosis > 6 for supposedly normal data)
- Check that empirical CDF matches theoretical CDF""",
    code_example="""\
import numpy as np
from scipy import stats
residuals = np.array(residuals)
skew = stats.skew(residuals)
kurt = stats.kurtosis(residuals)
_, normality_p = stats.normaltest(residuals) if len(residuals) > 20 else (0, 1.0)
if abs(skew) > 2:
    print(f"FAIL: Residuals highly skewed (skewness={skew:.2f})")
elif kurt > 6:
    print(f"FAIL: Residuals have heavy tails (kurtosis={kurt:.2f})")
elif normality_p < 0.01 and len(residuals) > 20:
    print(f"WARNING: Residuals may not be normal (p={normality_p:.4f})")
else:
    print(f"PASS: Residuals look reasonable (skew={skew:.2f}, kurt={kurt:.2f})")""",
)

register(
    name="sample_size_adequacy",
    description="Verify sufficient sample size for reliable statistical conclusions",
    priority=50,
    applicability_hint="Apply when computing statistics, training models on small datasets, or making comparisons between groups.",
    prompt_snippet="""\
Sample size check:
- For means: standard error = std/sqrt(n), need SE small relative to effect
- For proportions: need n*p > 5 and n*(1-p) > 5 for normal approximation
- For ML: check if test set is large enough for reliable metric estimates
- Flag results based on very small samples (n < 30) as having high uncertainty""",
    code_example="""\
import numpy as np
n = len(data)
se = np.std(data) / np.sqrt(n) if n > 0 else float('inf')
relative_se = se / abs(np.mean(data)) if np.mean(data) != 0 else float('inf')
if n < 10:
    print(f"FAIL: Sample too small (n={n}) for reliable statistics")
elif relative_se > 0.2:
    print(f"WARNING: High relative uncertainty ({relative_se*100:.0f}%) with n={n}")
else:
    print(f"PASS: Sample size adequate (n={n}, relative SE={relative_se*100:.1f}%)")""",
)
