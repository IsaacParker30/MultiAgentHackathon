from verification.registry import register

register(
    name="nan_inf_detection",
    description="Check all numerical arrays for NaN/Inf values",
    priority=95,
    applicability_hint="Apply to ALL numerical output. Always relevant regardless of domain.",
    prompt_snippet="""\
NaN / Inf / non-finite values -- check all numerical arrays for non-finite entries.
Any NaN or Inf invalidates the entire result.""",
    code_example="""\
import numpy as np
for name, arr in [("values", values)]:
    arr = np.asarray(arr)
    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))[0]
        print(f"FAIL: {name} has non-finite values at indices {bad.tolist()}")
    else:
        print(f"PASS: {name} all finite")""",
)

register(
    name="derivative_discontinuities",
    description="Detect discontinuities via finite-difference derivative analysis",
    priority=80,
    applicability_hint="Apply when results are a sequence of values across a swept parameter (energies vs bond length, property vs temperature, etc.).",
    prompt_snippet="""\
Derivative discontinuities -- compute finite differences (dy/dx) between consecutive
points. Flag where consecutive derivatives change by more than 3x the median absolute
derivative step. This catches solvers silently converging to a different solution branch.
Example: energies [-75.83, -75.84, -75.85, -75.83, -75.82] have a sign reversal in
differences at index 3 indicating a possible state switch.""",
    code_example="""\
import numpy as np
diffs = np.diff(values)
abs_diffs = np.abs(diffs)
median_step = np.median(abs_diffs[abs_diffs > 0]) if np.any(abs_diffs > 0) else 1e-10
jumps = np.where(abs_diffs > 3 * median_step)[0]
if len(jumps) > 0:
    print(f"FAIL: Derivative discontinuities at indices {jumps.tolist()}")
else:
    print("PASS: No derivative discontinuities detected")""",
)

register(
    name="non_monotonicity",
    description="Detect unexpected reversals in quantities that should vary smoothly",
    priority=70,
    applicability_hint="Apply when a quantity should monotonically increase or decrease (e.g., cumulative sums, sorted outputs, loss over training).",
    prompt_snippet="""\
Non-monotonicity / unexpected reversals -- if a quantity should vary smoothly or
monotonically, check for sign changes in finite differences that break the expected
trend. Distinguish between noise (small reversals) and genuine trend breaks.""",
    code_example="""\
import numpy as np
diffs = np.diff(values)
sign_changes = np.where(np.diff(np.sign(diffs)) != 0)[0]
if len(sign_changes) > len(values) * 0.1:
    print(f"FAIL: {len(sign_changes)} unexpected reversals detected")
else:
    print(f"PASS: Monotonicity acceptable ({len(sign_changes)} minor reversals)")""",
)

register(
    name="smoothness",
    description="Check for kinks or discontinuities via second finite differences",
    priority=65,
    applicability_hint="Apply to any curve or trajectory that should be smooth (potential energy surfaces, interpolated data, continuous simulations).",
    prompt_snippet="""\
Smoothness -- compute second finite differences. Large spikes relative to the median
indicate kinks or discontinuities even when first differences look acceptable. A spike
in second differences means the curve has a sharp bend at that point.
IMPORTANT: Exclude the first 2 and last 2 points from flagging — boundary effects
(steep repulsive walls, edge artifacts) often cause large second differences that are
physically expected, not errors. Only flag interior points.""",
    code_example="""\
import numpy as np
second_diffs = np.diff(values, n=2)
abs_sd = np.abs(second_diffs)
median_sd = np.median(abs_sd) if len(abs_sd) > 0 else 0
threshold = max(5 * median_sd, 1e-10)
# Exclude boundary points (first 2, last 2) — steep edges are expected
interior = abs_sd[2:-2] if len(abs_sd) > 4 else abs_sd
interior_idx = np.where(interior > threshold)[0] + 2 if len(abs_sd) > 4 else np.where(abs_sd > threshold)[0]
if len(interior_idx) > 0:
    print(f"FAIL: Smoothness violation at indices {interior_idx.tolist()}, max spike = {interior[interior_idx - 2].max():.2e}" if len(abs_sd) > 4 else f"FAIL: Smoothness violation at indices {interior_idx.tolist()}")
else:
    print("PASS: Curve is smooth (no interior second-derivative spikes)")""",
)

register(
    name="outlier_detection",
    description="Flag points deviating significantly from local interpolation of neighbors",
    priority=60,
    applicability_hint="Apply to any dataset where individual outliers would indicate errors (measurement data, simulation outputs, predictions).",
    prompt_snippet="""\
Outlier detection -- flag points whose value deviates from the linear interpolation
of their two neighbors by more than 3x the median such deviation across all points.
This catches isolated bad points that might pass other checks.""",
    code_example="""\
import numpy as np
deviations = []
for i in range(1, len(values) - 1):
    interpolated = (values[i-1] + values[i+1]) / 2
    deviations.append(abs(values[i] - interpolated))
deviations = np.array(deviations)
median_dev = np.median(deviations) if len(deviations) > 0 else 0
threshold = max(3 * median_dev, 1e-10)
outliers = np.where(deviations > threshold)[0] + 1
if len(outliers) > 0:
    print(f"FAIL: Outliers at indices {outliers.tolist()}")
else:
    print("PASS: No outliers detected")""",
)

register(
    name="value_range",
    description="Verify results fall within physically/mathematically reasonable bounds",
    priority=55,
    applicability_hint="Apply when the domain imposes known bounds (energies negative for bound systems, probabilities in [0,1], concentrations non-negative, etc.).",
    prompt_snippet="""\
Value range -- verify results fall in a physically or mathematically reasonable range
for the problem. Examples: energies should be negative for bound systems, loss should
be non-negative, accuracy in [0, 1], no unphysical negative concentrations. Use
reference values from literature if available to set tighter bounds.""",
    code_example="""\
import numpy as np
# Adjust bounds based on the specific problem
lower_bound = None  # Set based on domain
upper_bound = None  # Set based on domain
violations = []
if lower_bound is not None:
    below = np.where(np.asarray(values) < lower_bound)[0]
    if len(below) > 0:
        violations.append(f"values below {lower_bound} at indices {below.tolist()}")
if upper_bound is not None:
    above = np.where(np.asarray(values) > upper_bound)[0]
    if len(above) > 0:
        violations.append(f"values above {upper_bound} at indices {above.tolist()}")
if violations:
    print(f"FAIL: Value range -- " + "; ".join(violations))
else:
    print("PASS: All values within expected range")""",
)
