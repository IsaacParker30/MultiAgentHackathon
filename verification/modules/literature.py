from verification.registry import register

register(
    name="literature_comparison",
    description="Compare computed results against reference values from scientific literature",
    priority=85,
    requires_reference_values=True,
    applicability_hint="Apply whenever reference values from literature or established benchmarks are available for the computed quantities.",
    prompt_snippet="""\
Literature comparison -- for each computed quantity that has a reference value listed in
the REFERENCE VALUES section, compare the computed result to literature.

CRITICAL: Reference values often include METHOD-SPECIFIC expected ranges, e.g.:
  "1.17 eV (experimental); DFT-PBE systematically underestimates to 0.5-0.7 eV"
  "1.79 J/m^2 (experimental); DFT-PBE typically gives 1.3-1.5 J/m^2"
  "computed harmonic frequencies are typically 5-10% higher than experiment"
When such guidance is given, compare the computed value against the METHOD-SPECIFIC
range, NOT the raw experimental number. The method has known systematic biases —
PASS if the value falls within or near the expected range for that method.

Only compare to the raw experimental value when no method-specific guidance is given.
Rules for comparison (against the appropriate reference):
- Within expected range or within 5% of reference: PASS
- 5-15% deviation with no method-specific guidance: WARNING
- Beyond 15% deviation or far outside method-specific range: FAIL""",
    code_example="""\
import numpy as np
# For each quantity, use method-specific range if available, else experimental value.
# Example: if ref says "DFT-PBE gives 0.5-0.7 eV", compare to [0.5, 0.7].
reference_checks = [
    # (name, computed, ref_low, ref_high, unit)
    ("lattice_constant", 5.47, 5.45, 5.48, "A"),   # DFT-PBE expected range
    ("band_gap", 0.61, 0.5, 0.7, "eV"),            # DFT-PBE expected range
]
for name, comp, ref_low, ref_high, unit in reference_checks:
    if ref_low <= comp <= ref_high:
        print(f"PASS: {name} = {comp} {unit}, within expected range [{ref_low}, {ref_high}] {unit}")
    else:
        mid = (ref_low + ref_high) / 2
        deviation = min(abs(comp - ref_low), abs(comp - ref_high)) / mid * 100
        if deviation > 15:
            print(f"FAIL: {name} = {comp} {unit}, outside expected range [{ref_low}, {ref_high}] {unit}")
        else:
            print(f"WARNING: {name} = {comp} {unit}, near edge of expected range [{ref_low}, {ref_high}] {unit}")""",
)
