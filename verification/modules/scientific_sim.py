from verification.registry import register

register(
    name="conservation_laws",
    description="Check that conserved quantities remain constant during simulation",
    priority=85,
    applicability_hint="Apply to molecular dynamics, fluid simulations, or any time-evolution simulation where energy/mass/charge/momentum should be conserved.",
    prompt_snippet="""\
Conservation law checks:
- Total energy (kinetic + potential) should be conserved in NVE dynamics (drift < 1e-4 eV/atom/ps)
- Total momentum should be conserved if no external forces
- Mass/particle number should be constant
- Charge neutrality should be maintained
- Small drift is acceptable; large drift indicates integration errors or thermostat issues""",
    code_example="""\
import numpy as np
total_energies = np.array(total_energies)
drift = (total_energies[-1] - total_energies[0]) / len(total_energies)
max_fluctuation = np.max(np.abs(total_energies - total_energies.mean()))
relative_fluct = max_fluctuation / abs(total_energies.mean()) if total_energies.mean() != 0 else max_fluctuation
if relative_fluct > 0.01:
    print(f"FAIL: Energy not conserved, relative fluctuation = {relative_fluct:.2e}")
else:
    print(f"PASS: Energy conserved (relative fluctuation = {relative_fluct:.2e})")""",
)

register(
    name="symmetry_preservation",
    description="Verify that expected symmetries/degeneracies are maintained in results",
    priority=70,
    applicability_hint="Apply when the system has known symmetry (crystal symmetry, rotational symmetry, particle exchange symmetry) that should be reflected in the results.",
    prompt_snippet="""\
Symmetry preservation:
- Equivalent atoms/sites should have identical or degenerate properties
- Rotational symmetry should give identical energies for rotated configurations
- Time-reversal symmetry: forward/backward should give same energy
- Crystal symmetry: equivalent k-points should have same eigenvalues
- Deviations from expected symmetry indicate numerical issues or broken implementation""",
    code_example="""\
import numpy as np
# Check that symmetry-equivalent values are indeed equal within tolerance
tolerance = 1e-6
for group_name, group_values in symmetry_groups.items():
    spread = np.max(group_values) - np.min(group_values)
    if spread > tolerance:
        print(f"FAIL: Symmetry broken in {group_name}, spread = {spread:.2e}")
    else:
        print(f"PASS: Symmetry preserved in {group_name} (spread = {spread:.2e})")""",
)

register(
    name="convergence_with_parameters",
    description="Check that results converge as computational parameters are tightened",
    priority=65,
    applicability_hint="Apply when systematically varying basis set size, grid density, k-point mesh, cutoff energy, or other accuracy parameters.",
    prompt_snippet="""\
Convergence check:
- Results should converge (changes become smaller) as parameters are tightened
- Plot/check the difference between consecutive parameter values
- If the last two parameter settings give results differing by less than desired
  accuracy, the calculation is converged
- Oscillating results suggest instability, not convergence""",
    code_example="""\
import numpy as np
# values[i] = result at parameter_setting[i] (increasing accuracy)
diffs = np.abs(np.diff(values))
if len(diffs) >= 2 and diffs[-1] < diffs[0] * 0.1:
    print(f"PASS: Results converging (final diff={diffs[-1]:.2e}, initial diff={diffs[0]:.2e})")
elif len(diffs) >= 2 and diffs[-1] > diffs[-2]:
    print(f"FAIL: Results diverging or oscillating (diffs={diffs.tolist()})")
else:
    print(f"WARNING: Convergence unclear, may need more data points")""",
)

register(
    name="physical_bounds",
    description="Verify results respect fundamental physical constraints",
    priority=60,
    applicability_hint="Apply to any physical simulation (energies of bound states negative, no negative concentrations, temperatures positive, probabilities in [0,1], etc.).",
    prompt_snippet="""\
Physical bounds:
- Energies of bound states must be negative (relative to dissociation limit)
- Concentrations, densities, temperatures must be non-negative
- Probabilities and populations must be in [0, 1]
- Bond lengths must be positive and physically reasonable (0.5-5 A for most bonds)
- Interatomic distances must be > 0
- Adapt bounds to the specific physical system being simulated""",
    code_example="""\
import numpy as np
issues = []
if np.any(np.array(energies) > 0):
    issues.append("positive energies found for bound system")
if np.any(np.array(distances) < 0.5):
    issues.append("unphysically short distances (< 0.5 A)")
if issues:
    print(f"FAIL: Physical bounds violated -- " + "; ".join(issues))
else:
    print("PASS: All values within physical bounds")""",
)
