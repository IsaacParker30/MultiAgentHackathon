from verification.registry import register

register(
    name="mlip_energy_force_errors",
    description="Validate energy/force prediction errors against DFT reference",
    priority=90,
    applicability_hint="Apply when training or evaluating a machine-learning interatomic potential (MACE, NequIP, ANI, SchNet, Allegro, etc.).",
    requires_reference_values=True,
    prompt_snippet="""\
Energy/force error validation for MLIP:
- Energy MAE should typically be < 1-2 meV/atom for well-trained potentials
- Force MAE should typically be < 50-100 meV/A for well-trained potentials
- Compare train vs validation errors: ratio > 2x suggests overfitting
- If literature reference values are provided, compare against published benchmarks
- Check that errors decrease over training epochs""",
    code_example="""\
import numpy as np
energy_errors = np.abs(predicted_energies - reference_energies)
force_errors = np.abs(predicted_forces - reference_forces)
energy_mae = np.mean(energy_errors)
force_mae = np.mean(force_errors)
print(f"Energy MAE: {energy_mae*1000:.2f} meV/atom")
print(f"Force MAE: {force_mae*1000:.2f} meV/A")
if energy_mae > 0.002:
    print("FAIL: Energy MAE exceeds 2 meV/atom threshold")
else:
    print("PASS: Energy MAE within acceptable range")
if force_mae > 0.1:
    print("FAIL: Force MAE exceeds 100 meV/A threshold")
else:
    print("PASS: Force MAE within acceptable range")""",
)

register(
    name="mlip_eos_curve",
    description="Validate equation-of-state curves for physical reasonableness",
    priority=75,
    applicability_hint="Apply when computing E-V (energy-volume) or E-r (energy-distance) curves for materials or molecules.",
    prompt_snippet="""\
EOS / energy curve validation:
- Energy-volume curve should have a single clear minimum
- The minimum should NOT be at the boundary of the scan range
- The curve should be smooth and convex near the minimum
- Bulk modulus (proportional to second derivative at minimum) must be positive
- Compare equilibrium volume/distance and energy to literature if available""",
    code_example="""\
import numpy as np
min_idx = np.argmin(energies)
if min_idx == 0 or min_idx == len(energies) - 1:
    print("FAIL: Energy minimum at scan boundary -- range likely too narrow")
else:
    print(f"PASS: Energy minimum at index {min_idx}")
second_diff = np.diff(energies, n=2)
if min_idx > 0 and min_idx < len(second_diff) + 1:
    if second_diff[min_idx - 1] <= 0:
        print("FAIL: Curve is concave at minimum (negative curvature)")
    else:
        print("PASS: Curve is convex at minimum (positive curvature)")""",
)

register(
    name="mlip_phonon_stability",
    description="Check phonon frequencies for imaginary modes indicating structural instability",
    priority=70,
    applicability_hint="Apply when computing phonon dispersion or vibrational frequencies for crystal structures or molecules.",
    prompt_snippet="""\
Phonon / vibrational stability check:
- No imaginary frequencies for a structure at equilibrium (negative values in
  squared-frequency representation, or imaginary in linear frequency)
- Small imaginary frequencies near Gamma (< ~0.1 THz) may be numerical artifacts
- Large imaginary modes indicate the structure is unstable or the potential is wrong
- Acoustic modes should approach zero at the Gamma point""",
    code_example="""\
import numpy as np
imaginary_threshold = -0.1  # THz, below this is significant
imaginary = frequencies[frequencies < imaginary_threshold]
if len(imaginary) > 0:
    print(f"FAIL: {len(imaginary)} imaginary phonon modes, worst = {np.min(imaginary):.3f} THz")
else:
    print("PASS: No significant imaginary phonon frequencies")""",
)

register(
    name="mlip_force_consistency",
    description="Check energy-force consistency (forces should be negative gradient of energy)",
    priority=65,
    applicability_hint="Apply when both energies and forces are computed -- especially for validating MLIP predictions.",
    prompt_snippet="""\
Force-energy consistency -- forces should equal the negative gradient of energy with
respect to atomic positions. Compute numerical gradient of energy via finite differences
of position and compare to predicted forces. Large deviations indicate the model is
internally inconsistent (e.g., energy and force heads disagreeing).""",
    code_example="""\
import numpy as np
# Numerical force from energy differences
numerical_forces = -np.gradient(energies, positions)
force_error = np.abs(predicted_forces - numerical_forces)
max_error = np.max(force_error)
if max_error > 0.05:  # eV/A
    print(f"FAIL: Force-energy inconsistency, max deviation = {max_error:.4f} eV/A")
else:
    print(f"PASS: Forces consistent with energy gradient (max dev = {max_error:.4f} eV/A)")""",
)
