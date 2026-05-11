"""Test the standalone Validator with pre-computed data across multiple domains.

No PySCF or heavy computation at runtime — all test data is pre-computed,
so the full suite runs in under a minute.

Domains covered: quantum chemistry, solid-state DFT, surface science, ML training.
"""

from dotenv import load_dotenv

load_dotenv()

from validator import Validator

# ═══════════════════════════════════════════════════════════════════════
#  PRE-COMPUTED TEST DATA
# ═══════════════════════════════════════════════════════════════════════

# ── 1. H2 Bond Scan (HF/STO-3G) ──────────────────────────────────────
# Realistic values for a diatomic potential energy curve

H2_SCAN_GOOD = """\
H2 bond dissociation scan (HF/STO-3G, 12 points):
  d = 0.50 A  E = -0.84272831 Hartree
  d = 0.73 A  E = -1.11682743 Hartree
  d = 0.95 A  E = -1.06122456 Hartree
  d = 1.18 A  E = -0.98753218 Hartree
  d = 1.41 A  E = -0.95328147 Hartree
  d = 1.64 A  E = -0.94152836 Hartree
  d = 1.86 A  E = -0.93813254 Hartree
  d = 2.09 A  E = -0.93725641 Hartree
  d = 2.32 A  E = -0.93702185 Hartree
  d = 2.55 A  E = -0.93698432 Hartree
  d = 2.77 A  E = -0.93697215 Hartree
  d = 3.00 A  E = -0.93696874 Hartree
Minimum energy: -1.11682743 Hartree at d = 0.73 A"""

H2_SCAN_BAD = """\
H2 bond dissociation scan (HF/STO-3G, 12 points):
  d = 0.50 A  E = -0.84272831 Hartree
  d = 0.73 A  E = -1.11682743 Hartree
  d = 0.95 A  E = -1.06122456 Hartree
  d = 1.18 A  E = -0.98753218 Hartree
  d = 1.41 A  E = -0.95328147 Hartree
  d = 1.64 A  E = nan Hartree
  d = 1.86 A  E = -0.78813254 Hartree
  d = 2.09 A  E = -0.78725641 Hartree
  d = 2.32 A  E = -0.78702185 Hartree
  d = 2.55 A  E = -0.78698432 Hartree
  d = 2.77 A  E = -0.78697215 Hartree
  d = 3.00 A  E = -0.78696874 Hartree
Minimum energy: -1.11682743 Hartree at d = 0.73 A"""



# ── 2. Silicon Bulk Properties (DFT-PBE) ─────────────────────────────
# PBE is known to overestimate lattice constant by ~1% and underestimate band gap

SI_BULK_GOOD = """\
Silicon bulk calculation (DFT, PBE functional, PAW, 600 eV cutoff, 12x12x12 k-points):
  Equilibrium lattice constant: 5.47 A
  Bulk modulus: 89.2 GPa
  Cohesive energy: 4.55 eV/atom
  Band gap (indirect, Gamma-X): 0.61 eV"""

SI_BULK_BAD = """\
Silicon bulk calculation (DFT, PBE functional, PAW, 200 eV cutoff, 2x2x2 k-points):
  Equilibrium lattice constant: 7.20 A
  Bulk modulus: 25.3 GPa
  Cohesive energy: 1.82 eV/atom
  Band gap (indirect, Gamma-X): 3.82 eV"""



# ── 3. Cu(111) Surface Energy (DFT-PBE) ──────────────────────────────

CU_SURFACE_GOOD = """\
Cu(111) surface energy calculation (PBE, PAW pseudopotentials, 500 eV cutoff):
  Slab: 7 layers, 15 A vacuum
  Bulk energy per atom: -3.7296 eV
  Slab energy (14 atoms): -52.0144 eV
  Surface area per face: 5.583 A^2
  Surface energy: 1.36 J/m^2
  Work function: 4.92 eV"""

CU_SURFACE_BAD = """\
Cu(111) surface energy calculation (PBE, PAW pseudopotentials, 500 eV cutoff):
  Slab: 3 layers, 5 A vacuum
  Bulk energy per atom: -3.7296 eV
  Slab energy (6 atoms): -20.3245 eV
  Surface area per face: 5.583 A^2
  Surface energy: -0.47 J/m^2
  Work function: 12.3 eV"""



# ── 4. Water Molecule Properties (B3LYP) ─────────────────────────────

WATER_GOOD = """\
Water molecule geometry optimization (B3LYP/6-311+G(2d,2p)):
  O-H bond length: 0.962 A
  H-O-H bond angle: 105.1 degrees
  Dipole moment: 2.08 Debye
  Total energy: -76.4589 Hartree
  Vibrational frequencies: 1627, 3810, 3918 cm^-1"""

WATER_BAD = """\
Water molecule geometry optimization (B3LYP/6-311+G(2d,2p)):
  O-H bond length: 1.52 A
  H-O-H bond angle: 175.3 degrees
  Dipole moment: 0.12 Debye
  Total energy: -76.4589 Hartree
  Vibrational frequencies: 892, 1203, 1587 cm^-1"""



# ── 5. ML Training (CIFAR-10) ────────────────────────────────────────

ML_GOOD = """\
Training log (ResNet-18 on CIFAR-10, SGD lr=0.1 with cosine annealing, batch_size=128):
  Epoch  1/50: train_loss=2.302  val_loss=2.298  val_acc=0.102
  Epoch  5/50: train_loss=1.534  val_loss=1.601  val_acc=0.412
  Epoch 10/50: train_loss=0.891  val_loss=1.023  val_acc=0.643
  Epoch 20/50: train_loss=0.412  val_loss=0.623  val_acc=0.791
  Epoch 30/50: train_loss=0.198  val_loss=0.534  val_acc=0.832
  Epoch 40/50: train_loss=0.098  val_loss=0.498  val_acc=0.849
  Epoch 50/50: train_loss=0.067  val_loss=0.489  val_acc=0.851
Final metrics: val_accuracy=0.851, val_loss=0.489"""

ML_BAD = """\
Training log (ResNet-18 on CIFAR-10, SGD lr=10.0, batch_size=128):
  Epoch 1/50: train_loss=2.302  val_loss=2.310  val_acc=0.100
  Epoch 2/50: train_loss=45.231  val_loss=52.109  val_acc=0.100
  Epoch 3/50: train_loss=nan  val_loss=nan  val_acc=0.100
  Epoch 4/50: train_loss=nan  val_loss=nan  val_acc=0.100
  Epoch 5/50: train_loss=nan  val_loss=nan  val_acc=0.100
Final metrics: val_accuracy=0.100, val_loss=nan"""


# ═══════════════════════════════════════════════════════════════════════
#  TEST CASES
# ═══════════════════════════════════════════════════════════════════════

TESTS = [
    # --- Quantum chemistry: H2 bond scan ---
    {
        "label": "H2 bond scan (PASS expected)",
        "output": H2_SCAN_GOOD,
        "task": "H2 bond dissociation curve using Hartree-Fock with STO-3G basis",
        "refs": None,
        "modules": ["nan_inf_detection", "literature_comparison"],
        "expect_pass": True,
    },
    {
        "label": "H2 scan NaN + shift (FAIL expected)",
        "output": H2_SCAN_BAD,
        "task": "H2 bond dissociation curve using Hartree-Fock with STO-3G basis",
        "refs": None,
        "modules": ["nan_inf_detection", "literature_comparison"],
        "expect_pass": False,
    },
    # --- Solid-state: Si bulk properties ---
    {
        "label": "Si bulk DFT-PBE (PASS expected)",
        "output": SI_BULK_GOOD,
        "task": "Silicon bulk properties from DFT with PBE functional",
        "refs": None,
        "modules": ["nan_inf_detection", "literature_comparison", "value_range"],
        "expect_pass": True,
    },
    {
        "label": "Si bulk wildly wrong (FAIL expected)",
        "output": SI_BULK_BAD,
        "task": "Silicon bulk properties from DFT with PBE functional",
        "refs": None,
        "modules": ["nan_inf_detection", "literature_comparison", "value_range"],
        "expect_pass": False,
    },
    # --- Surface science: Cu(111) ---
    {
        "label": "Cu(111) surface DFT-PBE (PASS expected)",
        "output": CU_SURFACE_GOOD,
        "task": "Cu(111) surface energy from DFT slab calculation with PBE functional",
        "refs": None,
        "modules": ["nan_inf_detection", "literature_comparison", "physical_bounds"],
        "expect_pass": True,
    },
    {
        "label": "Cu(111) negative surface energy (FAIL expected)",
        "output": CU_SURFACE_BAD,
        "task": "Cu(111) surface energy from DFT slab calculation with PBE functional",
        "refs": None,
        "modules": ["nan_inf_detection", "literature_comparison", "physical_bounds"],
        "expect_pass": False,
    },
    # --- Molecular: water properties ---
    {
        "label": "Water B3LYP (PASS expected)",
        "output": WATER_GOOD,
        "task": "Water molecule geometry optimization and frequency analysis with B3LYP",
        "refs": None,
        "modules": ["nan_inf_detection", "literature_comparison", "value_range"],
        "expect_pass": True,
    },
    {
        "label": "Water wrong geometry (FAIL expected)",
        "output": WATER_BAD,
        "task": "Water molecule geometry optimization and frequency analysis with B3LYP",
        "refs": None,
        "modules": ["nan_inf_detection", "literature_comparison", "value_range"],
        "expect_pass": False,
    },
    # --- ML training ---
    {
        "label": "CIFAR-10 ResNet-18 training (PASS expected)",
        "output": ML_GOOD,
        "task": "ResNet-18 training on CIFAR-10 image classification",
        "refs": None,
        "modules": ["nan_inf_detection", "loss_convergence",
                     "overfitting_detection", "metric_bounds"],
        "expect_pass": True,
    },
    {
        "label": "ML training exploding loss (FAIL expected)",
        "output": ML_BAD,
        "task": "ResNet-18 training on CIFAR-10 image classification",
        "refs": None,
        "modules": ["nan_inf_detection", "loss_convergence", "metric_bounds"],
        "expect_pass": False,
    },
]


# ═══════════════════════════════════════════════════════════════════════
#  RUNNER
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  Validator Test Suite")
    print("  Pre-computed data | 5 domains | 10 cases")
    print("=" * 70)

    v = Validator(temperature=0.1, execution_timeout=30)

    results = {}
    for t in TESTS:
        label = t["label"]
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")

        result = v.validate(
            output=t["output"],
            task_description=t["task"],
            reference_values=t["refs"],
            modules=t["modules"],
        )

        print(f"  Modules: {', '.join(result.modules_used)}")
        if result.reference_values:
            print(f"  Refs ({result.refs_source}):")
            for k, val in result.reference_values.items():
                print(f"    {k}: {val}")
        print(f"  Script: {len(result.script)} chars | exit code: {result.script_exit_code}")
        if result.script_error:
            print(f"  Script error: {result.script_error[:200]}")
        if result.script_output.strip():
            print(f"  ┌─ Script Output ─")
            for line in result.script_output.strip().splitlines():
                print(f"  │ {line}")
            print(f"  └──────────────────")
        print(f"  Checks: {result.n_passed} PASS, {result.n_failed} FAIL, {result.n_warnings} WARNING")
        print(f"  Summary: {result.summary}")
        print(f"  >>> {'PASSED' if result.passed else 'FAILED'}")

        results[label] = result.passed

    # ── Final summary ──
    print(f"\n\n{'=' * 70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 70}")
    score = 0
    for t in TESTS:
        label = t["label"]
        passed = results[label]
        correct = (t["expect_pass"] and passed) or (not t["expect_pass"] and not passed)
        score += correct
        marker = "✓" if correct else "✗"
        verdict = "PASSED" if passed else "FAILED"
        print(f"  {marker}  {label:48s} → {verdict}")

    print(f"\n  Score: {score}/{len(TESTS)} correctly classified")
    print("=" * 70)
