"""Test the modular verification system end-to-end with dummy data."""

from verification import (
    get_all_modules,
    get_module_catalogue,
    assemble_evaluator_prompt,
    parse_module_selections,
)
from verification.registry import get_modules_by_names
from verification.selector import parse_module_selections


def test_registry():
    print("=" * 60)
    print("  TEST 1: Module Registry")
    print("=" * 60)
    mods = get_all_modules()
    print(f"  Registered modules: {len(mods)}")
    assert len(mods) == 22, f"Expected 22 modules, got {len(mods)}"

    ref_mods = [m for m in mods.values() if m.requires_reference_values]
    print(f"  Modules requiring reference values: {[m.name for m in ref_mods]}")
    assert len(ref_mods) == 2

    catalogue = get_module_catalogue()
    assert "nan_inf_detection" in catalogue
    assert "mlip_energy_force_errors" in catalogue
    print(f"  Catalogue: {len(catalogue)} chars, {catalogue.count(chr(10))+1} lines")
    print("  PASSED\n")


def test_module_selection_parsing():
    print("=" * 60)
    print("  TEST 2: Module Selection Parsing")
    print("=" * 60)

    # Case 1: Structured SELECTED MODULES section
    msg = """For this molecular dynamics task, I will check energy conservation and smoothness.

SELECTED MODULES:
- nan_inf_detection
- conservation_laws
- smoothness
- physical_bounds

Success criteria: total energy drift < 1e-4 eV/atom/ps."""
    result = parse_module_selections(msg)
    print(f"  Structured selection: {result}")
    assert "nan_inf_detection" in result
    assert "conservation_laws" in result
    assert "smoothness" in result
    assert "physical_bounds" in result
    print("  PASSED")

    # Case 2: Inline mention
    msg2 = "I will use modules: loss_convergence, overfitting_detection, metric_bounds for this ML task."
    result2 = parse_module_selections(msg2)
    print(f"  Inline selection: {result2}")
    assert "loss_convergence" in result2
    assert "overfitting_detection" in result2
    print("  PASSED")

    # Case 3: MLIP task
    msg3 = """This MLIP validation needs careful checking.

SELECTED MODULES:
- nan_inf_detection
- mlip_energy_force_errors
- mlip_eos_curve
- mlip_force_consistency
- literature_comparison
- derivative_discontinuities

VALIDATION THRESHOLDS:
- Energy MAE < 2 meV/atom"""
    result3 = parse_module_selections(msg3)
    print(f"  MLIP selection: {result3}")
    assert "mlip_energy_force_errors" in result3
    assert "mlip_eos_curve" in result3
    assert "literature_comparison" in result3
    assert len(result3) == 6
    print("  PASSED")

    # Case 4: Empty / no modules
    msg4 = "The success criteria is accuracy above 90%."
    result4 = parse_module_selections(msg4)
    print(f"  No modules found: {result4}")
    assert result4 == []
    print("  PASSED\n")


def test_reference_value_extraction():
    print("=" * 60)
    print("  TEST 3: Reference Value Extraction")
    print("=" * 60)

    lit_response = """Based on the literature for silicon:

REFERENCE VALUES:
- Si equilibrium lattice constant: 5.43 A (experimental)
- Si bulk modulus: 97.8 GPa (experimental)
- Si cohesive energy: 4.63 eV/atom (experimental)

EXPECTED RANGES:
- Lattice constant should be in [5.3, 5.6] A for DFT methods
- Bulk modulus should be in [80, 110] GPa

VALIDATION THRESHOLDS:
- Energy MAE < 2 meV/atom: acceptable
- Force MAE < 50 meV/A: acceptable"""

    # Simulate _extract_reference_values from workflow.py
    reference_values = {}
    in_ref_section = False
    for line in lit_response.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if any(h in upper for h in ["REFERENCE VALUES", "EXPECTED RANGES", "VALIDATION THRESHOLDS"]):
            in_ref_section = True
            continue
        if in_ref_section and stripped.startswith("- "):
            entry = stripped[2:].strip()
            if ":" in entry:
                key, val = entry.split(":", 1)
                reference_values[key.strip()] = val.strip()
        elif in_ref_section and stripped and not stripped.startswith("-"):
            in_ref_section = False

    print(f"  Extracted {len(reference_values)} reference values:")
    for k, v in reference_values.items():
        print(f"    {k}: {v}")
    assert len(reference_values) >= 5
    assert "Si equilibrium lattice constant" in reference_values
    assert "5.43" in reference_values["Si equilibrium lattice constant"]
    print("  PASSED\n")


def test_prompt_assembly_scientific():
    print("=" * 60)
    print("  TEST 4: Prompt Assembly — Scientific Simulation")
    print("=" * 60)

    modules = ["nan_inf_detection", "derivative_discontinuities", "smoothness",
               "physical_bounds", "literature_comparison"]
    ref_vals = {
        "H2 bond length": "0.74 A (experimental)",
        "H2 dissociation energy": "4.75 eV (experimental)",
    }
    prompt = assemble_evaluator_prompt(modules, ref_vals, refs_source="literature")
    print(f"  Prompt length: {len(prompt)} chars")
    assert "nan_inf_detection" in prompt
    assert "derivative_discontinuities" in prompt
    assert "REFERENCE VALUES" in prompt
    assert "0.74 A" in prompt
    assert "EXECUTION COMPLETE" in prompt
    assert "VALIDATION PASSED" in prompt or "VALIDATION FAILED" in prompt
    print("  Contains: base instructions, 5 module checks, reference values, output format")
    print("  PASSED\n")


def test_prompt_assembly_ml_training():
    print("=" * 60)
    print("  TEST 5: Prompt Assembly — ML Training")
    print("=" * 60)

    modules = ["nan_inf_detection", "loss_convergence", "overfitting_detection",
               "metric_bounds", "learning_rate_schedule"]
    prompt = assemble_evaluator_prompt(modules, reference_values=None)
    print(f"  Prompt length: {len(prompt)} chars")
    assert "loss_convergence" in prompt
    assert "overfitting_detection" in prompt
    assert "REFERENCE VALUES" not in prompt  # no ref values
    assert "EXECUTION COMPLETE" in prompt
    print("  Contains: base instructions, 5 ML checks, no reference values section")
    print("  PASSED\n")


def test_prompt_assembly_mlip():
    print("=" * 60)
    print("  TEST 6: Prompt Assembly — MLIP Training")
    print("=" * 60)

    modules = ["nan_inf_detection", "mlip_energy_force_errors", "mlip_eos_curve",
               "mlip_phonon_stability", "mlip_force_consistency", "literature_comparison"]
    ref_vals = {
        "Si lattice constant": "5.43 A (experimental)",
        "Si bulk modulus": "97.8 GPa (experimental)",
        "MACE-MP-0 energy MAE on MPtrj": "~20 meV/atom (benchmark)",
    }
    prompt = assemble_evaluator_prompt(modules, ref_vals)
    print(f"  Prompt length: {len(prompt)} chars")
    assert "mlip_energy_force_errors" in prompt
    assert "mlip_eos_curve" in prompt
    assert "phonon" in prompt.lower()
    assert "force_consistency" in prompt
    assert "5.43 A" in prompt
    assert "MACE-MP-0" in prompt
    print("  Contains: base instructions, 6 MLIP checks, 3 reference values")
    print("  PASSED\n")


def test_fallback_defaults():
    print("=" * 60)
    print("  TEST 7: Fallback Default Modules")
    print("=" * 60)

    defaults = [
        "nan_inf_detection", "derivative_discontinuities", "non_monotonicity",
        "smoothness", "outlier_detection", "value_range",
    ]
    prompt = assemble_evaluator_prompt(defaults)
    print(f"  Default prompt length: {len(prompt)} chars")
    for mod in defaults:
        assert mod in prompt, f"Missing default module: {mod}"
    print(f"  All 6 default modules present")
    print("  PASSED\n")


def test_priority_ordering():
    print("=" * 60)
    print("  TEST 8: Priority Ordering")
    print("=" * 60)

    names = ["smoothness", "nan_inf_detection", "outlier_detection", "loss_convergence"]
    mods = get_modules_by_names(names)
    priorities = [m.priority for m in mods]
    print(f"  Order: {[m.name for m in mods]}")
    print(f"  Priorities: {priorities}")
    assert priorities == sorted(priorities, reverse=True), "Modules not sorted by priority"
    assert mods[0].name == "nan_inf_detection"  # highest priority (95)
    print("  PASSED\n")


def test_full_pipeline_simulation():
    print("=" * 60)
    print("  TEST 9: Full Pipeline Simulation")
    print("=" * 60)

    # Step 1: Simulate Evaluator selecting modules during planning
    evaluator_response = """For this PySCF bond scan of H2, I will validate the energy curve.

SELECTED MODULES:
- nan_inf_detection
- derivative_discontinuities
- smoothness
- physical_bounds
- mlip_eos_curve
- literature_comparison

Success criteria: smooth PES with minimum near 0.74 A, all energies negative."""

    selected = parse_module_selections(evaluator_response)
    print(f"  Step 1 - Evaluator selected: {selected}")
    assert len(selected) == 6

    # Step 2: Simulate LiteratureReview providing reference values
    lit_response = """REFERENCE VALUES:
- H2 equilibrium bond length: 0.74 A (experimental)
- H2 ground state energy (HF/cc-pVDZ): -1.13 Hartree (computational benchmark)
- H2 dissociation energy: 4.75 eV (experimental)

EXPECTED RANGES:
- Bond length should be in [0.7, 0.8] A for most methods
- Total energy should be in [-1.2, -1.0] Hartree for HF/DZ"""

    reference_values = {}
    in_ref_section = False
    for line in lit_response.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if any(h in upper for h in ["REFERENCE VALUES", "EXPECTED RANGES", "VALIDATION THRESHOLDS"]):
            in_ref_section = True
            continue
        if in_ref_section and stripped.startswith("- "):
            entry = stripped[2:].strip()
            if ":" in entry:
                key, val = entry.split(":", 1)
                reference_values[key.strip()] = val.strip()
        elif in_ref_section and stripped and not stripped.startswith("-"):
            in_ref_section = False

    print(f"  Step 2 - Reference values: {len(reference_values)} entries")

    # Step 3: Assemble execution prompt
    execution_prompt = assemble_evaluator_prompt(selected, reference_values)
    print(f"  Step 3 - Execution prompt: {len(execution_prompt)} chars")

    # Verify the assembled prompt has everything needed
    assert "nan_inf_detection" in execution_prompt
    assert "derivative_discontinuities" in execution_prompt
    assert "physical_bounds" in execution_prompt
    assert "REFERENCE VALUES" in execution_prompt
    assert "0.74 A" in execution_prompt
    assert "EXECUTION COMPLETE" in execution_prompt

    print("  Step 4 - Verification: all modules + references present in prompt")
    print("  PASSED\n")


def test_literature_search_import():
    print("=" * 60)
    print("  TEST 10: Literature Search Module")
    print("=" * 60)
    from literature_search import (
        search_arxiv, search_semantic_scholar,
        ARXIV_TOOL_SCHEMA, SEMANTIC_SCHOLAR_TOOL_SCHEMA,
    )
    assert callable(search_arxiv)
    assert callable(search_semantic_scholar)
    assert ARXIV_TOOL_SCHEMA["function"]["name"] == "search_arxiv"
    assert SEMANTIC_SCHOLAR_TOOL_SCHEMA["function"]["name"] == "search_semantic_scholar"
    print("  Functions and schemas import correctly")
    print("  PASSED\n")


if __name__ == "__main__":
    print()
    test_registry()
    test_module_selection_parsing()
    test_reference_value_extraction()
    test_prompt_assembly_scientific()
    test_prompt_assembly_ml_training()
    test_prompt_assembly_mlip()
    test_fallback_defaults()
    test_priority_ordering()
    test_full_pipeline_simulation()
    test_literature_search_import()

    print("=" * 60)
    print("  ALL 10 TESTS PASSED")
    print("=" * 60)
