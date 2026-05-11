"""Smoke test: drive the script generator on hardcoded specs for LAMMPS, ASE, and PySCF.

Costs OpenAI API calls. Either invocation works from the MultiAgentHackathon/ directory:

    python -m script_generator.run_demo
    python script_generator/run_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from script_generator.agent import OUTPUT_DIR, build_script_generator_agent
else:
    from .agent import OUTPUT_DIR, build_script_generator_agent

SPECS = [
    {
        "code": "lammps",
        "job_name": "water_npt_demo",
        "settings": {
            "ensemble": "npt",
            "temperature_K": 300,
            "pressure_bar": 1.0,
            "timestep_fs": 1.0,
            "steps": 10000,
            "pair_style": "tip4p/2005",
            "data_file": "water.data",
        },
        "system": {"description": "512 TIP4P/2005 water molecules in a cubic box"},
    },
    {
        "code": "ase",
        "job_name": "cu_emt_demo",
        "settings": {"calculator": "EMT", "task": "single_point"},
        "system": {"description": "bulk fcc Cu, 4-atom conventional cell, a=3.6"},
    },
    {
        "code": "pyscf",
        "job_name": "h2o_b3lyp_demo",
        "settings": {"xc": "b3lyp", "basis": "def2-svp", "charge": 0, "spin": 0, "calculation": "neb"},
        "system": {"atoms": "O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24"},
    },
]

EXPECTED_MARKERS = {
    "lammps": ("units", "run"),
    "ase": ("from ase",),
    "pyscf": ("from pyscf",),
}


def _build_message(spec: dict) -> str:
    return (
        "Generate the simulation script for the following spec, then call save_script "
        "and reply with the script body.\n\n"
        f"```json\n{json.dumps(spec, indent=2)}\n```"
    )


def main() -> None:
    assistant, executor = build_script_generator_agent()

    failures: list[str] = []
    for spec in SPECS:
        print("\n" + "=" * 70)
        print(f"Spec: {spec['code']} / {spec['job_name']}")
        print("=" * 70)
        executor.initiate_chat(
            assistant,
            message=_build_message(spec),
            max_turns=12,
            clear_history=True,
        )

        ext = {"lammps": ".in", "ase": ".py", "pyscf": ".py"}[spec["code"]]
        out_path = Path(OUTPUT_DIR) / f"{spec['job_name']}{ext}"
        if not out_path.exists():
            failures.append(f"Missing output: {out_path}")
            continue
        body = out_path.read_text()
        if not body.strip():
            failures.append(f"Empty output: {out_path}")
            continue
        for marker in EXPECTED_MARKERS[spec["code"]]:
            if marker not in body:
                failures.append(f"{out_path} missing marker {marker!r}")

    print("\n" + "=" * 70)
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print(f"OK — wrote {len(SPECS)} scripts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
