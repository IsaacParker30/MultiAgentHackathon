"""Per-code template registry. Add a new simulation code by registering an entry here."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodeTemplate:
    name: str
    file_extension: str   # leading dot included, e.g. ".in", ".py"
    system_prompt: str    # appended to the base agent prompt
    few_shot: str         # one worked example: input JSON -> script
    docs_url: str = ""    # canonical docs landing page (shown to the agent)
    docs_domain: str = "" # bare hostname used as a `site:` filter and fetch allowlist


_LAMMPS = CodeTemplate(
    name="LAMMPS",
    file_extension=".in",
    system_prompt=(
        "LAMMPS rules:\n"
        "- Emit a classic input deck (no shell wrapping).\n"
        "- Default `units metal` and `atom_style atomic` unless settings override.\n"
        "- Order: units -> atom_style -> boundary -> read_data/lattice -> pair_style/coeff -> "
        "fixes/thermo/timestep -> run.\n"
        "- Always end with a `run <steps>` line; never leave the deck without a run.\n"
        "- If settings.ensemble is 'npt'/'nvt', use `fix ... npt`/`fix ... nvt` with the given T/P.\n"
    ),
    few_shot=(
        'Example input:\n'
        '```json\n'
        '{"code":"lammps","job_name":"argon_nvt",'
        '"settings":{"ensemble":"nvt","temperature_K":100,"timestep_fs":2.0,"steps":5000,'
        '"pair_style":"lj/cut 8.5","data_file":"argon.data"}}\n'
        '```\n'
        'Example output script:\n'
        '```\n'
        'units           metal\n'
        'atom_style      atomic\n'
        'boundary        p p p\n'
        'read_data       argon.data\n'
        'pair_style      lj/cut 8.5\n'
        'pair_coeff      * * 0.0104 3.4\n'
        'velocity        all create 100.0 12345 mom yes rot yes dist gaussian\n'
        'fix             1 all nvt temp 100.0 100.0 0.1\n'
        'timestep        0.002\n'
        'thermo          100\n'
        'run             5000\n'
        '```\n'
    ),
    docs_url="https://docs.lammps.org/",
    docs_domain="docs.lammps.org",
)


_ASE = CodeTemplate(
    name="ASE",
    file_extension=".py",
    system_prompt=(
        "ASE rules:\n"
        "- Emit a runnable Python script (no markdown wrappers in the saved file).\n"
        "- Use `from ase.build import ...` / `from ase.io import read, write` as appropriate.\n"
        "- Pick a calculator that matches settings.calculator (e.g. EMT, LennardJones, "
        "GPAW, NWChem). Default to EMT for metal toy systems if unspecified.\n"
        "- For dynamics: use `ase.md` (VelocityVerlet/Langevin) or `ase.optimize` (BFGS).\n"
        "- Print/log final energy; write trajectory to <job_name>.traj when sensible.\n"
    ),
    few_shot=(
        'Example input:\n'
        '```json\n'
        '{"code":"ase","job_name":"cu_bulk_emt",'
        '"settings":{"calculator":"EMT","task":"single_point"},'
        '"system":{"description":"bulk fcc Cu, 4-atom conventional cell"}}\n'
        '```\n'
        'Example output script:\n'
        '```python\n'
        'from ase.build import bulk\n'
        'from ase.calculators.emt import EMT\n'
        '\n'
        'atoms = bulk("Cu", "fcc", a=3.6, cubic=True)\n'
        'atoms.calc = EMT()\n'
        'energy = atoms.get_potential_energy()\n'
        'print(f"Total energy: {energy:.6f} eV")\n'
        '```\n'
    ),
    docs_url="https://wiki.fysik.dtu.dk/ase/",
    docs_domain="wiki.fysik.dtu.dk",
)


_PYSCF = CodeTemplate(
    name="PySCF",
    file_extension=".py",
    system_prompt=(
        "PySCF rules:\n"
        "- Emit a runnable Python script (no markdown wrappers in the saved file).\n"
        "- Build the molecule with `gto.M(atom=..., basis=..., charge=..., spin=...)`.\n"
        "- For DFT, use `dft.RKS(mol)` (or `UKS` if open-shell); set `mf.xc` from "
        "settings.xc (default 'b3lyp' if unspecified).\n"
        "- For HF, use `scf.RHF`/`scf.UHF`.\n"
        "- Always call `mf.kernel()` and print the resulting energy.\n"
        "- Honour settings.basis (default 'def2-svp'), settings.charge (default 0), "
        "settings.spin (default 0).\n"
    ),
    few_shot=(
        'Example input:\n'
        '```json\n'
        '{"code":"pyscf","job_name":"h2o_b3lyp",'
        '"settings":{"xc":"b3lyp","basis":"def2-svp","charge":0,"spin":0},'
        '"system":{"atoms":"O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24"}}\n'
        '```\n'
        'Example output script:\n'
        '```python\n'
        'from pyscf import gto, dft\n'
        '\n'
        'mol = gto.M(\n'
        '    atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",\n'
        '    basis="def2-svp",\n'
        '    charge=0,\n'
        '    spin=0,\n'
        ')\n'
        'mf = dft.RKS(mol)\n'
        'mf.xc = "b3lyp"\n'
        'energy = mf.kernel()\n'
        'print(f"DFT energy: {energy:.8f} Ha")\n'
        '```\n'
    ),
    docs_url="https://pyscf.org/user.html",
    docs_domain="pyscf.org",
)


TEMPLATES: dict[str, CodeTemplate] = {
    "lammps": _LAMMPS,
    "ase": _ASE,
    "pyscf": _PYSCF,
}


def register_template(code: str, template: CodeTemplate) -> None:
    """Register a new code at runtime (useful for tests and downstream extensions)."""
    TEMPLATES[code.lower()] = template
