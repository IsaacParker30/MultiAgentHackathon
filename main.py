"""
Atomistic Simulation Planner with Local Execution & Validation
===============================================================
Six-stage pipeline from problem description to validated results:

  1. planner        – chats with you to scope the work (intake only)
  2. domain_expert  – recommends DFT settings, MLIP choice, finetuning strategy
  3. coding_agent   – turns the approved plan into runnable scripts
  4. LOCAL EXECUTION – runs scripts that are safe to execute locally
  5. validator      – checks execution output against scientific references
  6. html_reporter  – produces a summary HTML of the full pipeline

Scripts are written to a per-run workspace directory so the main repo
stays clean.
"""

from __future__ import annotations

import datetime
import os
import re
import subprocess
import sys
import textwrap

from autogen import AssistantAgent, UserProxyAgent, LLMConfig
from dotenv import load_dotenv

from validator import Validator

load_dotenv()

# ── Shared LLM config ────────────────────────────────────────────────────────
llm_config = LLMConfig({
    "model": "gpt-5.4-mini",
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "api_type": "openai",
})

# ── Workspace setup ──────────────────────────────────────────────────────────
WORKSPACE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspaces")


def create_workspace() -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(WORKSPACE_ROOT, f"run_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


# ── Executability assessment ─────────────────────────────────────────────────
HEAVY_PACKAGES = {
    "vasp", "quantum_espresso", "qe", "gpaw", "abinit", "siesta",
    "lammps", "gromacs", "namd", "cp2k", "orca", "gaussian",
    "slurm", "pbs", "sge", "torque",
}

SAFE_PACKAGES = {
    "numpy", "scipy", "matplotlib", "math", "os", "sys", "json",
    "csv", "re", "collections", "itertools", "functools",
    "pyscf", "ase", "pymatgen", "mace", "nequip", "torch",
    "sklearn", "pandas",
}


def assess_executability(script: str, filename: str) -> tuple[bool, str]:
    """Decide whether a script is safe and simple enough to run locally.

    Returns (can_run, reason).
    """
    lower = script.lower()

    if filename.endswith((".slurm", ".pbs", ".sub")):
        return False, "job-scheduler script — needs HPC"

    if filename.endswith((".sh", ".bash")):
        hpc_markers = ["sbatch", "srun", "qsub", "bsub", "#SBATCH", "#PBS", "module load"]
        if any(m in script for m in hpc_markers):
            return False, "shell script with HPC scheduler commands"

    if not filename.endswith((".py", ".sh", ".bash")):
        return False, f"not a Python or shell script ({filename})"

    if filename.endswith(".py"):
        if "subprocess.run" in script or "os.system" in script:
            if any(cmd in lower for cmd in ["mpirun", "srun", "sbatch", "qsub"]):
                return False, "launches HPC jobs via subprocess"

        import_lines = [l for l in script.splitlines() if l.strip().startswith(("import ", "from "))]
        for line in import_lines:
            tokens = re.findall(r"(?:from|import)\s+([\w.]+)", line)
            for tok in tokens:
                root = tok.split(".")[0]
                if root in HEAVY_PACKAGES:
                    return False, f"imports heavy package '{root}'"

    return True, "safe to run locally"


# ── Local script executor ────────────────────────────────────────────────────
def execute_script(filepath: str, timeout: int = 120) -> tuple[str, int, str | None]:
    if filepath.endswith((".sh", ".bash")):
        cmd = ["bash", filepath]
    else:
        cmd = [sys.executable, filepath]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(filepath),
        )
        stderr = result.stderr.strip() if result.stderr.strip() else None
        return result.stdout, result.returncode, stderr
    except subprocess.TimeoutExpired:
        return "", 124, f"Script timed out after {timeout}s"


# ── Agent definitions ────────────────────────────────────────────────────────
planner = AssistantAgent(
    name="planner",
    llm_config=llm_config,
    system_message="""
You help users scope atomistic simulations. Ask brief questions to gather:
  - System  (atoms, phase, adsorbates)
  - Goal    (what to compute, target accuracy)
  - Method  (DFT, MLIP out-of-box, MLIP finetune, or "not sure")
  - HPC     (cluster, scheduler, GPU availability)

Ask 1-2 questions at a time. Stay concise.
Do NOT recommend specific functionals, models, or hyperparameters —
a domain expert handles that next.

When you have enough info, write a short summary with these sections:
  System, Goal, Method preference, HPC

Then on a new line, write exactly: Enter 'exit' to proceed with domain expert consultation.
""",
)

domain_expert = AssistantAgent(
    name="domain_expert",
    llm_config=llm_config,
    system_message="""
You are an expert in DFT and machine-learning interatomic potentials.
Given an intake summary, recommend specific technical settings with
one short justification each.

Cover only those that apply:
  - DFT settings: package, exchange-correlation functional, dispersion
    correction, basis set kind, if plane-wave, its cutoff or basis, k-point density, convergence
    thresholds
  - MLIP choice: foundation model (e.g. MACE-MP, SevenNet, ORB),
    out-of-box vs finetune, training set size, validation targets
  - Finetuning (if applicable): naive finetuning or multihead finetuning or frozen layers, learning rate, epochs,
    loss weights
  - Sanity checks: 2-3 specific tests for this system

Use this format:

  RECOMMENDATIONS

  DFT Settings
    - Functional: e.g. PBE+D3 — robust GGA with dispersion for adsorption
    - ...

  MLIP Settings
    - ...

  Validation
    - ...
""",
)

coding_agent = AssistantAgent(
    name="coding_agent",
    llm_config=llm_config,
    system_message="""
You turn an approved plan into runnable scripts. Use the intake AND
the expert's recommendations to produce specific code (use the actual
functional, k-points, model, etc.).
Comment liberally so a novice can follow along.

Generate as many scripts as the task requires (e.g. setup, run, analysis,
submit). For EACH script, output it in this exact format:

FILE: <meaningful_filename.ext>
```<language>
<code>
```

Choose filenames that clearly describe what the script does.

IMPORTANT: When the task can be done with pure Python + standard scientific
libraries (numpy, scipy, matplotlib, pyscf, ase, pymatgen, torch, etc.),
write scripts that are self-contained and can run locally without HPC
infrastructure. Prefer this whenever the computation is small enough
(single molecules, small unit cells, short training runs).
""",
)

html_reporter = AssistantAgent(
    name="html_reporter",
    llm_config=llm_config,
    system_message="""
You produce a single self-contained HTML file summarising a simulation plan
and its results.
Keep it clean and minimal — no external CSS frameworks.

Structure:
  - Title: the system being simulated
  - Section: Problem Statement  (from the intake)
  - Section: System & Goals     (from the intake)
  - Section: Technical Recommendations  (from the domain expert, as a table
    or bullet list)
  - Section: Generated Scripts  (filenames and brief description)
  - Section: Execution Results  (stdout from local runs, if any)
  - Section: Validation Results (PASS/FAIL checks with details, if any)
  - Section: Reference Values Used (show each reference value, its numeric
    range or expected value, and its source tag — e.g. "literature",
    "experimental", "estimated". Always include this section when reference
    values are provided in the input.)

If no execution or validation was performed, omit those sections.

Output ONLY the HTML, nothing else. No markdown fences around it.
""",
)

user = UserProxyAgent(name="user", human_input_mode="ALWAYS", code_execution_config=False)
proxy = UserProxyAgent(name="proxy", human_input_mode="NEVER", code_execution_config=False)


# ── Pipeline ─────────────────────────────────────────────────────────────────

def main():
    workspace = create_workspace()
    print(f"Workspace: {workspace}\n")

    # ── Step 1: intake conversation ──────────────────────────────────────
    problem = input("What would you like to simulate?\n> ").strip()
    chat = user.initiate_chat(planner, message=problem)

    intake = ""
    for m in reversed(chat.chat_history):
        content = m.get("content") or ""
        if any(kw in content.lower() for kw in ["system", "goal", "method"]):
            if len(content) > 80:
                intake = content.strip()
                break

    if not intake:
        print("\nNo intake summary produced. Run again with more detail.")
        raise SystemExit(0)

    # ── Step 2: consult the domain expert ────────────────────────────────
    print("\nConsulting domain expert...")
    expert_result = proxy.initiate_chat(
        domain_expert,
        message=f"Intake summary:\n\n{intake}\n\nProvide your recommendations.",
        max_turns=1,
    )
    recommendations = next(
        (m["content"] for m in reversed(expert_result.chat_history)
         if m.get("name") == "domain_expert"),
        expert_result.chat_history[-1].get("content", ""),
    )

    # ── Step 3: show the plan and ask for approval ───────────────────────
    print("\n" + "=" * 60)
    print("INTAKE SUMMARY")
    print("=" * 60)
    print(intake)
    print("\n" + "=" * 60)
    print(recommendations)
    print("=" * 60)

    plan_path = os.path.join(workspace, "plan.txt")
    with open(plan_path, "w") as f:
        f.write("INTAKE SUMMARY\n==============\n\n")
        f.write(intake + "\n\n")
        f.write(recommendations)
    print(f"\nSaved {plan_path}")

    answer = input("\nGenerate scripts for this plan? (y/n): ").strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit(0)

    # ── Step 4: generate scripts ─────────────────────────────────────────
    print("\nGenerating scripts...")
    result = proxy.initiate_chat(
        coding_agent,
        message=(
            f"Intake:\n\n{intake}\n\n"
            f"Recommendations:\n\n{recommendations}\n\n"
            "Generate all necessary scripts."
        ),
        max_turns=1,
    )
    response = next(
        (m["content"] for m in reversed(result.chat_history)
         if m.get("name") == "coding_agent"),
        result.chat_history[-1].get("content", ""),
    )

    # ── Step 5: extract and save scripts to workspace ────────────────────
    saved_files: list[str] = []
    for match in re.finditer(r"FILE:\s*(\S+)\n```[^\n]*\n(.*?)```", response, re.DOTALL):
        filename = match.group(1).strip()
        code = match.group(2).strip()
        filepath = os.path.join(workspace, filename)
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) != workspace else workspace, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(code + "\n")
        print(f"Saved {filepath}")
        saved_files.append(filename)

    if not saved_files:
        raw_path = os.path.join(workspace, "scripts_raw.txt")
        with open(raw_path, "w") as f:
            f.write(response)
        print(f"Could not extract named scripts. Raw response saved to {raw_path}")
        saved_files = ["scripts_raw.txt"]

    # ── Step 6: assess and execute locally ───────────────────────────────
    execution_results: dict[str, dict] = {}
    executable_scripts: list[tuple[str, str]] = []

    for filename in saved_files:
        filepath = os.path.join(workspace, filename)
        if not os.path.isfile(filepath):
            continue
        with open(filepath) as f:
            code = f.read()
        can_run, reason = assess_executability(code, filename)
        if can_run:
            executable_scripts.append((filename, filepath))
            print(f"  [RUNNABLE] {filename} — {reason}")
        else:
            print(f"  [SKIP]     {filename} — {reason}")

    if executable_scripts:
        run_answer = input(
            f"\n{len(executable_scripts)} script(s) can run locally. Execute them? (y/n): "
        ).strip().lower()

        if run_answer in {"y", "yes"}:
            for filename, filepath in executable_scripts:
                print(f"\n{'─' * 60}")
                print(f"Executing {filename}...")
                print(f"{'─' * 60}")
                stdout, exit_code, stderr = execute_script(filepath, timeout=120)
                execution_results[filename] = {
                    "stdout": stdout,
                    "exit_code": exit_code,
                    "stderr": stderr,
                }
                if stdout.strip():
                    for line in stdout.strip().splitlines():
                        print(f"  │ {line}")
                if exit_code != 0:
                    print(f"  ⚠ Exit code {exit_code}")
                    if stderr:
                        for line in stderr.strip().splitlines()[:10]:
                            print(f"  │ {line}")
                else:
                    print(f"  Exit code 0 (success)")

    # ── Step 7: validate execution results ───────────────────────────────
    validation_results: dict[str, dict] = {}

    if execution_results:
        validate_answer = input(
            "\nValidate execution results against scientific references? (y/n): "
        ).strip().lower()

        if validate_answer in {"y", "yes"}:
            print("\nRunning validation...")
            v = Validator(
                llm_config=llm_config,
                work_dir=workspace,
                execution_timeout=60,
            )

            for filename, exec_data in execution_results.items():
                if exec_data["exit_code"] != 0 and not exec_data["stdout"].strip():
                    print(f"\n  Skipping {filename} — script failed with no output")
                    continue

                print(f"\n{'─' * 60}")
                print(f"Validating {filename}...")
                print(f"{'─' * 60}")

                vresult = v.validate(
                    output=exec_data["stdout"],
                    task_description=f"{intake}\n\nScript: {filename}",
                )

                validation_results[filename] = {
                    "passed": vresult.passed,
                    "summary": vresult.summary,
                    "checks": [
                        {"status": c.status, "detail": c.detail}
                        for c in vresult.checks
                    ],
                    "modules_used": vresult.modules_used,
                    "refs_source": vresult.refs_source,
                    "reference_values": vresult.reference_values,
                    "assessment": vresult.llm_assessment,
                }

                print(f"  Modules: {', '.join(vresult.modules_used)}")
                if vresult.reference_values:
                    print(f"  References ({vresult.refs_source}):")
                    for k, val in vresult.reference_values.items():
                        print(f"    {k}: {val}")
                for c in vresult.checks:
                    print(f"  {c.status}: {c.detail}")
                print(f"  >>> {vresult.summary}")

    # ── Step 8: generate HTML summary ────────────────────────────────────
    print("\nGenerating HTML summary...")

    exec_section = ""
    if execution_results:
        exec_section = "\n\nExecution results:\n"
        for fname, data in execution_results.items():
            exec_section += f"\n--- {fname} (exit code {data['exit_code']}) ---\n"
            exec_section += data["stdout"][:2000] if data["stdout"] else "(no output)"
            if data["stderr"]:
                exec_section += f"\nStderr: {data['stderr'][:500]}"

    val_section = ""
    if validation_results:
        val_section = "\n\nValidation results:\n"
        for fname, vdata in validation_results.items():
            val_section += f"\n--- {fname}: {'PASSED' if vdata['passed'] else 'FAILED'} ---\n"
            val_section += f"Modules: {', '.join(vdata['modules_used'])}\n"
            val_section += f"Reference source: {vdata['refs_source']}\n"
            if vdata.get("reference_values"):
                val_section += "Reference values used:\n"
                for rk, rv in vdata["reference_values"].items():
                    val_section += f"  - {rk}: {rv}\n"
            for c in vdata["checks"]:
                val_section += f"  {c['status']}: {c['detail']}\n"
            val_section += f"Summary: {vdata['summary']}\n"
            if vdata["assessment"]:
                val_section += f"Assessment: {vdata['assessment']}\n"

    html_result = proxy.initiate_chat(
        html_reporter,
        message=(
            f"Intake summary:\n\n{intake}\n\n"
            f"Recommendations:\n\n{recommendations}\n\n"
            f"Generated scripts: {', '.join(saved_files)}"
            f"{exec_section}"
            f"{val_section}"
        ),
        max_turns=1,
    )
    html_content = next(
        (m["content"] for m in reversed(html_result.chat_history)
         if m.get("name") == "html_reporter"),
        html_result.chat_history[-1].get("content", ""),
    )
    html_path = os.path.join(workspace, "summary.html")
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"Saved {html_path}")

    # ── Done ─────────────────────────────────────────────────────────────
    print(f"\nAll outputs saved to: {workspace}")
    print(f"  plan.txt        — intake + recommendations")
    for f in saved_files:
        print(f"  {f:16s} — generated script")
    if execution_results:
        print(f"  (executed {len(execution_results)} script(s) locally)")
    if validation_results:
        passed = sum(1 for v in validation_results.values() if v["passed"])
        total = len(validation_results)
        print(f"  Validation: {passed}/{total} passed")
    print(f"  summary.html    — full report")


if __name__ == "__main__":
    main()
