"""
Simple Atomistic Simulation Planner
====================================
Four agents take you from a problem description to runnable scripts:

  planner        - chats with you to scope the work (intake only)
  domain_expert  - recommends DFT settings, MLIP choice, finetuning strategy
  coding_agent   - turns the approved plan into as many scripts as needed
  html_reporter  - produces a summary HTML of the problem and recommendations
"""

from autogen import AssistantAgent, UserProxyAgent, LLMConfig
from dotenv import load_dotenv
from pathlib import Path
import os
import re
import time
import json

load_dotenv()

# ── Timing instrumentation ───────────────────────────────────────────────────
timings = {}

# ── Shared LLM config ────────────────────────────────────────────────────────
gpt_5_4 = LLMConfig({
    "model": "gpt-5.4",
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "api_type": "openai",
})

gpt_5_config = LLMConfig({
    "model": "gpt-5.5",
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "api_type": "openai",
})

# ── Agent 1: planner gathers requirements (does NOT recommend settings) ──────
planner = AssistantAgent(
    name="planner",
    llm_config=gpt_5_4,
    system_message="""
You help users scope atomistic simulations. Ask brief questions to gather:
  - System  (atoms, phase, adsorbates)
  - Goal    (what to compute, target accuracy)
  - Method  (DFT (confirm which DFT package), MLIP out-of-box, MLIP finetune, or "not sure")
  - HPC     (cluster, scheduler, GPU availability)

Ask 1-2 questions at a time. Stay concise.
Do NOT recommend specific functionals, models, or hyperparameters —
a domain expert handles that next.

When you have enough info, write a short summary with these sections:
  System, Goal, Method preference, HPC

Then on a new line, write exactly: INTAKE_DONE
""",
)

# ── Agent 2: the domain expert recommends specific technical settings ────────
domain_expert = AssistantAgent(
    name="domain_expert",
    llm_config=gpt_5_config,
    system_message="""
You are an expert in DFT and machine-learning interatomic potentials.
Given an intake summary, recommend specific technical settings with
one short justification each.

Cover only those that apply to an atomistic simulation expert:
  - DFT settings: package, exchange-correlation functional, dispersion
    correction, basis set kind, if plane-wave, its cutoff or basis, k-point density, convergence
    thresholds
  - MLIP choice: foundation model (e.g. MACE-MP, SevenNet, ORB),
    out-of-box vs finetune, training set generation protocol (e.g. rattling, sampling from MD of a foundation model, active learning, etc.)
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

  Calculation Methodology 
    - ...
""",
)

# ── Agent 3: the coding agent ────────────────────────────────────────────────
coding_agent = AssistantAgent(
    name="coding_agent",
    llm_config=gpt_5_config,
    system_message="""
You turn an approved plan into runnable scripts. Use the intake AND
the expert's recommendations to produce specific code (use the actual
functional, k-points, model, etc.). Watch out for any specific mention of packages. 
When applicable, use ASE for DFT setup and MLIP training, but if the expert recommends a specific package, use that instead.
Keep custom functions to a minimum; rely on well-known libraries and packages. 
Comment liberally so a novice can follow along.

Generate as many scripts as the task requires in a numbered fashion, compartmentalizing in dedicated directories when
applicable (e.g. structure generation, DFT labeling, finetuning)
submit). For EACH script, output it in this exact format:

FILE: <meaningful_filename.ext>
```<language>
<code>
```

Choose filenames that clearly describe what the script does.
Name files with a numerical prefix to indicate order, e.g.: 01_structure_generation.py, 02_dft_labeling.py, 03_mlip_finetuning.py, etc.
Make a file called dependencies.txt if there are non-standard dependencies to install, and list installation commands for pip or conda.
Make a final script that is able to run end-to-end, assuming the user has the necessary software installed and data prepared as per your instructions.
""",
)

# ── Agent 4: the HTML reporter ───────────────────────────────────────────────
html_reporter = AssistantAgent(
    name="html_reporter",
    llm_config=gpt_5_4,
    system_message="""
You produce a single self-contained HTML file summarising a simulation plan.
Keep it clean and minimal — no external CSS frameworks.

Structure:
  - Title: the system being simulated
  - Section: Problem Statement  (from the intake)
  - Section: System & Goals     (from the intake)
  - Section: Technical Recommendations  (from the domain expert, as a table
    or bullet list)
  - Section: Generated Scripts  (just the filenames, as a list)

Output ONLY the HTML, nothing else. No markdown fences around it.
""",
)

# ── User (you) and a silent proxy for internal calls ─────────────────────────
user  = UserProxyAgent(
    name="user",
    human_input_mode="ALWAYS",
    code_execution_config=False,
    is_termination_msg=lambda m: "INTAKE_DONE" in (m.get("content") or ""),
)
proxy = UserProxyAgent(name="proxy", human_input_mode="NEVER",  code_execution_config=False)

# ── Step 1: intake conversation ──────────────────────────────────────────────
problem = input("What would you like to simulate?\n> ").strip()
_start = time.time()
chat = user.initiate_chat(planner, message=problem)
timings['intake'] = time.time() - _start

# Grab the planner's final message (contains the intake summary)
intake = ""
for m in reversed(chat.chat_history):
    content = m.get("content") or ""
    if "INTAKE_DONE" in content:
        intake = content.replace("INTAKE_DONE", "").strip()
        break

if not intake:
    print("\nNo intake summary produced. Run again with more detail.")
    raise SystemExit(0)

# ── Step 2: consult the domain expert ────────────────────────────────────────
print("\nConsulting domain expert...")
_start = time.time()
expert_result = proxy.initiate_chat(
    domain_expert,
    message=f"Intake summary:\n\n{intake}\n\nProvide your recommendations.",
    max_turns=1,
)
timings['domain_expert'] = time.time() - _start
recommendations = next(
    (m["content"] for m in reversed(expert_result.chat_history)
     if m.get("name") == "domain_expert"),
    expert_result.chat_history[-1].get("content", ""),
)

# ── Step 3: show the plan and ask for approval ───────────────────────────────
print("\n" + "=" * 60)
print("INTAKE SUMMARY")
print("=" * 60)
print(intake)
print("\n" + "=" * 60)
print(recommendations)
print("=" * 60)

Path("plan.txt").parent.mkdir(parents=True, exist_ok=True)
with open("plan.txt", "w") as f:
    f.write("INTAKE SUMMARY\n==============\n\n")
    f.write(intake + "\n\n")
    f.write(recommendations)
print("\nSaved plan.txt")

answer = input("\nGenerate scripts for this plan? (y/n): ").strip().lower()
if answer not in {"y", "yes"}:
    raise SystemExit(0)

# ── Step 4: generate scripts ─────────────────────────────────────────────────
print("\nGenerating scripts...")
_start = time.time()
result = proxy.initiate_chat(
    coding_agent,
    message=(
        f"Intake:\n\n{intake}\n\n"
        f"Recommendations:\n\n{recommendations}\n\n"
        "Generate all necessary scripts."
    ),
    max_turns=1,
)
timings['coding_agent'] = time.time() - _start
response = next(
    (m["content"] for m in reversed(result.chat_history)
     if m.get("name") == "coding_agent"),
    result.chat_history[-1].get("content", ""),
)

# ── Step 5: extract and save every named script ──────────────────────────────
saved_files = []
for match in re.finditer(r"FILE:\s*(\S+)\n```[^\n]*\n(.*?)```", response, re.DOTALL):
    filename = match.group(1).strip()
    code     = match.group(2).strip()
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        f.write(code + "\n")
    print(f"Saved {filename}")
    saved_files.append(filename)

if not saved_files:
    Path("scripts_raw.txt").parent.mkdir(parents=True, exist_ok=True)
    with open("scripts_raw.txt", "w") as f:
        f.write(response)
    print("Could not extract named scripts. Raw response saved to scripts_raw.txt")
    saved_files = ["scripts_raw.txt"]

# ── Step 6: generate HTML summary ────────────────────────────────────────────
print("\nGenerating HTML summary...")
_start = time.time()
html_result = proxy.initiate_chat(
    html_reporter,
    message=(
        f"Intake summary:\n\n{intake}\n\n"
        f"Recommendations:\n\n{recommendations}\n\n"
        f"Generated scripts: {', '.join(saved_files)}"
    ),
    max_turns=1,
)
timings['html_reporter'] = time.time() - _start
html_content = next(
    (m["content"] for m in reversed(html_result.chat_history)
     if m.get("name") == "html_reporter"),
    html_result.chat_history[-1].get("content", ""),
)
Path("summary.html").parent.mkdir(parents=True, exist_ok=True)
with open("summary.html", "w") as f:
    f.write(html_content)
print("Saved summary.html")

# ── Timing summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TIMING SUMMARY")
print("=" * 60)
total_time = 0.0
for step, elapsed in timings.items():
    print(f"{step:20s} {elapsed:8.2f}s")
    total_time += elapsed
print("-" * 60)
print(f"{'TOTAL':20s} {total_time:8.2f}s")
print("=" * 60)

# Write timings to JSON log
Path("timing_log.json").parent.mkdir(parents=True, exist_ok=True)
with open("timing_log.json", "w") as f:
    json.dump({"timestamp": time.time(), "timings": timings, "total": total_time}, f, indent=2)
print("Timing log saved to timing_log.json")