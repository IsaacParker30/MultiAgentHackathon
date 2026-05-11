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
import os
import re

load_dotenv()

# ── Shared LLM config ────────────────────────────────────────────────────────
llm_config = LLMConfig({
    "model": "gpt-4o-mini",
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "api_type": "openai",
})

# ── Agent 1: planner gathers requirements (does NOT recommend settings) ──────
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

Then on a new line, write exactly: INTAKE_DONE
""",
)

# ── Agent 2: the domain expert recommends specific technical settings ────────
domain_expert = AssistantAgent(
    name="domain_expert",
    llm_config=llm_config,
    system_message="""
You are an expert in DFT and machine-learning interatomic potentials.
Given an intake summary, recommend specific technical settings with
one short justification each.

Cover only those that apply:
  - DFT settings: package, exchange-correlation functional, dispersion
    correction, plane-wave cutoff or basis, k-point density, convergence
    thresholds
  - MLIP choice: foundation model (e.g. MACE-MP, SevenNet, ORB),
    out-of-box vs finetune, training set size, validation targets
  - Finetuning (if applicable): frozen layers, learning rate, epochs,
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

# ── Agent 3: the coding agent ────────────────────────────────────────────────
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
""",
)

# ── Agent 4: the HTML reporter ───────────────────────────────────────────────
html_reporter = AssistantAgent(
    name="html_reporter",
    llm_config=llm_config,
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
user  = UserProxyAgent(name="user",  human_input_mode="ALWAYS", code_execution_config=False)
proxy = UserProxyAgent(name="proxy", human_input_mode="NEVER",  code_execution_config=False)

# ── Step 1: intake conversation ──────────────────────────────────────────────
problem = input("What would you like to simulate?\n> ").strip()
chat = user.initiate_chat(planner, message=problem)

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

# ── Step 3: show the plan and ask for approval ───────────────────────────────
print("\n" + "=" * 60)
print("INTAKE SUMMARY")
print("=" * 60)
print(intake)
print("\n" + "=" * 60)
print(recommendations)
print("=" * 60)

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

# ── Step 5: extract and save every named script ──────────────────────────────
saved_files = []
for match in re.finditer(r"FILE:\s*(\S+)\n```[^\n]*\n(.*?)```", response, re.DOTALL):
    filename = match.group(1).strip()
    code     = match.group(2).strip()
    with open(filename, "w") as f:
        f.write(code + "\n")
    print(f"Saved {filename}")
    saved_files.append(filename)

if not saved_files:
    with open("scripts_raw.txt", "w") as f:
        f.write(response)
    print("Could not extract named scripts. Raw response saved to scripts_raw.txt")
    saved_files = ["scripts_raw.txt"]

# ── Step 6: generate HTML summary ────────────────────────────────────────────
print("\nGenerating HTML summary...")
html_result = proxy.initiate_chat(
    html_reporter,
    message=(
        f"Intake summary:\n\n{intake}\n\n"
        f"Recommendations:\n\n{recommendations}\n\n"
        f"Generated scripts: {', '.join(saved_files)}"
    ),
    max_turns=1,
)
html_content = next(
    (m["content"] for m in reversed(html_result.chat_history)
     if m.get("name") == "html_reporter"),
    html_result.chat_history[-1].get("content", ""),
)
with open("summary.html", "w") as f:
    f.write(html_content)
print("Saved summary.html")