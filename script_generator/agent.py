"""AG2 AssistantAgent that turns a structured JSON spec into a runnable simulation script.

Pairs the assistant with an executor `UserProxyAgent` so the registered `save_script`
tool actually fires. `build_script_generator_agent()` returns the (assistant, executor)
pair; drive them with `executor.initiate_chat(assistant, message=...)` or via the higher
level `agent.run(...)` helper used in `run_demo.py`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from autogen import AssistantAgent, LLMConfig, UserProxyAgent, register_function
from dotenv import load_dotenv

from .docs_tools import fetch_url, search_docs
from .templates import TEMPLATES

load_dotenv()

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "generated_scripts"


def save_script(
    code: Annotated[str, "Registered code key, e.g. 'lammps', 'ase', 'pyscf'"],
    job_name: Annotated[str, "Stem of the output filename (no extension)"],
    content: Annotated[str, "Full script body to write verbatim"],
) -> str:
    """Persist `content` to generated_scripts/<job_name><ext>; return the absolute path."""
    code_key = code.lower()
    if code_key not in TEMPLATES:
        raise ValueError(
            f"Unknown code {code!r}. Registered: {sorted(TEMPLATES.keys())}"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extension = TEMPLATES[code_key].file_extension
    out_path = OUTPUT_DIR / f"{job_name}{extension}"
    out_path.write_text(content)
    return str(out_path.resolve())


def _build_system_message() -> str:
    base = (
        "You are the Simulation Script Generator. You receive a structured JSON spec "
        "(in a ```json``` block) describing a computational chemistry / materials "
        "simulation job, and you produce a runnable input script for the requested code.\n\n"
        "Workflow for every request:\n"
        "1. Parse the JSON: extract `code`, `job_name`, `settings`, optional `system` and `notes`.\n"
        "2. If you are unsure about ANY keyword, default value, syntax, or unit for the "
        "target code, consult the official docs BEFORE writing the script:\n"
        "     a. Call `search_docs(code, query)` with a focused query (e.g. "
        "'fix npt syntax' for LAMMPS, 'Langevin dynamics' for ASE).\n"
        "     b. Pick the most relevant result and call `fetch_url(url)`.\n"
        "     c. Use what you read; cite the URL in a single comment at the top of the script.\n"
        "   Do NOT guess when the docs are one tool call away. Prefer 1-3 doc lookups per "
        "request when uncertain; skip them for things you are confident about.\n"
        "3. Generate the script body using the rules for that code (see below) plus anything "
        "you learned from the docs.\n"
        "4. Call `save_script(code, job_name, content)` to persist it.\n"
        "5. After the tool returns, reply with the script in a fenced code block plus a "
        "ONE-line summary: `Saved: <path>`.\n\n"
        "Hard rules:\n"
        "- Before writing any script, ensure you understand the spec and have consulted the docs as needed. "
        "- The script MUST be runnable as-is by the target code; no placeholder TODOs.\n"
        "- Honour every key in `settings` that you understand; Any unknown keys you should look at the docs.\n"
        "- If a required physical input is missing, choose a sensible default and mention it "
        "in a single comment at the top of the script.\n"
        "- Do not invent file paths that the user did not provide.\n"
        "- Never wrap the saved script in markdown fences; the file should be pure code.\n"
        "- `fetch_url` is restricted to the registered docs domains listed below; use "
        "`search_docs` first to find a permitted URL.\n\n"
        "Double check all scripts against the documentation:"
        "=== Per-code rules, docs, and examples ===\n\n"
    )
    sections = []
    for key, tmpl in TEMPLATES.items():
        docs_line = (
            f"Docs: {tmpl.docs_url} (search domain: {tmpl.docs_domain})\n"
            if tmpl.docs_url
            else ""
        )
        sections.append(
            f"--- {tmpl.name} (code key: '{key}', file extension: '{tmpl.file_extension}') ---\n"
            f"{docs_line}"
            f"{tmpl.system_prompt}\n{tmpl.few_shot}"
        )
    return base + "\n".join(sections)


def _llm_config() -> LLMConfig:
    return LLMConfig(
        {
            "model": "gpt-5.4-mini",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "api_type": "openai",
        }
    )


def build_script_generator_agent() -> tuple[AssistantAgent, UserProxyAgent]:
    """Return (assistant, executor). The executor runs the `save_script` tool calls."""
    assistant = AssistantAgent(
        name="script_generator",
        system_message=_build_system_message(),
        llm_config=_llm_config(),
    )
    executor = UserProxyAgent(
        name="script_generator_executor",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda msg: "Saved:" in (msg.get("content") or ""),
        max_consecutive_auto_reply=10,
        default_auto_reply="",
    )
    register_function(
        save_script,
        caller=assistant,
        executor=executor,
        name="save_script",
        description="Persist a generated simulation script to generated_scripts/<job_name><ext>.",
    )
    register_function(
        search_docs,
        caller=assistant,
        executor=executor,
        name="search_docs",
        description=(
            "Search the official documentation of a registered code. "
            "Args: code (e.g. 'lammps'|'ase'|'pyscf'), query (free text). "
            "Returns up to 5 hits as {title, url, snippet}. "
            "Call this when unsure about a keyword, syntax, or default value."
        ),
    )
    register_function(
        fetch_url,
        caller=assistant,
        executor=executor,
        name="fetch_url",
        description=(
            "Download a documentation page and return plain text (truncated to ~8k chars). "
            "URL must be on a registered docs domain; use search_docs first to find one."
        ),
    )
    return assistant, executor
