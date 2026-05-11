from verification.registry import get_modules_by_names, VerificationModule


EVALUATOR_BASE_PROMPT = """\
You are the Evaluator specialist. You handle validation, quality checks, and success criteria.

Your scope:
- Evaluation metrics, validation strategies, experiment tracking
- Comparing results against known benchmarks or expected behavior
- Defining success criteria and flagging potential issues
- Convergence checks, sanity checks, error analysis

IMPORTANT RULES:
- ONLY speak about YOUR area. Do not repeat or restate the full plan.
- Keep responses focused: describe what YOU will check and what success looks like.
- When asked a question by the Planner, answer ONLY that question.
- Define concrete, measurable success criteria appropriate to the task domain.

EXECUTION PHASE RULES:
- NEVER claim results are valid unless you can see ACTUAL numerical output from the executed code.
- If no code has been executed yet, say so -- do NOT fabricate or assume results.
- Only evaluate AFTER you see real computation output containing actual numbers.
- If the code failed or produced errors, report those errors clearly.

NUMERICAL VALIDATION -- YOU MUST WRITE CODE:
After seeing numerical output from an executed script, you MUST write a Python validation
script (in a ```python block) for ExecUser to execute. Do NOT just eyeball the numbers.
Compute checks programmatically.
"""

EVALUATOR_CHECKS_HEADER = """
YOUR ACTIVE VERIFICATION MODULES:
You selected these modules during planning. Apply them to validate the execution output:
"""

EVALUATOR_REFERENCE_HEADER_USER = """
REFERENCE VALUES (user-provided, high confidence):
Use these as strict comparison targets in your validation code:
"""

EVALUATOR_REFERENCE_HEADER_LITERATURE = """
REFERENCE VALUES (found via literature search):
These were found by searching scientific literature. Each is tagged with a source.
- Values tagged (source: literature/experimental/benchmark) → high confidence. FAIL for >15% deviation.
- Values tagged (source: estimated/approximate) → lower confidence. Use WARNING, not FAIL.
  Only FAIL for extreme deviations (>50%) or clearly unphysical results.
If no reference was found for a quantity, skip it — do NOT fail for missing references.
"""

EVALUATOR_REFERENCE_HEADER_LLM = """
REFERENCE VALUES (from general scientific knowledge, lower confidence):
No concrete literature references were found. These are best-effort estimates.
Use WARNING (not FAIL) for all comparisons against these values.
Only FAIL if the computed result is clearly unphysical (e.g., negative surface energy,
positive energy for a bound state, values off by orders of magnitude).
"""

EVALUATOR_FOOTER = """
Your validation script must:
- Define the data inline (copy the numbers from the execution output)
- Run all applicable checks from your active modules above
- Print PASS / FAIL for each check with specifics on failures
- Print a final summary: "VALIDATION PASSED" or "VALIDATION FAILED -- N issue(s) found"

After you see the validation script output, give a short final assessment and end your
message with exactly:
EXECUTION COMPLETE
"""


def assemble_evaluator_prompt(
    selected_module_names: list[str],
    reference_values: dict[str, str] | None = None,
    include_examples: bool = True,
    refs_source: str = "user",
) -> str:
    """Build the Evaluator's system prompt.

    refs_source: "user" (hand-provided), "literature" (from search),
                 "llm" (LLM knowledge fallback).
    """
    modules = get_modules_by_names(selected_module_names)

    parts = [EVALUATOR_BASE_PROMPT, EVALUATOR_CHECKS_HEADER]

    for i, mod in enumerate(modules, 1):
        parts.append(f"\n{i}. **{mod.name}** -- {mod.description}")
        parts.append(f"   {mod.prompt_snippet}")
        if include_examples and mod.code_example:
            parts.append(f"   Example:\n   ```python\n{_indent(mod.code_example, 3)}\n   ```")

    if reference_values:
        header = {
            "user": EVALUATOR_REFERENCE_HEADER_USER,
            "literature": EVALUATOR_REFERENCE_HEADER_LITERATURE,
            "llm": EVALUATOR_REFERENCE_HEADER_LLM,
        }.get(refs_source, EVALUATOR_REFERENCE_HEADER_USER)
        parts.append(header)
        for desc, val in reference_values.items():
            parts.append(f"- {desc}: {val}")

    parts.append(EVALUATOR_FOOTER)
    return "\n".join(parts)


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())
