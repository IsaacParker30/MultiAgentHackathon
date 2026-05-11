"""Standalone validation API for computational and experimental results.

Usage:
    from validator import Validator

    v = Validator(model="gpt-5.4-mini", api_key="sk-...")
    result = v.validate(
        output="d = 0.50 A  E = -1.043 Hartree\\n...",
        task_description="H2 bond scan with PySCF",
    )
    print(result.passed)   # True / False
    print(result.summary)  # "Validation PASSED: 5 of 5 checks passed."
    print(result.checks)   # [CheckResult(...), ...]

When no reference_values are provided, the Validator automatically:
1. Searches scientific literature (arXiv, Semantic Scholar) for reference values
2. Falls back to LLM knowledge if no literature found
Literature-sourced refs can trigger FAIL; LLM-knowledge refs only trigger WARNING.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Literal

from autogen import ConversableAgent, LLMConfig

from verification import (
    assemble_evaluator_prompt,
    get_module_catalogue,
    parse_module_selections,
)
from verification.registry import get_all_modules

_CHECK_RE = re.compile(r"^\s*(PASS|FAIL|WARNING)\s*:\s*(.+)", re.IGNORECASE)
_SUMMARY_RE = re.compile(r"VALIDATION\s+(PASSED|FAILED)", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)

DEFAULT_FALLBACK_MODULES = [
    "nan_inf_detection",
    "derivative_discontinuities",
    "non_monotonicity",
    "smoothness",
    "outlier_detection",
    "value_range",
]


@dataclass
class CheckResult:
    status: Literal["PASS", "FAIL", "WARNING"]
    detail: str
    raw_line: str


@dataclass
class ValidationResult:
    passed: bool
    checks: list[CheckResult]
    n_passed: int
    n_failed: int
    n_warnings: int
    script: str
    script_output: str
    script_exit_code: int
    script_error: str | None
    llm_assessment: str
    modules_used: list[str]
    reference_values: dict[str, str]
    refs_source: str
    summary: str


class Validator:
    """Standalone validator for computational / experimental results.

    Generates a validation script via LLM, executes it, and parses the
    PASS/FAIL/WARNING output lines to produce a structured verdict.
    """

    def __init__(
        self,
        llm_config: dict | LLMConfig | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        default_modules: list[str] | None = None,
        python_bin: str | None = None,
        work_dir: str | None = None,
        execution_timeout: int = 60,
    ):
        self._llm_config = self._build_llm_config(
            llm_config, model, api_key, base_url, temperature,
        )
        self._default_modules = default_modules
        self._python_bin = python_bin or sys.executable
        self._work_dir = work_dir or tempfile.mkdtemp(prefix="validator_")
        self._execution_timeout = execution_timeout

    # ── Public API ──────────────────────────────────────────────

    def validate(
        self,
        output: str,
        task_description: str = "",
        reference_values: dict[str, str] | None = None,
        modules: list[str] | None = None,
        timeout: int | None = None,
    ) -> ValidationResult:
        timeout = timeout or self._execution_timeout

        modules_used = self._resolve_modules(modules, task_description, output)

        # Determine reference source and auto-discover if needed
        if reference_values is not None:
            refs_source = "user"
        elif "literature_comparison" in modules_used:
            reference_values, refs_source = self._lookup_references(
                task_description, output,
            )
        else:
            refs_source = "user"

        script = self._generate_validation_script(
            output, task_description, modules_used, reference_values, refs_source,
        )

        refs = reference_values or {}

        if not script:
            return ValidationResult(
                passed=False, checks=[], n_passed=0, n_failed=0, n_warnings=0,
                script="", script_output="", script_exit_code=-1,
                script_error="LLM did not generate a validation script",
                llm_assessment="", modules_used=modules_used,
                reference_values=refs, refs_source=refs_source,
                summary="Validation FAILED: no validation script was generated.",
            )

        stdout, exit_code, stderr = self._execute_script(script, timeout)
        checks = self.parse_check_results(stdout)

        if not checks and exit_code != 0:
            return ValidationResult(
                passed=False, checks=[], n_passed=0, n_failed=0, n_warnings=0,
                script=script, script_output=stdout, script_exit_code=exit_code,
                script_error=stderr, llm_assessment="", modules_used=modules_used,
                reference_values=refs, refs_source=refs_source,
                summary=f"Validation FAILED: script crashed (exit code {exit_code}). {stderr or ''}",
            )

        # When refs are from LLM knowledge only (not literature), downgrade
        # reference-comparison FAILs to WARNING — LLM doesn't reliably do this itself.
        if refs_source == "llm":
            checks = self._apply_llm_leniency(checks)

        n_passed = sum(1 for c in checks if c.status == "PASS")
        n_failed = sum(1 for c in checks if c.status == "FAIL")
        n_warnings = sum(1 for c in checks if c.status == "WARNING")
        passed = n_failed == 0

        assessment = self._get_llm_assessment(stdout, task_description)
        summary = self._build_summary(checks)

        return ValidationResult(
            passed=passed,
            checks=checks,
            n_passed=n_passed,
            n_failed=n_failed,
            n_warnings=n_warnings,
            script=script,
            script_output=stdout,
            script_exit_code=exit_code,
            script_error=stderr,
            llm_assessment=assessment,
            modules_used=modules_used,
            reference_values=refs,
            refs_source=refs_source,
            summary=summary,
        )

    @staticmethod
    def parse_check_results(script_output: str) -> list[CheckResult]:
        results = []
        for line in script_output.splitlines():
            if _SUMMARY_RE.search(line):
                continue
            m = _CHECK_RE.match(line)
            if m:
                results.append(CheckResult(
                    status=m.group(1).upper(),
                    detail=m.group(2).strip(),
                    raw_line=line.strip(),
                ))
        return results

    # ── Internal methods ────────────────────────────────────────

    def _resolve_modules(
        self,
        explicit: list[str] | None,
        task_description: str,
        output: str,
    ) -> list[str]:
        if explicit:
            return self._filter_valid_modules(explicit)
        if self._default_modules:
            return self._filter_valid_modules(self._default_modules)
        return self._select_modules_via_llm(task_description, output[:500])

    def _filter_valid_modules(self, names: list[str]) -> list[str]:
        valid = set(get_all_modules().keys())
        return [n for n in names if n in valid]

    def _lookup_references(
        self, task_description: str, output: str,
    ) -> tuple[dict[str, str], str]:
        """Search literature for reference values, fall back to LLM knowledge."""
        from literature_search import search_arxiv, search_semantic_scholar

        search_text = ""
        try:
            search_text = search_arxiv(task_description)
        except Exception:
            pass
        if not search_text or "No results found" in search_text:
            try:
                search_text = search_semantic_scholar(task_description)
            except Exception:
                pass

        has_literature = bool(
            search_text
            and "No results found" not in search_text
            and "Error" not in search_text[:20]
        )

        lit_block = (
            f"Literature search results:\n{search_text[:2000]}\n\n"
            if has_literature
            else "No literature search results were found.\n\n"
        )

        message = (
            f"Task: {task_description}\n\n"
            f"Output:\n{output[:800]}\n\n"
            f"{lit_block}"
            f"Provide reference values for validating these computational results.\n\n"
            f"CRITICAL: The task describes a specific computational METHOD. You MUST provide\n"
            f"the expected range FOR THAT METHOD, not just raw experimental values.\n"
            f"Computational methods have known systematic biases:\n"
            f"- DFT-PBE: overestimates lattice constants ~1%, underestimates band gaps by\n"
            f"  up to 50%, underestimates surface energies by 20-30%\n"
            f"- Hartree-Fock: overestimates band gaps, minimal basis sets shift bond lengths\n"
            f"- B3LYP: harmonic frequencies ~5-10% above experimental fundamentals\n"
            f"Give the METHOD-SPECIFIC expected range as the primary reference. You may also\n"
            f"note the experimental value for context.\n\n"
            f"Tag sources:\n"
            f"- (source: literature) — from search results or well-known published values\n"
            f"- (source: experimental) — known experimental measurement\n"
            f"- (source: estimated) — your general knowledge, less certain\n\n"
            f"Format:\nREFERENCE VALUES:\n"
            f"- [quantity]: [method-specific range] (source: [type]); [notes]\n\n"
            f"If nothing is relevant, respond: NO REFERENCES FOUND"
        )
        response = self._single_llm_turn(
            system=(
                "You are a scientific reference specialist. Provide reference values "
                "for validating computational results. ALWAYS give the expected range "
                "for the SPECIFIC METHOD used in the task (e.g., DFT-PBE range, not "
                "just experimental). Tag each value with its source confidence."
            ),
            message=message,
        )
        refs = self._parse_reference_response(response)
        if not refs:
            return refs, "llm"
        # Check if any ref is actually from literature (not just estimated)
        all_values = " ".join(refs.values()).lower()
        has_concrete = any(
            tag in all_values
            for tag in ["source: literature", "source: experimental", "source: benchmark"]
        )
        return refs, "literature" if has_concrete else "llm"

    @staticmethod
    def _parse_reference_response(response: str) -> dict[str, str]:
        if "NO REFERENCES FOUND" in response.upper():
            return {}
        reference_values: dict[str, str] = {}
        in_section = False
        for line in response.splitlines():
            stripped = line.strip()
            if "REFERENCE VALUES" in stripped.upper():
                in_section = True
                continue
            if in_section and stripped.startswith("- "):
                entry = stripped[2:].strip()
                if ":" in entry:
                    key, val = entry.split(":", 1)
                    reference_values[key.strip()] = val.strip()
            elif in_section and stripped and not stripped.startswith("-"):
                in_section = False
        return reference_values

    def _select_modules_via_llm(self, task_description: str, output_snippet: str) -> list[str]:
        catalogue = get_module_catalogue()
        prompt = (
            f"Given this task and output, select the most relevant verification modules.\n\n"
            f"Task: {task_description}\n\n"
            f"Output snippet:\n{output_snippet}\n\n"
            f"Available modules:\n{catalogue}\n\n"
            f"Respond with:\nSELECTED MODULES:\n- module_name\n- ..."
        )
        response = self._single_llm_turn(
            system="Select the relevant verification modules from the list. "
                   "Respond with a SELECTED MODULES section listing module names.",
            message=prompt,
        )
        modules = parse_module_selections(response)
        return modules or DEFAULT_FALLBACK_MODULES

    def _generate_validation_script(
        self,
        output: str,
        task_description: str,
        modules: list[str],
        reference_values: dict[str, str] | None,
        refs_source: str = "user",
    ) -> str:
        evaluator_prompt = assemble_evaluator_prompt(
            selected_module_names=modules,
            reference_values=reference_values,
            refs_source=refs_source,
        )
        module_list = ", ".join(modules)
        context = f"Task: {task_description}\n\n" if task_description else ""
        message = (
            f"{context}"
            f"The following is the output from a computation. Write a Python validation "
            f"script to check these results.\n\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"1. You MUST include a check for EACH of these modules: {module_list}\n"
            f"   Do NOT skip any module. Each module must produce at least one PASS/FAIL line.\n"
            f"2. Each check must print exactly one line starting with 'PASS: ' or 'FAIL: '\n"
            f"3. Track results in a list: results = []. Append True for PASS, False for FAIL.\n"
            f"4. At the end: print 'VALIDATION PASSED' if all(results) else "
            f"'VALIDATION FAILED -- N issue(s) found'\n\n"
            f"{output}"
        )
        response = self._single_llm_turn(
            system=evaluator_prompt,
            message=message,
        )
        match = _CODE_BLOCK_RE.search(response)
        return match.group(1).strip() if match else ""

    def _execute_script(self, script: str, timeout: int) -> tuple[str, int, str | None]:
        fd, tmp_path = tempfile.mkstemp(suffix=".py", dir=self._work_dir)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(script)
            result = subprocess.run(
                [self._python_bin, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self._work_dir,
            )
            stderr = result.stderr.strip() if result.stderr.strip() else None
            return result.stdout, result.returncode, stderr
        except subprocess.TimeoutExpired:
            return "", 124, f"Script timed out after {timeout} seconds"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _get_llm_assessment(self, script_output: str, task_description: str) -> str:
        message = (
            f"Here is the output from running the validation script:\n\n"
            f"{script_output}\n\n"
            f"Give a brief (2-3 sentence) assessment of these results."
        )
        return self._single_llm_turn(
            system="You are a scientific validation specialist. Give a concise assessment.",
            message=message,
        )

    def _single_llm_turn(self, system: str, message: str) -> str:
        agent = ConversableAgent(
            name="ValidatorAgent",
            system_message=system,
            llm_config=self._llm_config,
            human_input_mode="NEVER",
        )
        sender = ConversableAgent(
            name="Caller",
            human_input_mode="NEVER",
            llm_config=False,
            max_consecutive_auto_reply=0,
        )
        chat = sender.initiate_chat(agent, message=message, max_turns=1, silent=True)
        return chat.summary or ""

    @staticmethod
    def _apply_llm_leniency(checks: list[CheckResult]) -> list[CheckResult]:
        """When references are from LLM knowledge only, downgrade small-deviation
        FAILs to WARNING. Keep FAILs for NaN/Inf, large deviations (>25%), etc."""
        hard_keywords = [
            "non-finite", "nan", "inf", "negative energy", "negative surface",
            "unphysical", "not finite",
        ]
        pct_re = re.compile(r"(\d+\.?\d*)\s*%")
        range_re = re.compile(
            r"(-?\d+\.?\d*)\s*\S*.*?outside.*?"
            r"\[(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\]",
        )
        threshold = 25.0

        result = []
        for c in checks:
            if c.status != "FAIL":
                result.append(c)
                continue
            lower = c.detail.lower()
            if any(kw in lower for kw in hard_keywords):
                result.append(c)
                continue

            deviation = None
            m = pct_re.search(c.detail)
            if m:
                deviation = float(m.group(1))
            else:
                m = range_re.search(c.detail)
                if m:
                    val, lo, hi = float(m.group(1)), float(m.group(2)), float(m.group(3))
                    mid = (lo + hi) / 2
                    if mid != 0:
                        deviation = abs(val - mid) / abs(mid) * 100

            if deviation is not None and deviation < threshold:
                result.append(CheckResult(
                    status="WARNING", detail=c.detail, raw_line=c.raw_line,
                ))
            else:
                result.append(c)
        return result

    @staticmethod
    def _build_summary(checks: list[CheckResult]) -> str:
        total = len(checks)
        n_failed = sum(1 for c in checks if c.status == "FAIL")
        n_warnings = sum(1 for c in checks if c.status == "WARNING")
        n_passed = sum(1 for c in checks if c.status == "PASS")

        if n_failed == 0:
            s = f"Validation PASSED: {n_passed} of {total} checks passed."
            if n_warnings:
                s += f" ({n_warnings} warning{'s' if n_warnings > 1 else ''})"
            return s

        failures = [c.detail for c in checks if c.status == "FAIL"]
        failure_str = "; ".join(failures[:5])
        if len(failures) > 5:
            failure_str += f"; ... and {len(failures) - 5} more"
        return (
            f"Validation FAILED: {n_failed} of {total} checks failed"
            f"{f', {n_warnings} warning(s)' if n_warnings else ''}. "
            f"Failed: {failure_str}"
        )

    @staticmethod
    def _build_llm_config(
        llm_config: dict | LLMConfig | None,
        model: str | None,
        api_key: str | None,
        base_url: str | None,
        temperature: float,
    ) -> LLMConfig:
        if llm_config is not None:
            if isinstance(llm_config, LLMConfig):
                return llm_config
            return LLMConfig(**llm_config)
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Provide api_key or set OPENAI_API_KEY env var")
        config = {"model": model or "gpt-5.4-mini", "api_key": api_key}
        if base_url:
            config["base_url"] = base_url
        return LLMConfig(config, temperature=temperature)
