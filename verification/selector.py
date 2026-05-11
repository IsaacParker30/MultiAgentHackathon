import re

from verification.registry import get_all_modules


def parse_module_selections(evaluator_message: str) -> list[str]:
    """Parse the Evaluator's planning-phase response to extract selected module names.

    The Evaluator is instructed to list modules in a SELECTED MODULES section.
    This function looks for module names that appear in the registry.
    """
    all_module_names = set(get_all_modules().keys())
    selected = []

    lines = evaluator_message.splitlines()
    in_selection_section = False

    for line in lines:
        stripped = line.strip().lower()

        if "selected modules" in stripped or "i will use" in stripped or "modules:" in stripped:
            in_selection_section = True
            # Check if modules are listed inline on this same line
            inline_found = _extract_names_from_line(line, all_module_names)
            selected.extend(inline_found)
            continue

        if in_selection_section:
            if stripped.startswith("- ") or stripped.startswith("* ") or re.match(r"^\d+\.", stripped):
                found = _extract_names_from_line(line, all_module_names)
                selected.extend(found)
            elif stripped == "":
                continue
            elif any(header in stripped for header in ["success criteria", "validation", "checks"]):
                in_selection_section = False
            else:
                found = _extract_names_from_line(line, all_module_names)
                if found:
                    selected.extend(found)
                else:
                    in_selection_section = False

    # Fallback: scan entire message for module names if nothing found in structured section
    if not selected:
        selected = _extract_names_from_line(evaluator_message, all_module_names)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for name in selected:
        if name not in seen:
            seen.add(name)
            unique.append(name)

    return unique


def _extract_names_from_line(text: str, valid_names: set[str]) -> list[str]:
    """Find all valid module names that appear in a line of text."""
    found = []
    text_lower = text.lower()
    for name in valid_names:
        if name in text_lower:
            found.append(name)
    return found
