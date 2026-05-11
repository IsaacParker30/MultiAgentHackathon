from dataclasses import dataclass, field


@dataclass
class VerificationModule:
    name: str
    description: str
    prompt_snippet: str
    code_example: str
    applicability_hint: str
    priority: int = 50
    requires_reference_values: bool = False


_REGISTRY: dict[str, VerificationModule] = {}


def register(
    name: str,
    description: str,
    prompt_snippet: str,
    code_example: str,
    applicability_hint: str,
    priority: int = 50,
    requires_reference_values: bool = False,
) -> VerificationModule:
    module = VerificationModule(
        name=name,
        description=description,
        prompt_snippet=prompt_snippet,
        code_example=code_example,
        applicability_hint=applicability_hint,
        priority=priority,
        requires_reference_values=requires_reference_values,
    )
    _REGISTRY[module.name] = module
    return module


def get_all_modules() -> dict[str, VerificationModule]:
    return dict(_REGISTRY)


def get_modules_by_names(names: list[str]) -> list[VerificationModule]:
    modules = [_REGISTRY[n] for n in names if n in _REGISTRY]
    return sorted(modules, key=lambda m: m.priority, reverse=True)


def get_module_catalogue() -> str:
    lines = []
    for mod in sorted(_REGISTRY.values(), key=lambda m: m.priority, reverse=True):
        lines.append(f"- **{mod.name}**: {mod.description}")
        lines.append(f"  When to use: {mod.applicability_hint}")
    return "\n".join(lines)
