from verification.registry import (
    VerificationModule,
    get_all_modules,
    get_module_catalogue,
    get_modules_by_names,
    register,
)
from verification.assembler import assemble_evaluator_prompt
from verification.selector import parse_module_selections

# Import all modules to trigger registration
import verification.modules  # noqa: F401

__all__ = [
    "VerificationModule",
    "get_all_modules",
    "get_module_catalogue",
    "get_modules_by_names",
    "register",
    "assemble_evaluator_prompt",
    "parse_module_selections",
]
