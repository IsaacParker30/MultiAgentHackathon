"""JSON spec parsed from a chat message handed to the script generator agent."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class SimSpec:
    code: str
    job_name: str
    settings: dict[str, Any] = field(default_factory=dict)
    system: dict[str, Any] | None = None
    notes: str | None = None

    def to_prompt_block(self) -> str:
        return json.dumps(
            {
                "code": self.code,
                "job_name": self.job_name,
                "settings": self.settings,
                "system": self.system,
                "notes": self.notes,
            },
            indent=2,
        )


def _extract_json(message: str) -> dict[str, Any]:
    match = _FENCED_JSON.search(message)
    raw = match.group(1) if match else message.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Could not parse a JSON spec from the message. Wrap it in a ```json ... ``` block."
        ) from exc
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON must be an object.")
    return obj


def parse_spec(message: str, registered_codes: list[str] | None = None) -> SimSpec:
    obj = _extract_json(message)

    missing = [k for k in ("code", "job_name") if k not in obj]
    if missing:
        raise ValueError(f"Spec is missing required fields: {missing}")

    code = str(obj["code"]).lower()
    if registered_codes is not None and code not in registered_codes:
        raise ValueError(
            f"Unknown code {code!r}. Registered codes: {sorted(registered_codes)}"
        )

    return SimSpec(
        code=code,
        job_name=str(obj["job_name"]),
        settings=obj.get("settings") or {},
        system=obj.get("system"),
        notes=obj.get("notes"),
    )
