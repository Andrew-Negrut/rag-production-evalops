from __future__ import annotations

from typing import Dict, List


# Empty allowlist by default. Populate explicitly before enabling tools.
TOOL_ALLOWLIST: Dict[str, Dict[str, str]] = {}


def allowed_tools() -> List[dict]:
    return [
        {"name": name, "description": meta.get("description", "")}
        for name, meta in TOOL_ALLOWLIST.items()
    ]
