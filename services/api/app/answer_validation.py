from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.citations import extract_citation_numbers


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]


def validate_grounded_answer(text: str, max_source_index: int) -> ValidationResult:
    """
    Production-friendly validation:
    - Caller handles exact refusal string separately.
    - Otherwise: require at least one citation overall.
    - All citations must be integers in [1..max_source_index].
    """
    t = (text or "").strip()
    if not t:
        return ValidationResult(ok=False, errors=["empty answer"])

    nums = extract_citation_numbers(t)

    if len(nums) == 0:
        return ValidationResult(ok=False, errors=["no citations found"])

    bad = [n for n in nums if n < 1 or n > max_source_index]
    if bad:
        return ValidationResult(ok=False, errors=[f"invalid citation indices: {sorted(set(bad))}"])

    return ValidationResult(ok=True, errors=[])
