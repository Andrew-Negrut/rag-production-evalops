# from __future__ import annotations

# from dataclasses import dataclass
# from typing import List

# from app.citations import extract_citation_numbers


# @dataclass
# class ValidationResult:
#     ok: bool
#     errors: List[str]


# def validate_grounded_answer(text: str, max_source_index: int) -> ValidationResult:
#     """
#     Production-friendly validation:
#     - Caller handles exact refusal string separately.
#     - Otherwise: require at least one citation overall.
#     - All citations must be integers in [1..max_source_index].
#     """
#     t = (text or "").strip()
#     if not t:
#         return ValidationResult(ok=False, errors=["empty answer"])

#     nums = extract_citation_numbers(t)

#     if len(nums) == 0:
#         return ValidationResult(ok=False, errors=["no citations found"])

#     bad = [n for n in nums if n < 1 or n > max_source_index]
#     if bad:
#         return ValidationResult(ok=False, errors=[f"invalid citation indices: {sorted(set(bad))}"])

#     return ValidationResult(ok=True, errors=[])

# from __future__ import annotations

# import re
# from dataclasses import dataclass
# from typing import List

# from app.citations import extract_citation_numbers


# _REFUSAL_RE = re.compile(
#     r"\b(i\s+don't\s+know|i\s+do\s+not\s+know|not\s+enough\s+info|insufficient\s+info)\b",
#     re.IGNORECASE,
# )


# @dataclass
# class ValidationResult:
#     ok: bool
#     errors: List[str]


# def validate_grounded_answer(text: str, max_source_index: int) -> ValidationResult:
#     """
#     Option B validation:
#     - Refusals are allowed without citations.
#     - Otherwise: require at least one citation overall.
#     - All citations must be integers in [1..max_source_index].
#     """
#     t = (text or "").strip()
#     if not t:
#         return ValidationResult(ok=False, errors=["empty answer"])

#     # Allow refusals without citations.
#     if _REFUSAL_RE.search(t):
#         return ValidationResult(ok=True, errors=[])

#     nums = extract_citation_numbers(t)
#     if len(nums) == 0:
#         return ValidationResult(ok=False, errors=["no citations found"])

#     bad = [n for n in nums if n < 1 or n > max_source_index]
#     if bad:
#         return ValidationResult(ok=False, errors=[f"invalid citation indices: {sorted(set(bad))}"])

#     return ValidationResult(ok=True, errors=[])

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.citations import extract_citation_numbers

REFUSAL_PREFIX = "I don't know based on the provided sources."


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]


def validate_grounded_answer(text: str, max_source_index: int) -> ValidationResult:
    """
    Option B validation (recommended):
    - Exact refusals (REFUSAL_PREFIX) are allowed without citations.
    - Otherwise: require at least one citation overall.
    - All citations must be integers in [1..max_source_index].
    """
    t = (text or "").strip()
    if not t:
        return ValidationResult(ok=False, errors=["empty answer"])

    # Allow ONLY the exact refusal contract without citations.
    if t.startswith(REFUSAL_PREFIX):
        return ValidationResult(ok=True, errors=[])

    nums = extract_citation_numbers(t)
    if len(nums) == 0:
        return ValidationResult(ok=False, errors=["no citations found"])

    bad = [n for n in nums if n < 1 or n > max_source_index]
    if bad:
        return ValidationResult(ok=False, errors=[f"invalid citation indices: {sorted(set(bad))}"])

    return ValidationResult(ok=True, errors=[])
