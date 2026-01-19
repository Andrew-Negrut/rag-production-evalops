from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Protocol, Dict, Any


@dataclass(frozen=True)
class LLMResult:
    text: str
    usage: Optional[Dict[str, Any]] = None  # tokens/cost later


class LLMProvider(Protocol):
    def generate(self, *, system: str, user: str, tools: Optional[List[dict]] = None) -> LLMResult: ...


class FakeProvider:
    """
    Deterministic provider for tests + CI (no network, no API keys).

    Strategy:
    - Parse the SOURCES block produced by build_sources_block()
    - Reuse a short span from source [1] so the answer is always grounded
    - Always include [1] so citation plumbing is exercised
    """
    def generate(self, *, system: str, user: str, tools: Optional[List[dict]] = None) -> LLMResult:
        text = (user or "").strip()

        # Try to extract the content of source [1]
        src1 = ""
        if "[1]" in text:
            after = text.split("[1]", 1)[1]

            # Stop at next source marker or USER QUESTION.
            for stop in ("\n[2]", "\n[3]", "\n[4]", "\n[5]", "\nUSER QUESTION:"):
                if stop in after:
                    after = after.split(stop, 1)[0]
                    break

            src1 = after.strip()

        if not src1:
            # Fallback: still non-refusal + citation.
            return LLMResult(text="The provided sources contain relevant information. [1]")

        # Use a short snippet from source [1] as the "answer"
        snippet = " ".join(src1.split())[:220].rstrip()
        return LLMResult(text=f"{snippet} [1]")



class OpenAIProvider:
    def __init__(self, api_key: str, model: str):
        self.model = model
        # OpenAI official python client
        from openai import OpenAI  # type: ignore
        self.client = OpenAI(api_key=api_key)

    def generate(self, *, system: str, user: str, tools: Optional[List[dict]] = None) -> LLMResult:
        # Uses the Responses API. Tool calling can be added later via tools=...
        # Function calling flow is documented by OpenAI. :contentReference[oaicite:0]{index=0}
        kwargs = {}
        if tools:
            kwargs["tools"] = tools

        resp = self.client.responses.create(
            model=self.model,
            instructions=system,
            input=[{"role": "user", "content": user}],
            **kwargs,
        )
        return LLMResult(text=resp.output_text, usage=getattr(resp, "usage", None))


def get_llm_provider() -> LLMProvider:
    provider = os.environ.get("LLM_PROVIDER", "fake").lower()
    if provider == "fake":
        return FakeProvider()

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        model = os.environ.get("LLM_MODEL", "gpt-5").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAIProvider(api_key=api_key, model=model)

    raise RuntimeError(f"Unknown LLM_PROVIDER={provider}")
