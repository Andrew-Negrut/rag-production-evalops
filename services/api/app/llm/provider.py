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
    """
    def generate(self, *, system: str, user: str, tools: Optional[List[dict]] = None) -> LLMResult:
        # Always cite [1] so your citation plumbing is exercised.
        return LLMResult(text="From the sources, Qdrant stores vectors for similarity search. [1]")


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
