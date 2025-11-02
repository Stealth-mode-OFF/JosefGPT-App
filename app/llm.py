from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

from app.config import get_settings


class BaseChatLLM:
    mode: str = "unknown"
    model_name: str = "unknown"

    def generate(self, messages: Sequence[dict], *, temperature: float, max_tokens: int) -> str:
        raise NotImplementedError


class OfflineChatLLM(BaseChatLLM):
    mode = "offline"
    model_name = "rule-based-summariser"

    def generate(self, messages: Sequence[dict], *, temperature: float, max_tokens: int) -> str:
        user_content = ""
        system_content = ""
        for message in messages:
            role = message.get("role")
            if role == "user":
                user_content = message.get("content", "")
            elif role == "system":
                system_content = message.get("content", "")

        context_snippets: List[str] = []
        question = user_content
        if "Context:" in user_content:
            ctx_part = user_content.split("Context:", 1)[1]
            if "Question:" in ctx_part:
                context_body, question = ctx_part.split("Question:", 1)
            else:
                context_body, question = ctx_part, ""
            for raw_chunk in context_body.split("\n---"):
                chunk = " ".join(raw_chunk.strip().split())
                if chunk:
                    context_snippets.append(chunk[:220])
        question = " ".join(question.strip().split())

        parts: List[str] = ["🤖 Offline JosefGPT response"]
        if system_content:
            parts.append(f"System focus: {system_content.strip()}")
        if question:
            parts.append(f"Question: {question}")
        if context_snippets:
            parts.append("Context highlights:")
            for snippet in context_snippets[:3]:
                parts.append(f"- {snippet}")
        parts.append(
            "Guidance: leverage available context, run small experiments, and automate repeatable wins."
        )
        return "\n".join(parts)


class OpenAIChatLLM(BaseChatLLM):
    mode = "openai"

    def __init__(self, model_name: str):
        from openai import OpenAI  # delayed import to avoid dependency in offline mode

        self.model_name = model_name
        self._client = OpenAI()

    def generate(self, messages: Sequence[dict], *, temperature: float, max_tokens: int) -> str:
        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()


@lru_cache(maxsize=1)
def get_chat_llm() -> BaseChatLLM:
    settings = get_settings()
    mode = (settings.llm_mode or "auto").strip().lower()
    api_key_present = bool(settings.openai_api_key)
    if mode == "offline":
        return OfflineChatLLM()
    if mode == "online":
        mode = "openai"
    if mode in {"openai", "auto"} and api_key_present:
        try:
            return OpenAIChatLLM(settings.openai_model)
        except Exception:  # pragma: no cover - safeguard if OpenAI init fails
            if mode == "openai":
                raise
    return OfflineChatLLM()
