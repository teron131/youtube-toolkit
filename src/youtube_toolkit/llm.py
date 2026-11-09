"""A chat model on the LLM that the LLM_* settings name, on any OpenAI-compatible API: each caller picks its own optional settings, and `invoke` turns one system and one user message into the reply text."""

from __future__ import annotations

from openai import OpenAI, omit
from openai.types.shared import ReasoningEffort

from .config import LLMConfig

REQUEST_TIMEOUT_SECONDS = 120


class ChatModel:
    """A setting left as None is not sent, so the provider's default applies; every call, from any thread, shares one connection pool, which closes when the model is garbage-collected."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        reasoning_effort: ReasoningEffort = None,
        temperature: float | None = None,
    ) -> None:
        self._model = config.model
        self._reasoning_effort = omit if reasoning_effort is None else reasoning_effort
        self._temperature = omit if temperature is None else temperature
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=3,
        )

    def invoke(self, system: str, user: str) -> str:
        """The reply text, empty when the model sends none; the SDK retries transport failures."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            reasoning_effort=self._reasoning_effort,
            temperature=self._temperature,
        )
        content = response.choices[0].message.content if response.choices else None
        return content or ""
