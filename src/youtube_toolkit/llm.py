"""A chat model on any OpenAI-compatible Chat Completions API, set up by the LLM_* settings."""

from __future__ import annotations

from typing import overload

from openai import AsyncOpenAI, OpenAI, OpenAIError, omit
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ParsedChatCompletion,
)
from openai.types.shared import ReasoningEffort
from pydantic import BaseModel, ValidationError

from .config import LLMConfig

REQUEST_TIMEOUT_SECONDS = 120


class ChatModel:
    """Settings left as None are not sent, so the host's defaults apply. Sync calls from any thread share one connection pool, closed when the model is garbage-collected; async calls must all run on one event loop, since the async pool binds to the first."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        reasoning_effort: ReasoningEffort = None,
        temperature: float | None = None,
    ) -> None:
        self._model = config.model
        self._host = config.host
        self._reasoning_effort = omit if reasoning_effort is None else reasoning_effort
        self._temperature = omit if temperature is None else temperature
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=3,
        )
        self._async_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=3,
        )

    @overload
    def invoke(self, system: str, user: str) -> str: ...
    @overload
    def invoke[Reply: BaseModel](
        self, system: str, user: str, *, schema: type[Reply]
    ) -> Reply: ...
    def invoke(
        self,
        system: str,
        user: str,
        *,
        schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        """The reply text, empty when the model sends none, or with `schema`, the reply as that Pydantic model through structured outputs; a request that still fails after the SDK's retries, or a reply that does not fit `schema`, raises RuntimeError."""
        messages = _messages(system, user)
        try:
            if schema is not None:
                return _parsed(
                    self._client.chat.completions.parse(
                        model=self._model,
                        messages=messages,
                        reasoning_effort=self._reasoning_effort,
                        temperature=self._temperature,
                        response_format=schema,
                    )
                )
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                reasoning_effort=self._reasoning_effort,
                temperature=self._temperature,
            )
        except (OpenAIError, ValidationError) as error:
            raise RuntimeError(f"{self._host} chat request failed: {error}") from error
        return _reply(response)

    @overload
    async def ainvoke(self, system: str, user: str) -> str: ...
    @overload
    async def ainvoke[Reply: BaseModel](
        self, system: str, user: str, *, schema: type[Reply]
    ) -> Reply: ...
    async def ainvoke(
        self,
        system: str,
        user: str,
        *,
        schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        """`invoke` on the async client."""
        messages = _messages(system, user)
        try:
            if schema is not None:
                return _parsed(
                    await self._async_client.chat.completions.parse(
                        model=self._model,
                        messages=messages,
                        reasoning_effort=self._reasoning_effort,
                        temperature=self._temperature,
                        response_format=schema,
                    )
                )
            response = await self._async_client.chat.completions.create(
                model=self._model,
                messages=messages,
                reasoning_effort=self._reasoning_effort,
                temperature=self._temperature,
            )
        except (OpenAIError, ValidationError) as error:
            raise RuntimeError(f"{self._host} chat request failed: {error}") from error
        return _reply(response)


def _messages(system: str, user: str) -> list[ChatCompletionMessageParam]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _reply(response: ChatCompletion) -> str:
    content = response.choices[0].message.content if response.choices else None
    return content or ""


def _parsed[Reply: BaseModel](response: ParsedChatCompletion[Reply]) -> Reply:
    parsed = response.choices[0].message.parsed if response.choices else None
    if parsed is None:
        raise RuntimeError("The model refused or sent no reply that fits the schema")
    return parsed
