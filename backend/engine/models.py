"""
Model abstraction layer.

Provides a single async interface `generate()` that works with both
xAI Grok (OpenAI-compatible REST) and Google Gemini (google-genai SDK).

Critical fix over v1:
- Gemini streaming previously used a sync queue.Queue filled inside
  run_in_executor, which blocked the entire executor thread for the
  duration of the stream before yielding anything. This is now replaced
  with the native `client.aio` async interface from google-genai.
- asyncio.get_event_loop() is deprecated in 3.10+; replaced with
  asyncio.get_running_loop() where needed.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI, APIError as OpenAIAPIError
from google import genai
from google.genai import types as genai_types

from backend.config import XAI_API_KEY, XAI_BASE_URL, GOOGLE_API_KEY

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client singletons — created once per process lifetime
# ---------------------------------------------------------------------------

_grok_client: AsyncOpenAI | None = None
_gemini_client: genai.Client | None = None


def _grok() -> AsyncOpenAI:
    global _grok_client
    if _grok_client is None:
        if not XAI_API_KEY:
            raise RuntimeError("XAI_API_KEY is not set in .env")
        _grok_client = AsyncOpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL)
    return _grok_client


def _gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set in .env")
        _gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _gemini_client


# ---------------------------------------------------------------------------
# Router helpers
# ---------------------------------------------------------------------------

def _is_grok(model: str) -> bool:
    return model.startswith("grok")


def _gemini_config(system_prompt: str, temperature: float, max_tokens: int) -> genai_types.GenerateContentConfig:
    return genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def generate(
    system_prompt: str,
    user_message: str,
    model_name: str,
    temperature: float = 0.4,
    max_tokens: int = 2048,
) -> str:
    """
    Generate a complete response. Returns the full text string.
    Raises RuntimeError on model/API failure after logging.
    """
    try:
        if _is_grok(model_name):
            return await _grok_complete(system_prompt, user_message, model_name, temperature, max_tokens)
        return await _gemini_complete(system_prompt, user_message, model_name, temperature, max_tokens)
    except Exception as exc:
        log.error("generate() failed for model=%s: %s", model_name, exc, exc_info=True)
        raise RuntimeError(f"Model call failed ({model_name}): {exc}") from exc


async def generate_stream(
    system_prompt: str,
    user_message: str,
    model_name: str,
    temperature: float = 0.4,
    max_tokens: int = 2048,
) -> AsyncGenerator[str, None]:
    """
    Yield text chunks as they arrive from the model.
    Uses truly async streaming — never blocks the event loop.
    """
    try:
        if _is_grok(model_name):
            async for chunk in _grok_stream(system_prompt, user_message, model_name, temperature, max_tokens):
                yield chunk
        else:
            async for chunk in _gemini_stream(system_prompt, user_message, model_name, temperature, max_tokens):
                yield chunk
    except Exception as exc:
        log.error("generate_stream() failed for model=%s: %s", model_name, exc, exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Grok — xAI (OpenAI-compatible REST)
# ---------------------------------------------------------------------------

async def _grok_complete(system_prompt, user_message, model_name, temperature, max_tokens) -> str:
    response = await _grok().chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


async def _grok_stream(system_prompt, user_message, model_name, temperature, max_tokens) -> AsyncGenerator[str, None]:
    stream = await _grok().chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    async for chunk in stream:
        text = chunk.choices[0].delta.content
        if text:
            yield text


# ---------------------------------------------------------------------------
# Gemini — google-genai SDK (native async via client.aio)
# ---------------------------------------------------------------------------

async def _gemini_complete(system_prompt, user_message, model_name, temperature, max_tokens) -> str:
    """
    Uses client.aio.models.generate_content() — the true async path in the
    google-genai SDK (no thread executor needed).
    """
    cfg = _gemini_config(system_prompt, temperature, max_tokens)
    response = await _gemini().aio.models.generate_content(
        model=model_name,
        contents=user_message,
        config=cfg,
    )
    return response.text or ""


async def _gemini_stream(system_prompt, user_message, model_name, temperature, max_tokens) -> AsyncGenerator[str, None]:
    """
    Uses client.aio.models.generate_content_stream() — truly async; each
    chunk is yielded as soon as the model produces it without blocking the
    event loop or an executor thread.
    """
    cfg = _gemini_config(system_prompt, temperature, max_tokens)
    async for chunk in await _gemini().aio.models.generate_content_stream(
        model=model_name,
        contents=user_message,
        config=cfg,
    ):
        if chunk.text:
            yield chunk.text
