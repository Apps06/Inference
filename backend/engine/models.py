"""
Model abstraction layer — v3 (Agentic).

Per-agent key isolation with Gemini → OpenAI fallback.
Supports Gemini function calling for agentic tool use.
"""
from __future__ import annotations

import logging
from typing import AsyncGenerator

from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI
from backend.config import GEMINI_KEYS, OPENAI_KEYS, GEMINI_MODEL, OPENAI_MODEL

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-agent client pools (lazily initialised)
# ---------------------------------------------------------------------------

_gemini_clients: dict[int, genai.Client] = {}
_openai_clients: dict[int, AsyncOpenAI] = {}


def _get_gemini(agent_index: int) -> genai.Client:
    """Return a Gemini client using the key assigned to this agent index."""
    if agent_index not in _gemini_clients:
        key = GEMINI_KEYS[agent_index % len(GEMINI_KEYS)]
        if not key:
            raise RuntimeError(f"GEMINI_KEY_{agent_index + 1} is not set in .env")
        _gemini_clients[agent_index] = genai.Client(api_key=key)
    return _gemini_clients[agent_index]


def _get_openai(agent_index: int) -> AsyncOpenAI:
    """Return an OpenAI client using the key assigned to this agent index."""
    if agent_index not in _openai_clients:
        key = OPENAI_KEYS[agent_index % len(OPENAI_KEYS)]
        if not key:
            raise RuntimeError(f"OPENAI_KEY_{agent_index + 1} is not set in .env")
        _openai_clients[agent_index] = AsyncOpenAI(api_key=key)
    return _openai_clients[agent_index]


async def _call_gemini_with_retry(client: genai.Client, model: str, contents: list, config: genai_types.GenerateContentConfig, max_retries: int = 3):
    """Wraps Gemini generate_content with exponential backoff for 429 rate limits."""
    import asyncio
    for attempt in range(max_retries):
        try:
            return await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "too many" in err_str:
                if attempt < max_retries - 1:
                    sleep_time = (attempt + 1) * 15  # 15s, 30s
                    log.warning("Gemini 429 Rate Limit hit. Sleeping %ds before retry %d/3...", sleep_time, attempt + 1)
                    await asyncio.sleep(sleep_time)
                    continue
            raise


# ---------------------------------------------------------------------------
# Simple generation (no tool use) — Gemini → OpenAI fallback
# ---------------------------------------------------------------------------

async def generate(
    system_prompt: str,
    user_message: str,
    model_name: str = "",
    temperature: float = 0.4,
    max_tokens: int = 2048,
    agent_index: int = 0,
) -> str:
    """
    Generate a response. Tries Gemini first, falls back to OpenAI.
    No tool use — used for the judge and non-agentic calls.
    """
    gemini_model = model_name or GEMINI_MODEL

    # ── Step 1: Try Gemini ──────────────────────────────────
    gemini_key = GEMINI_KEYS[agent_index % len(GEMINI_KEYS)]
    if gemini_key:
        try:
            client = _get_gemini(agent_index)
            combined_prompt = f"{system_prompt}\n\n{user_message}"
            response = await _call_gemini_with_retry(
                client=client,
                model=gemini_model,
                contents=[combined_prompt],
                config=genai_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            text = response.text or ""
            if text:
                log.info("Agent %d: Gemini succeeded (model=%s)", agent_index, gemini_model)
                return text
        except Exception as exc:
            log.warning("Agent %d: Gemini failed (%s). Falling back to OpenAI...", agent_index, exc)

    # ── Step 2: Fallback to OpenAI ──────────────────────────
    openai_key = OPENAI_KEYS[agent_index % len(OPENAI_KEYS)]
    if openai_key:
        try:
            client = _get_openai(agent_index)
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            log.info("Agent %d: OpenAI succeeded (model=%s)", agent_index, OPENAI_MODEL)
            return text
        except Exception as exc:
            log.error("Agent %d: OpenAI also failed: %s", agent_index, exc)
            raise RuntimeError(
                f"All models failed for agent {agent_index}. "
                f"OpenAI error: {exc}"
            ) from exc

    raise RuntimeError(
        f"No API keys configured for agent index {agent_index}. "
        f"Set GEMINI_KEY_{agent_index + 1} or OPENAI_KEY_{agent_index + 1} in .env"
    )


# ---------------------------------------------------------------------------
# AGENTIC generation — Gemini function calling with tool use loop
# ---------------------------------------------------------------------------

async def generate_agentic(
    system_prompt: str,
    user_message: str,
    tools: genai_types.Tool | None = None,
    tool_registry: dict | None = None,
    csv_text: str = "",
    model_name: str = "",
    temperature: float = 0.4,
    max_tokens: int = 2048,
    agent_index: int = 0,
    max_tool_rounds: int = 2,
) -> tuple[str, list[dict]]:
    """
    Agentic generation: Gemini reasons, optionally calls tools, then produces
    a final answer incorporating tool results.

    Returns:
        (final_text, tool_calls)
        tool_calls is a list of {"tool": name, "args": {...}, "result": "..."}
    """
    gemini_model = model_name or GEMINI_MODEL
    tool_calls_log: list[dict] = []

    gemini_key = GEMINI_KEYS[agent_index % len(GEMINI_KEYS)]
    if not gemini_key:
        # No Gemini key — fall back to simple generation (no tools)
        text = await generate(system_prompt, user_message, model_name, temperature, max_tokens, agent_index)
        return text, []

    client = _get_gemini(agent_index)
    combined_prompt = f"{system_prompt}\n\n{user_message}"

    # Build config
    config = genai_types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if tools:
        config.tools = [tools]

    try:
        # Initial call — Gemini may return text OR function calls
        contents = [combined_prompt]

        for _round in range(max_tool_rounds + 1):
            response = await _call_gemini_with_retry(
                client=client,
                model=gemini_model,
                contents=contents,
                config=config,
            )

            # Check if Gemini wants to call a function
            function_calls = []
            text_parts = []

            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        function_calls.append(part.function_call)
                    elif part.text:
                        text_parts.append(part.text)

            if not function_calls:
                # No more tool calls — return the text
                final_text = "\n".join(text_parts) or response.text or ""
                log.info(
                    "Agent %d: Gemini agentic completed with %d tool calls",
                    agent_index, len(tool_calls_log),
                )
                return final_text, tool_calls_log

            # Execute each tool call
            function_responses = []
            for fc in function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}

                log.info("Agent %d: calling tool '%s' with args %s", agent_index, tool_name, tool_args)

                # Execute the tool
                if tool_registry and tool_name in tool_registry:
                    fn = tool_registry[tool_name]
                    # Inject csv_text for tools that need it
                    import inspect
                    sig = inspect.signature(fn)
                    if "csv_text" in sig.parameters:
                        tool_args["csv_text"] = csv_text
                    result_str = fn(**tool_args)
                else:
                    result_str = f'{{"error": "Unknown tool: {tool_name}"}}'

                tool_calls_log.append({
                    "tool": tool_name,
                    "args": {k: v for k, v in tool_args.items() if k != "csv_text"},
                    "result": result_str,
                })

                function_responses.append(
                    genai_types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result_str},
                    )
                )

            # Wait 3s before sending results back to Gemini to avoid RPM burst
            await asyncio.sleep(3)

            # Build next turn: assistant's response + function results
            contents = [
                combined_prompt,
                genai_types.Content(
                    role="model",
                    parts=response.candidates[0].content.parts,
                ),
                genai_types.Content(
                    role="user",
                    parts=function_responses,
                ),
            ]

        # If we exhausted rounds, return whatever text we have
        return response.text or "", tool_calls_log

    except Exception as exc:
        log.warning("Agent %d: Gemini agentic failed (%s). Falling back to simple generation.", agent_index, exc)
        # Fallback: simple generation without tools
        text = await generate(system_prompt, user_message, model_name, temperature, max_tokens, agent_index)
        return text, []
