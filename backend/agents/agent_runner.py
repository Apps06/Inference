"""
Agent runner — v3 (Agentic with Tool Use).

Each agent is a genuine agentic loop:
  1. Receives context + debate history
  2. Reasons about what tools to call (Gemini function calling)
  3. Executes tools (statistical analysis, legal lookup, etc.)
  4. Incorporates tool results into final analysis

Agents fire sequentially with a stagger to respect free-tier rate limits.
Tool events are streamed to the frontend in real-time.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from backend.engine.models import generate, generate_agentic
from backend.engine.state import AgentMessage, DebateState
from backend.agents.prompts import (
    AGENTS,
    CONTEXT_BLOCK,
    DEBATE_HISTORY_BLOCK,
    JUDGE_AGENT,
)
from backend.agents.tools import AGENT_TOOLS, TOOL_REGISTRY

log = logging.getLogger(__name__)

_MAX_CSV_CHARS = 4_000
_MAX_HISTORY_CHARS = 8_000


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _context(state: DebateState) -> str:
    metrics = state.get("fairness_metrics", {})
    metrics_str = (
        "\n".join(f"  {k}: {v}" for k, v in metrics.items())
        if metrics else "  Not computed."
    )
    csv_snippet = (state.get("dataset_csv") or "")[:_MAX_CSV_CHARS]
    return CONTEXT_BLOCK.format(
        use_case=state.get("use_case", "general"),
        protected_attributes=", ".join(state.get("protected_attributes") or []),
        target_column=state.get("target_column", "unknown"),
        dataset_summary=state.get("dataset_summary", ""),
        fairness_metrics=metrics_str,
        dataset_csv=csv_snippet,
    )


def _history(state: DebateState, exclude_id: str = "") -> str:
    messages = state.get("debate_messages") or []
    if not messages:
        return "No prior arguments — provide your initial analysis."

    lines = [
        f"[Round {m['round'] + 1}] {m['role']}: {m['content']}"
        for m in messages
        if m.get("agent") != exclude_id
    ]
    text = "\n\n".join(lines)
    if len(text) > _MAX_HISTORY_CHARS:
        text = "...(truncated)\n\n" + text[-_MAX_HISTORY_CHARS:]

    return DEBATE_HISTORY_BLOCK.format(debate_history=text)


# ---------------------------------------------------------------------------
# Agentic prompt — instructs the model to use tools
# ---------------------------------------------------------------------------

AGENTIC_INSTRUCTION = """
You have access to analysis tools. Use them to strengthen your arguments with evidence.

IMPORTANT: When you have relevant data available, call at least one tool to ground your analysis in concrete evidence before giving your response. Do not just theorize — investigate.

After receiving tool results, incorporate the findings into your analysis. Cite specific numbers and metrics from the tool outputs.
"""


# ---------------------------------------------------------------------------
# Single agent — agentic execution
# ---------------------------------------------------------------------------

async def _run_one(
    agent_meta: dict,
    state: DebateState,
    round_num: int,
    agent_index: int,
    on_tool_use: Callable | None = None,
) -> AgentMessage:
    use_case = state.get("use_case", "general")
    system = agent_meta["system_prompt"].replace("{use_case}", use_case)
    context = _context(state)
    csv_text = state.get("dataset_csv", "")

    # Enhance system prompt with agentic instructions
    system_agentic = system + "\n" + AGENTIC_INSTRUCTION

    if round_num == 0:
        user_msg = (
            f"{context}\n\n"
            f"User question: {state.get('user_query', 'Audit this dataset for bias.')}\n\n"
            "Provide your initial analysis. Use your tools to investigate the data if available."
        )
    else:
        history = _history(state, exclude_id=agent_meta["id"])
        user_msg = (
            f"{context}\n\n"
            f"{history}\n\n"
            f"Round {round_num + 1}: Respond to the arguments above. "
            "Use your tools to verify or challenge claims with evidence. "
            "Update your position based on new findings."
        )

    # Get the agent's tool set
    agent_id = agent_meta["id"]
    tools = AGENT_TOOLS.get(agent_id)

    # Run the agentic loop
    content, tool_calls = await generate_agentic(
        system_prompt=system_agentic,
        user_message=user_msg,
        tools=tools,
        tool_registry=TOOL_REGISTRY,
        csv_text=csv_text,
        model_name=state.get("model_backend", "gemini-2.5-flash"),
        temperature=0.4 if round_num == 0 else 0.55,
        max_tokens=1024,
        agent_index=agent_index,
    )

    # Emit tool-use events for the frontend
    if on_tool_use and tool_calls:
        for tc in tool_calls:
            await on_tool_use({
                "agent": agent_id,
                "agent_name": agent_meta["name"],
                "tool": tc["tool"],
                "args": tc["args"],
            })

    # Format the response — prepend tool usage summary if tools were called
    if tool_calls:
        tool_summary = "\n".join(
            f"🔧 **Used tool:** `{tc['tool']}` → analyzed {', '.join(f'{k}={v}' for k, v in tc['args'].items())}"
            for tc in tool_calls
        )
        content = f"{tool_summary}\n\n{content}"

    return AgentMessage(
        agent=agent_id,
        role=agent_meta["name"],
        content=content,
        round=round_num,
    )


async def _run_one_safe(
    agent_meta: dict,
    state: DebateState,
    round_num: int,
    agent_index: int,
    on_tool_use: Callable | None = None,
) -> AgentMessage:
    """Wraps _run_one with per-agent error handling."""
    try:
        return await _run_one(agent_meta, state, round_num, agent_index, on_tool_use)
    except Exception as exc:
        log.error("Agent %s round %d failed: %s", agent_meta["id"], round_num, exc)
        return AgentMessage(
            agent=agent_meta["id"],
            role=agent_meta["name"],
            content=f"[Agent error — {exc}]",
            round=round_num,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_all_agents_parallel(
    state: DebateState,
    round_num: int,
    on_tool_use: Callable | None = None,
) -> list[AgentMessage]:
    """Run all 5 analyst agents with a stagger delay.

    Even with per-agent keys, Gemini free tier enforces IP-based rate limits.
    # Stagger: 10s delay between agents to prevent IP-based rate limiting on free tier
    """
    results: list[AgentMessage] = []
    for i, agent in enumerate(AGENTS):
        if i > 0:
            await asyncio.sleep(10)
        msg = await _run_one_safe(agent, state, round_num, agent_index=i, on_tool_use=on_tool_use)
        results.append(msg)
    return results


async def run_judge(state: DebateState) -> str:
    """Run the Final Judge and return its raw response string.
    Uses simple generation (no tools) — the judge synthesizes, not investigates."""
    use_case = state.get("use_case", "general")
    system = JUDGE_AGENT["system_prompt"].replace("{use_case}", use_case)
    context = _context(state)
    history = _history(state)

    user_msg = (
        f"{context}\n\n"
        f"{history}\n\n"
        "All debate rounds are complete. "
        "Produce the final bias audit report as valid JSON."
    )

    try:
        return await generate(
            system_prompt=system,
            user_message=user_msg,
            model_name=state.get("model_backend", "gemini-2.5-flash"),
            temperature=0.2,
            max_tokens=2048,
            agent_index=0,
        )
    except Exception as exc:
        log.error("Final judge failed: %s", exc)
        import json
        return json.dumps({
            "bias_score": 0,
            "severity": "ERROR",
            "summary": f"MODEL FAILURE: The judge could not synthesize a final report. Error: {exc}",
            "flagged_issues": [],
            "mitigation_steps": [],
            "legal_risk": "N/A",
            "confidence": "N/A"
        })
