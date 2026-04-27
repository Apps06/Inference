"""
Agent runner — v2.

Changes from v1:
- Per-agent exception handling: a single agent failure is caught and
  replaced with an error AgentMessage, so one bad API call can't crash
  an entire debate round.
- Context is capped to avoid silent token-limit failures for large CSVs.
- Use_case template variable is substituted cleanly.
"""
from __future__ import annotations

import asyncio
import logging

from backend.engine.models import generate
from backend.engine.state import AgentMessage, DebateState
from backend.agents.prompts import (
    AGENTS,
    CONTEXT_BLOCK,
    DEBATE_HISTORY_BLOCK,
    JUDGE_AGENT,
)

log = logging.getLogger(__name__)

_MAX_CSV_CHARS = 4_000   # ~1k tokens; keeps prompt sane for large datasets
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
# Single agent
# ---------------------------------------------------------------------------

async def _run_one(agent_meta: dict, state: DebateState, round_num: int) -> AgentMessage:
    use_case = state.get("use_case", "general")
    system = agent_meta["system_prompt"].replace("{use_case}", use_case)
    context = _context(state)

    if round_num == 0:
        user_msg = (
            f"{context}\n\n"
            f"User question: {state.get('user_query', 'Audit this dataset for bias.')}\n\n"
            "Provide your initial analysis."
        )
    else:
        history = _history(state, exclude_id=agent_meta["id"])
        user_msg = (
            f"{context}\n\n"
            f"{history}\n\n"
            f"Round {round_num + 1}: Respond to the arguments above. "
            "Challenge weak points, defend strong ones, and update your position."
        )

    content = await generate(
        system_prompt=system,
        user_message=user_msg,
        model_name="grok-4.20-reasoning",
        temperature=0.4 if round_num == 0 else 0.55,
        max_tokens=1024,
    )
    return AgentMessage(
        agent=agent_meta["id"],
        role=agent_meta["name"],
        content=content,
        round=round_num,
    )


async def _run_one_safe(agent_meta: dict, state: DebateState, round_num: int) -> AgentMessage:
    """Wraps _run_one with per-agent error handling so one failure can't abort the round."""
    try:
        return await _run_one(agent_meta, state, round_num)
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

async def run_all_agents_parallel(state: DebateState, round_num: int) -> list[AgentMessage]:
    """Run all 5 analyst agents concurrently for a single debate round."""
    tasks = [_run_one_safe(agent, state, round_num) for agent in AGENTS]
    return list(await asyncio.gather(*tasks))


async def run_judge(state: DebateState) -> str:
    """Run the Final Judge and return its raw response string."""
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
        # Primary attempt: Use Gemini for the final judgment
        return await generate(
            system_prompt=system,
            user_message=user_msg,
            model_name="gemini-2.5-pro",
            temperature=0.2,
            max_tokens=2048,
        )
    except Exception as exc:
        log.warning("Gemini failed during final judgment: %s. Falling back to Grok.", exc)
        try:
            # Fallback attempt: Use Grok if Gemini is down
            return await generate(
                system_prompt=system,
                user_message=user_msg,
                model_name="grok-4.20-reasoning",
                temperature=0.2,
                max_tokens=2048,
            )
        except Exception as fallback_exc:
            log.error("Fallback Grok model also failed: %s", fallback_exc)
            # Final safety net: Return a valid JSON error string so the frontend doesn't crash
            import json
            return json.dumps({
                "error": "Model Failure",
                "message": "Both primary (Gemini) and fallback (Grok) models failed to generate the final report."
            })
