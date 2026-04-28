"""
LangGraph Debate Orchestrator — v2.

Graph:
  START
    → profile_data       (sync-safe: pandas in executor)
    → run_fairness_tools (sync-safe: fairlearn in executor)
    → debate_round       (async: parallel LLM calls) ─┐
    ↑____________ round_router (conditional) ──────────┘
    ↓
    run_judge            (async: final LLM synthesis)
    → END

Events are pushed onto an asyncio.Queue that the SSE generator drains.
This keeps LangGraph fully decoupled from the HTTP layer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from backend.config import MAX_DEBATE_ROUNDS
from backend.engine.state import DebateState
from backend.agents.agent_runner import run_all_agents_parallel, run_judge
from backend.agents.prompts import JUDGE_AGENT
from backend.tools.data_profiler import profile_dataset
from backend.tools.fairness_metrics import compute_fairness_metrics
from backend.rl.trajectory import log_trajectory

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict:
    """
    Parse the judge's JSON response, handling markdown code-fences and
    stray text before/after the JSON object.
    """
    # 1. Try raw parse first (cleanest case)
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # 2. Strip common markdown fences
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # 3. Find the outermost { ... }
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try even more aggressive cleaning if still failing
            # (Sometimes LLMs put comments or trailing commas)
            pass

    # 4. Final attempt: find anything that looks like JSON
    # This is useful if there's text before/after the block
    try:
        # Find the first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
    except Exception:
        pass

    raise ValueError("No valid JSON object found in judge response.")


async def _run_in_executor(fn, *args) -> Any:
    """Run a blocking function in the default thread pool without blocking the loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

async def node_prep_data(state: DebateState) -> dict:
    csv = state.get("dataset_csv", "")
    protected = state.get("protected_attributes", [])
    target = state.get("target_column", "")

    if csv and protected and target:
        summary_task = _run_in_executor(profile_dataset, csv, protected, target)
        metrics_task = _run_in_executor(compute_fairness_metrics, csv, protected, target)
        summary, metrics = await asyncio.gather(summary_task, metrics_task)
    else:
        summary = "No dataset provided — agents will rely on the user query."
        metrics = {"note": "No dataset or attributes provided — metrics skipped."}

    log.debug("Prep complete: summary %d chars, metrics %d keys", len(summary), len(metrics))
    return {"dataset_summary": summary, "fairness_metrics": metrics}


async def node_debate_round(state: DebateState, config: RunnableConfig) -> dict:
    round_num = state.get("current_round", 0)
    queue: asyncio.Queue | None = config.get("configurable", {}).get("event_queue")

    if queue:
        await queue.put({"event": "round_start", "data": {"round": round_num}})

    async def _on_tool(tool_data):
        if queue:
            await queue.put({"event": "tool_use", "data": tool_data})

    try:
        messages = await run_all_agents_parallel(state, round_num, on_tool_use=_on_tool)
    except Exception as exc:
        log.error("Debate round %d failed: %s", round_num, exc, exc_info=True)
        raise

    if queue:
        for msg in messages:
            await queue.put({"event": "agent_message", "data": msg})

    return {"debate_messages": messages, "current_round": round_num + 1}


async def node_run_judge(state: DebateState, config: RunnableConfig) -> dict:
    queue: asyncio.Queue | None = config.get("configurable", {}).get("event_queue")

    if queue:
        await queue.put({
            "event": "agent_message",
            "data": {
                "agent": "final_judge",
                "role": f"{JUDGE_AGENT['emoji']} {JUDGE_AGENT['name']}",
                "content": "Synthesizing all arguments...",
                "round": state.get("current_round", 0),
            },
        })

    # Cooldown: let Gemini rate-limit windows fully reset before the judge fires
    await asyncio.sleep(8)

    raw = await run_judge(state)

    try:
        report = _extract_json(raw)
    except ValueError:
        log.warning("Judge returned unparseable JSON; raw=%r", raw[:300])
        # If JSON parse fails, show the raw text as the summary
        report = {
            "bias_score": None,
            "severity": "Pending",
            "summary": raw.strip() if raw.strip() else "The judge could not produce a structured report.",
            "flagged_issues": [],
            "mitigation_steps": [],
            "legal_risk": "Review manually",
            "confidence": "Low",
            "parse_error": True,
        }

    # Persist trajectory for RL
    try:
        debate_id = log_trajectory({**state, "final_report": report})
        report["debate_id"] = debate_id
    except Exception as exc:
        log.warning("Failed to log RL trajectory: %s", exc)
        report["debate_id"] = "log-failed"

    if queue:
        await queue.put({"event": "final_report", "data": report})
        await queue.put({"event": "done", "data": {}})

    return {"final_report": report}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _round_router(state: DebateState) -> str:
    if state.get("current_round", 0) < state.get("max_rounds", MAX_DEBATE_ROUNDS):
        return "debate_round"
    return "run_judge"


# ---------------------------------------------------------------------------
# Graph factory (compiled once per process)
# ---------------------------------------------------------------------------

def _build_graph():
    g = StateGraph(DebateState)
    g.add_node("prep_data",          node_prep_data)
    g.add_node("debate_round",       node_debate_round)
    g.add_node("run_judge",          node_run_judge)

    g.set_entry_point("prep_data")
    g.add_edge("prep_data",          "debate_round")
    g.add_conditional_edges(
        "debate_round",
        _round_router,
        {"debate_round": "debate_round", "run_judge": "run_judge"},
    )
    g.add_edge("run_judge", END)
    return g.compile()


_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Public streaming entry point
# ---------------------------------------------------------------------------

async def run_debate_stream(
    user_query: str,
    dataset_csv: str,
    use_case: str,
    protected_attributes: list[str],
    target_column: str,
    model_backend: str,
    max_rounds: int = MAX_DEBATE_ROUNDS,
) -> AsyncGenerator[dict, None]:
    """
    Run the full debate graph and yield SSE-compatible event dicts.
    The graph runs in a background Task; events are relayed via an asyncio.Queue.
    """
    queue: asyncio.Queue[dict] = asyncio.Queue()

    initial: DebateState = {
        "user_query":           user_query,
        "dataset_csv":          dataset_csv,
        "dataset_summary":      "",
        "use_case":             use_case,
        "protected_attributes": protected_attributes,
        "target_column":        target_column,
        "model_backend":        model_backend,
        "fairness_metrics":     {},
        "debate_messages":      [],
        "current_round":        0,
        "max_rounds":           max_rounds,
        "final_report":         {},
        "error":                "",
    }

    async def _run():
        try:
            await _GRAPH.ainvoke(initial, config={"configurable": {"event_queue": queue}})
        except Exception as exc:
            log.error("Graph execution failed: %s", exc, exc_info=True)
            await queue.put({"event": "error", "data": {"message": str(exc)}})
            await queue.put({"event": "done",  "data": {}})

    task = asyncio.create_task(_run())

    while True:
        event = await queue.get()
        yield event
        if event.get("event") == "done":
            break

    # Ensure task is collected even if client disconnects
    try:
        await asyncio.wait_for(task, timeout=5.0)
    except asyncio.TimeoutError:
        task.cancel()
