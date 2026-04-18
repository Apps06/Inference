"""
DebateState — shared state object that flows through the LangGraph graph.
All agents read from and write to this state.
"""
from typing import TypedDict, Annotated
import operator


class AgentMessage(TypedDict):
    agent: str
    role: str         # emoji + display label
    content: str
    round: int


class DebateState(TypedDict):
    # ── Input ────────────────────────────────────────────
    user_query: str                   # Raw user chat message
    dataset_summary: str              # Profiled stats from data_profiler
    dataset_csv: str                  # Raw CSV text (first 200 rows)
    use_case: str                     # e.g. "job_hiring", "loan_approval"
    protected_attributes: list[str]   # e.g. ["gender", "age", "caste"]
    target_column: str                # e.g. "hired", "approved"
    model_backend: str                # "grok" | "gemini"

    # ── Fairness Tool Output ─────────────────────────────
    fairness_metrics: dict            # Output from Fairlearn/basic stats

    # ── Debate Messages (append-only) ────────────────────
    debate_messages: Annotated[list[AgentMessage], operator.add]

    # ── Round Control ────────────────────────────────────
    current_round: int
    max_rounds: int

    # ── Final Output ─────────────────────────────────────
    final_report: dict    # bias_score, flagged_issues, mitigations, etc.
    error: str            # Non-empty if something went wrong
