"""
RL Trajectory Logger — v2.

Logs every complete debate run as a JSONL record for MARTI/OpenRLHF training.
v2 adds:
  - per_agent_rewards   : individual credit assignment per agent
  - marti_format        : restructured trajectory in MARTI-native schema
  - get_trajectory()    : retrieve a single debate by ID
  - get_stats()         : aggregate statistics across all logged debates
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.engine.state import DebateState
from backend.rl.reward import compute_reward, compute_per_agent_rewards

# ── Storage path ─────────────────────────────────────────────────────────────
TRAJECTORY_DIR  = Path(__file__).parent / "trajectories"
TRAJECTORY_FILE = TRAJECTORY_DIR / "debates.jsonl"


def _ensure_dir() -> None:
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# MARTI format converter
# ---------------------------------------------------------------------------

def _to_marti_format(
    state:          DebateState,
    per_agent_rwds: dict[str, dict],
) -> list[dict]:
    """
    Convert a debate trajectory to MARTI's expected schema.

    MARTI expects a list of (agent_id, observation, action, reward) tuples
    where:
      - observation = the context visible to the agent before its turn
      - action      = the agent's response text
      - reward      = scalar float from per-agent reward
    """
    messages     = state.get("debate_messages", [])
    context_base = {
        "use_case":             state.get("use_case"),
        "protected_attributes": state.get("protected_attributes"),
        "target_column":        state.get("target_column"),
        "fairness_metrics":     state.get("fairness_metrics", {}),
        "dataset_summary":      state.get("dataset_summary", ""),
        "user_query":           state.get("user_query", ""),
    }

    marti_turns = []
    for i, msg in enumerate(messages):
        agent_id = msg.get("agent", "unknown")
        round_num = msg.get("round", 0)

        # Observation = everything the agent could see before its turn
        prior_messages = [
            {"agent": m["agent"], "role": m["role"], "content": m["content"], "round": m["round"]}
            for m in messages[:i]
        ]
        observation = {**context_base, "prior_messages": prior_messages, "round": round_num}

        agent_reward = per_agent_rwds.get(agent_id, {})

        marti_turns.append({
            "agent_id":    agent_id,
            "round":       round_num,
            "observation": observation,
            "action":      msg.get("content", ""),
            "reward":      agent_reward.get("total", 0.0),
            "reward_breakdown": agent_reward,
        })

    return marti_turns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_trajectory(state: DebateState) -> str:
    """
    Log a completed debate to JSONL. Returns the debate_id.
    """
    _ensure_dir()
    debate_id       = str(uuid.uuid4())
    report          = state.get("final_report", {})
    reward          = compute_reward(state)
    per_agent_rwds  = compute_per_agent_rewards(state)
    marti_format    = _to_marti_format(state, per_agent_rwds)

    record = {
        "debate_id":          debate_id,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "model_backend":      state.get("model_backend"),
        "use_case":           state.get("use_case"),
        "protected_attributes": state.get("protected_attributes"),
        "target_column":      state.get("target_column"),
        "max_rounds":         state.get("max_rounds"),
        # Full transcript
        "trajectory":         state.get("debate_messages", []),
        # Fairness ground truth
        "fairness_metrics":   state.get("fairness_metrics", {}),
        # Final output
        "final_report":       report,
        # RL signals
        "reward":             reward,
        "per_agent_rewards":  per_agent_rwds,
        # MARTI-native format
        "marti_format":       marti_format,
    }

    with open(TRAJECTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return debate_id


def load_trajectories() -> list[dict]:
    """Load all logged trajectories for training or display."""
    _ensure_dir()
    if not TRAJECTORY_FILE.exists():
        return []
    records = []
    with open(TRAJECTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def get_trajectory(debate_id: str) -> Optional[dict]:
    """Retrieve a single debate record by ID. Returns None if not found."""
    _ensure_dir()
    if not TRAJECTORY_FILE.exists():
        return None
    with open(TRAJECTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if debate_id in line:
                try:
                    record = json.loads(line)
                    if record.get("debate_id") == debate_id:
                        return record
                except json.JSONDecodeError:
                    continue
    return None


def get_stats() -> dict:
    """
    Aggregate statistics across all logged trajectories.
    Returns totals, averages, and per-agent reward means.
    """
    records = load_trajectories()
    if not records:
        return {
            "total_debates": 0,
            "avg_bias_score": None,
            "avg_total_reward": None,
            "avg_per_agent_rewards": {},
            "use_case_breakdown": {},
            "model_breakdown": {},
        }

    bias_scores   = [r["final_report"].get("bias_score", 0) for r in records if r.get("final_report")]
    total_rewards = [r["reward"].get("total", 0) for r in records if r.get("reward")]

    # Per-agent reward means
    agent_reward_sums: dict[str, dict[str, float]] = {}
    agent_counts: dict[str, int] = {}
    for r in records:
        for agent_id, rwds in (r.get("per_agent_rewards") or {}).items():
            if agent_id not in agent_reward_sums:
                agent_reward_sums[agent_id] = {}
                agent_counts[agent_id] = 0
            agent_counts[agent_id] += 1
            for k, v in rwds.items():
                if isinstance(v, (int, float)):
                    agent_reward_sums[agent_id][k] = agent_reward_sums[agent_id].get(k, 0.0) + v

    avg_per_agent: dict[str, dict] = {}
    for agent_id, sums in agent_reward_sums.items():
        n = agent_counts[agent_id]
        avg_per_agent[agent_id] = {k: round(v / n, 4) for k, v in sums.items()}

    # Use-case and model breakdowns
    use_case_counts: dict[str, int] = {}
    model_counts:    dict[str, int] = {}
    for r in records:
        uc = r.get("use_case", "unknown")
        mb = r.get("model_backend", "unknown")
        use_case_counts[uc] = use_case_counts.get(uc, 0) + 1
        model_counts[mb]    = model_counts.get(mb, 0) + 1

    return {
        "total_debates":      len(records),
        "avg_bias_score":     round(sum(bias_scores) / len(bias_scores), 2) if bias_scores else None,
        "avg_total_reward":   round(sum(total_rewards) / len(total_rewards), 4) if total_rewards else None,
        "avg_per_agent_rewards": avg_per_agent,
        "use_case_breakdown": use_case_counts,
        "model_breakdown":    model_counts,
    }


def get_trajectory_list(limit: int = 50) -> list[dict]:
    """
    Return a lightweight summary list of past debates (no full transcript).
    Sorted newest-first.
    """
    records = load_trajectories()
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    summaries = []
    for r in records[:limit]:
        rpt = r.get("final_report", {})
        summaries.append({
            "debate_id":    r.get("debate_id"),
            "timestamp":    r.get("timestamp"),
            "use_case":     r.get("use_case"),
            "model_backend": r.get("model_backend"),
            "max_rounds":   r.get("max_rounds"),
            "protected_attributes": r.get("protected_attributes"),
            "target_column": r.get("target_column"),
            "bias_score":   rpt.get("bias_score"),
            "severity":     rpt.get("severity"),
            "total_reward": (r.get("reward") or {}).get("total"),
        })
    return summaries
