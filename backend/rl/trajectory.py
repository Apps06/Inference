"""
RL Trajectory Logger.

Logs every complete debate run as a JSONL record for future MARTI/OpenRLHF training.
Each record contains the full debate trajectory + reward signal.

File: rl/trajectories/debates.jsonl
Each line: one JSON object per debate run.
"""
import json
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path

from backend.engine.state import DebateState
from backend.rl.reward import compute_reward

# ── Storage path ─────────────────────────────────────────────────────────────
TRAJECTORY_DIR = Path(__file__).parent / "trajectories"
TRAJECTORY_FILE = TRAJECTORY_DIR / "debates.jsonl"


def _ensure_dir():
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)


def log_trajectory(state: DebateState) -> str:
    """
    Log a completed debate to JSONL. Returns the debate_id.
    """
    _ensure_dir()
    debate_id = str(uuid.uuid4())
    report = state.get("final_report", {})
    reward = compute_reward(state)

    record = {
        "debate_id": debate_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_backend": state.get("model_backend"),
        "use_case": state.get("use_case"),
        "protected_attributes": state.get("protected_attributes"),
        "target_column": state.get("target_column"),
        "max_rounds": state.get("max_rounds"),
        # Full transcript
        "trajectory": state.get("debate_messages", []),
        # Fairness ground truth
        "fairness_metrics": state.get("fairness_metrics", {}),
        # Final output
        "final_report": report,
        # RL signal
        "reward": reward,
    }

    with open(TRAJECTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return debate_id


def load_trajectories() -> list[dict]:
    """Load all logged trajectories for training."""
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
