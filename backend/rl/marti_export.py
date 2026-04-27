"""
MARTI Export Script.

Converts debates.jsonl into formats ready for:
  1. MARTI v2 training  (marti_train.jsonl)
  2. GRPO group sampling (grpo_groups.jsonl)
  3. OpenRLHF SFT warm-up (sft_pairs.jsonl)

Usage:
    python -m backend.rl.marti_export
    python -m backend.rl.marti_export --out-dir ./exports --min-reward 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.rl.trajectory import load_trajectories, TRAJECTORY_DIR


# ---------------------------------------------------------------------------
# MARTI v2 native format
# ---------------------------------------------------------------------------

def export_marti(records: list[dict], out_path: Path) -> int:
    """
    Export MARTI-native training file.

    Schema per line:
    {
      "debate_id": str,
      "use_case":  str,
      "turns": [
        {
          "agent_id":    str,
          "round":       int,
          "observation": dict,
          "action":      str,
          "reward":      float,
          "reward_breakdown": dict
        }, ...
      ],
      "global_reward": float,
      "global_reward_breakdown": dict
    }
    """
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            marti_turns = r.get("marti_format")
            if not marti_turns:
                # Fallback: reconstruct from raw trajectory
                marti_turns = []
                per_agent = r.get("per_agent_rewards", {})
                for msg in r.get("trajectory", []):
                    aid = msg.get("agent", "unknown")
                    marti_turns.append({
                        "agent_id":         aid,
                        "round":            msg.get("round", 0),
                        "observation":      {"user_query": r.get("user_query", "")},
                        "action":           msg.get("content", ""),
                        "reward":           (per_agent.get(aid) or {}).get("total", 0.0),
                        "reward_breakdown": per_agent.get(aid, {}),
                    })

            record = {
                "debate_id":                r.get("debate_id"),
                "use_case":                 r.get("use_case"),
                "model_backend":            r.get("model_backend"),
                "protected_attributes":     r.get("protected_attributes"),
                "target_column":            r.get("target_column"),
                "turns":                    marti_turns,
                "global_reward":            (r.get("reward") or {}).get("total", 0.0),
                "global_reward_breakdown":  r.get("reward", {}),
            }
            f.write(json.dumps(record) + "\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# GRPO group format
# ---------------------------------------------------------------------------

def export_grpo(records: list[dict], out_path: Path, group_size: int = 4) -> int:
    """
    Export GRPO-compatible groups.

    GRPO requires multiple completions for the EXACT SAME prompt so it can
    compute group-relative advantages. We group turns by a hash of their
    full observation (which includes prior_messages).

    Schema per line:
    {
      "prompt_id": str,
      "role":      str,
      "prompt":    str,
      "completions": [
        {"text": str, "reward": float}, ...
      ]
    }
    """
    from hashlib import md5

    # Group by (agent_id, observation_hash) to ensure identical prompts
    groups: dict[tuple[str, str], list[dict]] = {}
    obs_strings: dict[str, str] = {}
    use_cases: dict[str, str] = {}

    for r in records:
        uc = r.get("use_case", "general")
        for turn in (r.get("marti_format") or []):
            aid = turn.get("agent_id", "unknown")
            obs_str = json.dumps(turn.get("observation", {}), sort_keys=True)
            obs_hash = md5(obs_str.encode()).hexdigest()

            key = (aid, obs_hash)
            obs_strings[obs_hash] = obs_str
            use_cases[obs_hash] = uc

            groups.setdefault(key, []).append({
                "text":   turn.get("action", ""),
                "reward": turn.get("reward", 0.0),
            })

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for (aid, obs_hash), turns in groups.items():
            # Chunk into groups of `group_size`
            for i in range(0, len(turns), group_size):
                chunk = turns[i : i + group_size]
                if len(chunk) < 2:
                    continue  # GRPO needs at least 2 completions for relative scoring
                
                prompt_id = md5(f"{obs_hash}|{aid}|{i}".encode()).hexdigest()[:12]
                record = {
                    "prompt_id":   prompt_id,
                    "use_case":    use_cases[obs_hash],
                    "agent_id":    aid,
                    "prompt":      obs_strings[obs_hash],
                    "completions": [{"text": t["text"], "reward": t["reward"]} for t in chunk],
                }
                f.write(json.dumps(record) + "\n")
                written += 1
    return written


# ---------------------------------------------------------------------------
# OpenRLHF SFT warm-up pairs
# ---------------------------------------------------------------------------

def export_sft(records: list[dict], out_path: Path, min_reward: float = 0.6) -> int:
    """
    Export high-quality debate turns as supervised fine-tuning (SFT) pairs.
    Only includes turns with reward >= min_reward.

    Schema per line:
    {
      "system":    str,  # agent system prompt placeholder
      "prompt":    str,  # context / observation
      "response":  str,  # agent's high-quality response
      "reward":    float
    }
    """
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            for turn in (r.get("marti_format") or []):
                if turn.get("reward", 0) < min_reward:
                    continue
                obs = turn.get("observation", {})
                prompt_text = (
                    f"Use case: {obs.get('use_case', 'general')}\n"
                    f"Protected attributes: {obs.get('protected_attributes', [])}\n"
                    f"Target column: {obs.get('target_column', '')}\n"
                    f"User query: {obs.get('user_query', '')}\n"
                    f"Dataset summary: {obs.get('dataset_summary', '')[:500]}\n"
                    f"Prior debate:\n" +
                    "\n".join(
                        f"  [{m.get('agent')}]: {(m.get('content') or '')[:300]}"
                        for m in (obs.get("prior_messages") or [])[-4:]
                    )
                )
                record = {
                    "system":   f"You are the {turn['agent_id'].replace('_', ' ').title()} agent.",
                    "prompt":   prompt_text,
                    "response": turn.get("action", ""),
                    "reward":   turn.get("reward", 0.0),
                }
                f.write(json.dumps(record) + "\n")
                written += 1
    return written


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export INFERENCE debate trajectories for MARTI/GRPO/OpenRLHF training."
    )
    parser.add_argument(
        "--out-dir", default=str(TRAJECTORY_DIR / "exports"),
        help="Output directory (default: backend/rl/trajectories/exports/)"
    )
    parser.add_argument(
        "--min-reward", type=float, default=0.5,
        help="Minimum global reward to include a debate in the export (default: 0.5)"
    )
    parser.add_argument(
        "--grpo-group-size", type=int, default=4,
        help="Number of completions per GRPO group (default: 4)"
    )
    parser.add_argument(
        "--sft-min-reward", type=float, default=0.6,
        help="Minimum per-agent reward for SFT pairs (default: 0.6)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading trajectories...")
    all_records = load_trajectories()
    print(f"  Found {len(all_records)} debates total.")

    filtered = [r for r in all_records if (r.get("reward") or {}).get("total", 0) >= args.min_reward]
    print(f"  {len(filtered)} debates meet min_reward >= {args.min_reward}")

    if not filtered:
        print("No debates meet the minimum reward threshold. Run some debates first.")
        sys.exit(0)

    # MARTI v2
    marti_path = out_dir / "marti_train.jsonl"
    n = export_marti(filtered, marti_path)
    print(f"  [MARTI]  {n} records → {marti_path}")

    # GRPO
    grpo_path = out_dir / "grpo_groups.jsonl"
    n = export_grpo(filtered, grpo_path, group_size=args.grpo_group_size)
    print(f"  [GRPO]   {n} groups  → {grpo_path}")

    # SFT
    sft_path = out_dir / "sft_pairs.jsonl"
    n = export_sft(filtered, sft_path, min_reward=args.sft_min_reward)
    print(f"  [SFT]    {n} pairs   → {sft_path}")

    print("\nDone. To train with MARTI:")
    print(f"  git clone https://github.com/TsinghuaC3I/MARTI.git")
    print(f"  cd MARTI && pip install -e .")
    print(f"  python scripts/train_multi_agent.py --data {marti_path}")


if __name__ == "__main__":
    main()
