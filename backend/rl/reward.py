"""
RL Reward Function.

Computes a scalar reward [0.0, 1.0] for a completed debate trajectory.
This reward will be used as the training signal in MARTI/PPO training.

Reward components:
1. Fairness Coverage     — did the agents catch real metric violations?
2. Report Quality        — is the final report well-formed and actionable?
3. Debate Depth          — did agents challenge each other substantively?
4. Legal Completeness    — were relevant regulations cited?
5. Mitigation Actionability — are mitigation steps concrete?
"""
from backend.engine.state import DebateState


def compute_reward(state: DebateState) -> dict:
    """
    Returns a dict with component scores and a total [0.0, 1.0].
    """
    report = state.get("final_report", {})
    metrics = state.get("fairness_metrics", {})
    messages = state.get("debate_messages", [])

    scores = {}

    # ── 1. Fairness Coverage ─────────────────────────────────────────────────
    # Reward if bias_score correlates with real violations
    real_violations = sum(
        1 for k, v in metrics.items()
        if ("violated" in k and v is True) or
           ("difference" in k and isinstance(v, float) and abs(v) > 0.1)
    )
    reported_issues = len(report.get("flagged_issues", []))
    bias_score = report.get("bias_score", 0)

    if real_violations > 0:
        coverage = min(reported_issues / real_violations, 1.0)
    else:
        # Reward correctly finding no bias (bias_score should be low)
        coverage = 1.0 if bias_score < 30 else 0.3

    scores["fairness_coverage"] = round(coverage, 3)

    # ── 2. Report Quality ────────────────────────────────────────────────────
    required_keys = {"bias_score", "flagged_issues", "mitigation_steps", "summary", "severity"}
    present_keys = set(report.keys())
    quality = len(required_keys & present_keys) / len(required_keys)
    scores["report_quality"] = round(quality, 3)

    # ── 3. Debate Depth ──────────────────────────────────────────────────────
    # Reward multi-round debates where agents reference each other
    rounds = max((m.get("round", 0) for m in messages), default=0) + 1
    cross_refs = sum(
        1 for m in messages
        if any(ref in (m.get("content") or "").lower()
               for ref in ["statistician", "auditor", "adversary", "reviewer", "expert", "judge"])
    )
    depth_score = min((cross_refs / max(len(messages), 1)) * 1.5, 1.0)
    depth_score = (depth_score + min(rounds / 3, 1.0)) / 2
    scores["debate_depth"] = round(depth_score, 3)

    # ── 4. Legal Completeness ────────────────────────────────────────────────
    legal_keywords = ["dpdp", "gdpr", "eeoc", "article 15", "title vii", "regulation", "act", "law", "legal"]
    legal_hits = sum(
        1 for m in messages
        if any(kw in (m.get("content") or "").lower() for kw in legal_keywords)
    )
    scores["legal_completeness"] = round(min(legal_hits / 3, 1.0), 3)

    # ── 5. Mitigation Actionability ──────────────────────────────────────────
    mitigations = report.get("mitigation_steps", [])
    if not mitigations:
        scores["mitigation_actionability"] = 0.0
    else:
        actionable = sum(
            1 for m in mitigations
            if isinstance(m, dict) and len(m.get("description", "")) > 50
        )
        scores["mitigation_actionability"] = round(actionable / max(len(mitigations), 1), 3)

    # ── Total Reward ─────────────────────────────────────────────────────────
    weights = {
        "fairness_coverage": 0.35,
        "report_quality": 0.20,
        "debate_depth": 0.20,
        "legal_completeness": 0.10,
        "mitigation_actionability": 0.15,
    }
    total = sum(scores[k] * weights[k] for k in weights)
    scores["total"] = round(total, 4)

    return scores
