"""
RL Reward Function — v3.

Fixes over v2:
- `import re` moved to module level (was re-imported inside the function
  body on every single call).
- `content_relevance` completely rewritten. v2 matched agent text against
  raw metric dict keys (e.g. "gender_demographic_parity_difference") which
  agents never literally write. Now extracts:
    (a) numeric values from the metrics dict as both decimal strings ("0.5")
        and percentage strings ("50%") — agents commonly write "50%".
    (b) column names mentioned in protected_attributes / target_column.
    (c) use-case domain keywords.
  This makes the score non-zero for agents that clearly discuss the data.

All other component logic is unchanged.
"""
from __future__ import annotations

import re
from backend.engine.state import DebateState

# ── Agent role keywords ──────────────────────────────────────────────────────
_ROLE_KEYWORDS: dict[str, list[str]] = {
    "data_statistician": [
        "percent", "rate", "mean", "median", "distribution",
        "correlation", "disparity", "proportion", "statistic",
        "sample", "missing", "confidence", "significance",
    ],
    "fairness_auditor": [
        "disparate impact", "statistical parity", "equalized odds",
        "calibration", "4/5", "80%", "threshold", "violation",
        "fairness", "metric", "benchmark", "definition",
    ],
    "domain_expert": [
        "context", "domain", "real-world", "systemic", "historical",
        "pipeline", "proxy", "qualification", "legitimate", "downstream",
        "impact", "root cause",
    ],
    "bias_adversary": [
        "counter", "alternative", "confound", "sample size", "methodology",
        "cherry", "reverse", "business necessity", "steelman", "defense",
        "challenge", "weakness",
    ],
    "ethical_reviewer": [
        "dpdp", "gdpr", "eeoc", "article", "title vii", "regulation",
        "law", "act", "constitutional", "transparency", "accountability",
        "intersect", "procedural",
    ],
    "final_judge": [
        "synthesiz", "weigh", "evidence", "resolv", "consensus",
        "bias_score", "severity", "mitigation", "recommend",
    ],
}

_LEGAL_KEYWORDS = [
    "dpdp", "gdpr", "eeoc", "article 15", "title vii",
    "regulation", "act", "law", "legal",
]

_CROSS_REF_TERMS = [
    "statistician", "auditor", "adversary", "reviewer", "expert", "judge",
    "agent 1", "agent 2", "agent 3", "agent 4", "agent 5",
]

# Regex that finds standalone numbers (e.g. "0.5", "40", "-0.4")
_NUM_RE = re.compile(r"-?\d+\.?\d*")


def _build_content_tokens(
    metrics: dict,
    protected_attributes: list[str],
    target_column: str,
    use_case: str,
) -> set[str]:
    """
    Build a set of tokens we expect a relevant agent to mention.

    Includes:
    - Numeric metric values as decimal strings AND percentage strings
      (0.5 → "0.5" and "50%"; -0.4 → "0.4" in "40%" form, etc.)
    - Protected attribute names (e.g. "gender", "caste")
    - Target column name (e.g. "hired")
    - Use-case domain words (e.g. "job_hiring" → "hiring", "job")
    """
    tokens: set[str] = set()

    # Numeric values from metrics
    for v in metrics.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            fv = float(v)
            tokens.add(str(round(abs(fv), 2)))          # "0.5", "0.4"
            pct = round(abs(fv) * 100)
            if pct > 0:
                tokens.add(f"{pct}%")                    # "50%", "40%"
                tokens.add(str(pct))                     # "50", "40"

    # Protected attributes and target column
    for attr in (protected_attributes or []):
        tokens.add(attr.lower())
    if target_column:
        tokens.add(target_column.lower())

    # Use-case words
    for word in re.split(r"[_\-\s]+", (use_case or "")):
        if len(word) > 2:
            tokens.add(word.lower())

    # Remove trivially short tokens that would cause false positives
    tokens = {t for t in tokens if len(t) >= 2}
    return tokens


# ---------------------------------------------------------------------------
# Global reward
# ---------------------------------------------------------------------------

def compute_reward(state: DebateState) -> dict:
    """
    Returns a dict with component scores and a total [0.0, 1.0].
    """
    report   = state.get("final_report", {})
    metrics  = state.get("fairness_metrics", {})
    messages = state.get("debate_messages", [])

    scores: dict[str, float] = {}

    # ── 1. Fairness Coverage ─────────────────────────────────────────────
    real_violations = sum(
        1 for k, v in metrics.items()
        if ("violated" in k and v is True)
        or ("difference" in k and isinstance(v, float) and abs(v) > 0.1)
    )
    reported_issues = len(report.get("flagged_issues", []))
    bias_score = report.get("bias_score", 0)

    if real_violations > 0:
        coverage = min(reported_issues / real_violations, 1.0)
    else:
        coverage = 1.0 if bias_score < 30 else 0.3

    scores["fairness_coverage"] = round(coverage, 3)

    # ── 2. Report Quality ────────────────────────────────────────────────
    required_keys = {"bias_score", "flagged_issues", "mitigation_steps", "summary", "severity"}
    quality = len(required_keys & set(report.keys())) / len(required_keys)
    scores["report_quality"] = round(quality, 3)

    # ── 3. Debate Depth ──────────────────────────────────────────────────
    rounds = max((m.get("round", 0) for m in messages), default=0) + 1
    cross_refs = sum(
        1 for m in messages
        if any(ref in (m.get("content") or "").lower() for ref in _CROSS_REF_TERMS)
    )
    depth_score = min((cross_refs / max(len(messages), 1)) * 1.5, 1.0)
    depth_score = (depth_score + min(rounds / 3, 1.0)) / 2
    scores["debate_depth"] = round(depth_score, 3)

    # ── 4. Legal Completeness ─────────────────────────────────────────────
    legal_hits = sum(
        1 for m in messages
        if any(kw in (m.get("content") or "").lower() for kw in _LEGAL_KEYWORDS)
    )
    scores["legal_completeness"] = round(min(legal_hits / 3, 1.0), 3)

    # ── 5. Mitigation Actionability ───────────────────────────────────────
    mitigations = report.get("mitigation_steps", [])
    if not mitigations:
        scores["mitigation_actionability"] = 0.0
    else:
        actionable = sum(
            1 for m in mitigations
            if isinstance(m, dict) and len(m.get("description", "")) > 50
        )
        scores["mitigation_actionability"] = round(actionable / max(len(mitigations), 1), 3)

    # ── Total ─────────────────────────────────────────────────────────────
    weights = {
        "fairness_coverage":        0.35,
        "report_quality":           0.20,
        "debate_depth":             0.20,
        "legal_completeness":       0.10,
        "mitigation_actionability": 0.15,
    }
    scores["total"] = round(sum(scores[k] * weights[k] for k in weights), 4)
    return scores


# ---------------------------------------------------------------------------
# Per-agent reward
# ---------------------------------------------------------------------------

def compute_per_agent_rewards(state: DebateState) -> dict[str, dict]:
    """
    Compute a reward breakdown for every individual agent.

    Components:
      content_relevance      — mentions actual metric values / column names
      cross_agent_engagement — references other agents by role name
      role_adherence         — uses keywords appropriate to the agent's role
      specificity            — contains numbers and percentages
    """
    messages   = state.get("debate_messages", [])
    metrics    = state.get("fairness_metrics", {})
    protected  = state.get("protected_attributes") or []
    target_col = state.get("target_column", "")
    use_case   = state.get("use_case", "")

    # Build expected content tokens once for the whole debate
    content_tokens = _build_content_tokens(metrics, protected, target_col, use_case)

    # Group all messages by agent_id
    agent_msgs: dict[str, list[str]] = {}
    for m in messages:
        aid = m.get("agent", "unknown")
        agent_msgs.setdefault(aid, []).append((m.get("content") or "").lower())

    per_agent: dict[str, dict] = {}

    for agent_id, contents in agent_msgs.items():
        full_text = " ".join(contents)
        role_kws  = _ROLE_KEYWORDS.get(agent_id, [])

        # A. Content relevance — how many expected tokens appear in the text
        if content_tokens:
            hits = sum(1 for tok in content_tokens if tok in full_text)
            # Normalise: hitting 30%+ of tokens is "perfect"
            content_relevance = min(hits / max(len(content_tokens) * 0.3, 1), 1.0)
        else:
            content_relevance = 0.5   # no dataset — neutral score

        # B. Cross-agent engagement — mentions other agent role names
        cross_hits = sum(1 for term in _CROSS_REF_TERMS if term in full_text)
        cross_engagement = min(cross_hits / 3, 1.0)

        # C. Role adherence — uses keywords appropriate to role
        if role_kws:
            role_hits = sum(1 for kw in role_kws if kw.lower() in full_text)
            role_adherence = min(role_hits / max(len(role_kws) * 0.4, 1), 1.0)
        else:
            role_adherence = 0.5

        # D. Specificity — count standalone numbers / percentages in text
        num_matches = len(_NUM_RE.findall(full_text))
        specificity = min(num_matches / 5, 1.0)

        agent_weights = {
            "content_relevance":      0.30,
            "cross_agent_engagement": 0.25,
            "role_adherence":         0.25,
            "specificity":            0.20,
        }
        component_scores = {
            "content_relevance":      round(content_relevance, 3),
            "cross_agent_engagement": round(cross_engagement, 3),
            "role_adherence":         round(role_adherence, 3),
            "specificity":            round(specificity, 3),
        }
        component_scores["total"] = round(
            sum(component_scores[k] * agent_weights[k] for k in agent_weights), 4
        )
        per_agent[agent_id] = component_scores

    return per_agent
