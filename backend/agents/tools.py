"""
Agent Tools — callable functions that agents can invoke during reasoning.

Each tool is a plain Python function. Gemini function-calling decides
which tools to call; results are fed back into the agent's context
for a second reasoning pass.

Tools are grouped by agent role so each agent only sees relevant tools.
"""
from __future__ import annotations

import io
import json
import logging
import math

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════

def compute_group_disparity(csv_text: str, attribute: str, target: str) -> str:
    """Compute selection-rate disparity across groups for a protected attribute."""
    try:
        df = pd.read_csv(io.StringIO(csv_text))
        if attribute not in df.columns or target not in df.columns:
            return json.dumps({"error": f"Column '{attribute}' or '{target}' not found."})

        # Infer positive label
        vals = df[target].dropna().unique()
        pos = 1 if set(vals) <= {0, 1} else vals[0]
        for kw in ("yes", "true", "1", "hired", "approved", "accepted", "positive"):
            matches = [v for v in vals if str(v).strip().lower() == kw]
            if matches:
                pos = matches[0]
                break

        rates = {}
        for grp, sub in df.groupby(attribute):
            rates[str(grp)] = round(float((sub[target] == pos).mean()), 4)

        max_r = max(rates.values()) if rates else 0
        min_r = min(rates.values()) if rates else 0
        di_ratio = round(min_r / max_r, 4) if max_r > 0 else 0.0

        return json.dumps({
            "attribute": attribute,
            "group_selection_rates": rates,
            "disparate_impact_ratio": di_ratio,
            "four_fifths_rule_violated": di_ratio < 0.8,
            "disparity_gap": round(max_r - min_r, 4),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def analyze_feature_correlation(csv_text: str, feature: str, target: str) -> str:
    """Analyze correlation between a feature and the target variable."""
    try:
        df = pd.read_csv(io.StringIO(csv_text))
        if feature not in df.columns or target not in df.columns:
            return json.dumps({"error": f"Column '{feature}' or '{target}' not found."})

        # Try numeric correlation
        feat_numeric = pd.to_numeric(df[feature], errors="coerce")
        tgt_numeric = pd.to_numeric(df[target], errors="coerce")

        if feat_numeric.notna().sum() > 10 and tgt_numeric.notna().sum() > 10:
            corr = float(feat_numeric.corr(tgt_numeric))
            return json.dumps({
                "feature": feature,
                "target": target,
                "pearson_correlation": round(corr, 4),
                "strength": "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak",
                "direction": "positive" if corr > 0 else "negative",
                "is_potential_proxy": abs(corr) > 0.3,
            })
        else:
            # Categorical — compute contingency
            cross = pd.crosstab(df[feature], df[target], normalize="index")
            return json.dumps({
                "feature": feature,
                "target": target,
                "type": "categorical",
                "outcome_rates_by_group": cross.to_dict(),
            })
    except Exception as e:
        return json.dumps({"error": str(e)})


def check_proxy_variable(csv_text: str, feature: str, protected_attribute: str) -> str:
    """Check if a feature acts as a proxy for a protected attribute."""
    try:
        df = pd.read_csv(io.StringIO(csv_text))
        if feature not in df.columns or protected_attribute not in df.columns:
            return json.dumps({"error": "Column not found."})

        f_num = pd.to_numeric(df[feature], errors="coerce")
        p_num = pd.to_numeric(df[protected_attribute], errors="coerce")

        if f_num.notna().sum() > 10 and p_num.notna().sum() > 10:
            corr = float(f_num.corr(p_num))
            is_proxy = abs(corr) > 0.3
        else:
            # Categorical: Cramér's V
            try:
                contingency = pd.crosstab(df[feature], df[protected_attribute])
                from scipy.stats import chi2_contingency
                chi2, _, _, _ = chi2_contingency(contingency)
                n = contingency.sum().sum()
                min_dim = min(contingency.shape) - 1
                cramers_v = math.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                corr = round(cramers_v, 4)
                is_proxy = cramers_v > 0.3
            except Exception:
                corr = None
                is_proxy = False

        return json.dumps({
            "feature": feature,
            "protected_attribute": protected_attribute,
            "correlation_strength": round(corr, 4) if corr is not None else "N/A",
            "is_likely_proxy": is_proxy,
            "recommendation": (
                f"ALERT: '{feature}' is strongly correlated with '{protected_attribute}' — "
                "likely acts as a proxy variable. Consider removing or monitoring."
                if is_proxy else
                f"'{feature}' shows weak correlation with '{protected_attribute}' — "
                "unlikely to be a proxy variable."
            ),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def search_legal_precedent(topic: str, jurisdiction: str = "global") -> str:
    """Search for relevant legal precedents and regulations related to AI bias."""
    # Knowledge base of key legal precedents and regulations
    precedents = {
        "hiring": [
            {
                "case": "Griggs v. Duke Power Co. (1971)",
                "jurisdiction": "US",
                "ruling": "Employment practices must be job-related. Disparate impact violates Title VII even without discriminatory intent.",
                "relevance": "Establishes that neutral-appearing criteria can be discriminatory if they disproportionately affect protected groups."
            },
            {
                "case": "EEOC v. Target Corp (2015)",
                "jurisdiction": "US",
                "ruling": "Pre-employment assessments with adverse impact violate Title VII unless validated.",
                "relevance": "AI screening tools must be validated for job-relatedness."
            },
            {
                "regulation": "EU AI Act Article 6 (High-Risk AI)",
                "jurisdiction": "EU",
                "requirement": "AI systems used in employment decisions are classified as high-risk, requiring conformity assessments, human oversight, and bias auditing.",
            },
        ],
        "lending": [
            {
                "case": "Fair Housing Act / ECOA",
                "jurisdiction": "US",
                "ruling": "Prohibits discrimination in lending based on race, religion, national origin, sex, disability, familial status.",
                "relevance": "Proxy discrimination through zip codes or education level can violate fair lending laws."
            },
            {
                "regulation": "India RBI Fair Lending Guidelines",
                "jurisdiction": "India",
                "requirement": "Banks must ensure non-discriminatory lending practices. Algorithmic decisions must be explainable.",
            },
        ],
        "general": [
            {
                "regulation": "India DPDP Act 2023, Section 4",
                "jurisdiction": "India",
                "requirement": "Processing of personal data must be for a lawful purpose. Sensitive personal data includes caste, religion, political belief.",
            },
            {
                "regulation": "EU GDPR Article 22",
                "jurisdiction": "EU",
                "requirement": "Right not to be subject to purely automated decision-making with legal or significant effects. Requires human oversight.",
            },
            {
                "regulation": "Article 14-15, Indian Constitution",
                "jurisdiction": "India",
                "requirement": "Equality before law (Art 14). Prohibition of discrimination on grounds of religion, race, caste, sex, place of birth (Art 15).",
            },
        ],
    }

    # Find relevant precedents
    results = []
    topic_lower = topic.lower()
    for category, cases in precedents.items():
        if category in topic_lower or category == "general":
            for case in cases:
                if jurisdiction.lower() in ("global", "all") or jurisdiction.lower() in case.get("jurisdiction", "").lower():
                    results.append(case)

    return json.dumps({
        "topic": topic,
        "jurisdiction": jurisdiction,
        "precedents_found": len(results),
        "precedents": results[:5],
    })


def assess_intersectional_bias(csv_text: str, attributes: list, target: str) -> str:
    """Assess intersectional bias across multiple protected attributes combined."""
    try:
        df = pd.read_csv(io.StringIO(csv_text))
        valid_attrs = [a for a in attributes if a in df.columns]
        if len(valid_attrs) < 2 or target not in df.columns:
            return json.dumps({"error": "Need at least 2 valid protected attributes and target column."})

        # Infer positive label
        vals = df[target].dropna().unique()
        pos = 1 if set(vals) <= {0, 1} else vals[0]

        # Create intersection groups
        df["_intersection"] = df[valid_attrs].astype(str).agg(" × ".join, axis=1)
        rates = {}
        for grp, sub in df.groupby("_intersection"):
            if len(sub) >= 5:  # minimum sample size
                rates[grp] = {"rate": round(float((sub[target] == pos).mean()), 4), "n": len(sub)}

        if len(rates) < 2:
            return json.dumps({"error": "Not enough intersectional groups with sufficient sample size."})

        all_rates = [v["rate"] for v in rates.values()]
        return json.dumps({
            "attributes": valid_attrs,
            "intersectional_groups": rates,
            "max_rate": max(all_rates),
            "min_rate": min(all_rates),
            "gap": round(max(all_rates) - min(all_rates), 4),
            "most_disadvantaged": min(rates, key=lambda k: rates[k]["rate"]),
            "most_advantaged": max(rates, key=lambda k: rates[k]["rate"]),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def compute_counterfactual_fairness(csv_text: str, protected_attribute: str, target: str) -> str:
    """Test counterfactual fairness: what happens to outcomes if we flip the protected attribute?"""
    try:
        df = pd.read_csv(io.StringIO(csv_text))
        if protected_attribute not in df.columns or target not in df.columns:
            return json.dumps({"error": "Column not found."})

        groups = df[protected_attribute].unique()
        if len(groups) != 2:
            return json.dumps({
                "note": f"Counterfactual test works best with binary attributes. Found {len(groups)} groups.",
                "groups": [str(g) for g in groups],
            })

        vals = df[target].dropna().unique()
        pos = 1 if set(vals) <= {0, 1} else vals[0]

        g1, g2 = groups[0], groups[1]
        rate1 = float((df[df[protected_attribute] == g1][target] == pos).mean())
        rate2 = float((df[df[protected_attribute] == g2][target] == pos).mean())

        return json.dumps({
            "protected_attribute": protected_attribute,
            "group_1": {"name": str(g1), "positive_rate": round(rate1, 4)},
            "group_2": {"name": str(g2), "positive_rate": round(rate2, 4)},
            "counterfactual_gap": round(abs(rate1 - rate2), 4),
            "is_counterfactually_fair": abs(rate1 - rate2) < 0.05,
            "interpretation": (
                "Outcomes are approximately equal regardless of group membership."
                if abs(rate1 - rate2) < 0.05 else
                f"Significant counterfactual gap detected: switching from '{g1}' to '{g2}' "
                f"changes the positive outcome rate by {abs(rate1 - rate2):.1%}."
            ),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════
# TOOL REGISTRY — maps tool name → function
# ═══════════════════════════════════════════════════════════════════

TOOL_REGISTRY = {
    "compute_group_disparity": compute_group_disparity,
    "analyze_feature_correlation": analyze_feature_correlation,
    "check_proxy_variable": check_proxy_variable,
    "search_legal_precedent": search_legal_precedent,
    "assess_intersectional_bias": assess_intersectional_bias,
    "compute_counterfactual_fairness": compute_counterfactual_fairness,
}


# ═══════════════════════════════════════════════════════════════════
# GEMINI FUNCTION DECLARATIONS — per agent role
# ═══════════════════════════════════════════════════════════════════

from google.genai import types

_STAT_TOOLS = [
    types.FunctionDeclaration(
        name="compute_group_disparity",
        description="Compute selection-rate disparity across groups for a protected attribute. Returns group rates, disparate impact ratio, and 4/5ths rule check.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "attribute": types.Schema(type="STRING", description="Protected attribute column name"),
                "target": types.Schema(type="STRING", description="Target/outcome column name"),
            },
            required=["attribute", "target"],
        ),
    ),
    types.FunctionDeclaration(
        name="analyze_feature_correlation",
        description="Analyze the correlation between a feature column and the target variable to detect potential bias pathways.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "feature": types.Schema(type="STRING", description="Feature column to analyze"),
                "target": types.Schema(type="STRING", description="Target/outcome column name"),
            },
            required=["feature", "target"],
        ),
    ),
]

_FAIRNESS_TOOLS = [
    types.FunctionDeclaration(
        name="compute_group_disparity",
        description="Compute selection-rate disparity across groups for a protected attribute. Returns group rates, disparate impact ratio, and 4/5ths rule check.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "attribute": types.Schema(type="STRING", description="Protected attribute column name"),
                "target": types.Schema(type="STRING", description="Target/outcome column name"),
            },
            required=["attribute", "target"],
        ),
    ),
    types.FunctionDeclaration(
        name="compute_counterfactual_fairness",
        description="Test counterfactual fairness: compare outcomes if the protected attribute value were flipped.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "protected_attribute": types.Schema(type="STRING", description="Protected attribute column"),
                "target": types.Schema(type="STRING", description="Target/outcome column"),
            },
            required=["protected_attribute", "target"],
        ),
    ),
]

_DOMAIN_TOOLS = [
    types.FunctionDeclaration(
        name="check_proxy_variable",
        description="Check if a feature acts as a proxy for a protected attribute by computing their statistical correlation.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "feature": types.Schema(type="STRING", description="Feature column to check"),
                "protected_attribute": types.Schema(type="STRING", description="Protected attribute column"),
            },
            required=["feature", "protected_attribute"],
        ),
    ),
    types.FunctionDeclaration(
        name="analyze_feature_correlation",
        description="Analyze the correlation between a feature and the target to assess relevance.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "feature": types.Schema(type="STRING", description="Feature column to analyze"),
                "target": types.Schema(type="STRING", description="Target/outcome column"),
            },
            required=["feature", "target"],
        ),
    ),
]

_ADVERSARY_TOOLS = [
    types.FunctionDeclaration(
        name="compute_counterfactual_fairness",
        description="Test counterfactual fairness to check if the disparity claim holds under a counterfactual test.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "protected_attribute": types.Schema(type="STRING", description="Protected attribute column"),
                "target": types.Schema(type="STRING", description="Target/outcome column"),
            },
            required=["protected_attribute", "target"],
        ),
    ),
    types.FunctionDeclaration(
        name="analyze_feature_correlation",
        description="Analyze feature correlation to find confounding variables that might explain the disparity.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "feature": types.Schema(type="STRING", description="Feature column"),
                "target": types.Schema(type="STRING", description="Target/outcome column"),
            },
            required=["feature", "target"],
        ),
    ),
]

_LEGAL_TOOLS = [
    types.FunctionDeclaration(
        name="search_legal_precedent",
        description="Search for relevant legal precedents, regulations, and case law related to the AI bias topic.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "topic": types.Schema(type="STRING", description="Topic to search (e.g. 'hiring discrimination', 'lending bias')"),
                "jurisdiction": types.Schema(type="STRING", description="Jurisdiction filter (e.g. 'US', 'EU', 'India', 'global')"),
            },
            required=["topic"],
        ),
    ),
    types.FunctionDeclaration(
        name="assess_intersectional_bias",
        description="Assess intersectional bias across multiple protected attributes combined.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "attributes": types.Schema(
                    type="ARRAY",
                    items=types.Schema(type="STRING"),
                    description="List of protected attribute column names to intersect",
                ),
                "target": types.Schema(type="STRING", description="Target/outcome column"),
            },
            required=["attributes", "target"],
        ),
    ),
]

# Map agent IDs to their tool sets
AGENT_TOOLS = {
    "data_statistician": types.Tool(function_declarations=_STAT_TOOLS),
    "fairness_auditor":  types.Tool(function_declarations=_FAIRNESS_TOOLS),
    "domain_expert":     types.Tool(function_declarations=_DOMAIN_TOOLS),
    "bias_adversary":    types.Tool(function_declarations=_ADVERSARY_TOOLS),
    "ethical_reviewer":  types.Tool(function_declarations=_LEGAL_TOOLS),
}
