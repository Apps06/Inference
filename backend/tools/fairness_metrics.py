"""
Fairness metrics — wraps Fairlearn to compute key bias metrics
across protected groups. Returns a flat dict for LLM context injection.

Fixes over v1:
- Positive label inference now uses a keyword allowlist ('yes', 'approved',
  'hired', etc.) before falling back to the majority class, so string-label
  targets like "Approved"/"Rejected" are handled correctly.
- Pre-model audit mode no longer reports per-group accuracy (it would always
  be 1.0 because y_pred == y_binary in that mode, making the metric useless).
- Equalized Odds in pre-model mode is now skipped for the same reason.
"""
from __future__ import annotations

import io
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
        MetricFrame,
    )
    from sklearn.metrics import accuracy_score
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False


# ── Positive-label heuristic ────────────────────────────────────────────────
# Ordered by preference. The first matching keyword wins.
_POSITIVE_KEYWORDS = (
    "yes", "true", "1", "hired", "approved", "accepted",
    "passed", "qualified", "selected", "positive", "granted",
    "admitted", "success", "good", "high",
)


def _infer_positive_label(series: pd.Series):
    """
    Return the value in `series` that represents the positive/outcome class.

    Priority:
      1. Binary numeric {0, 1} → 1
      2. Known positive-keyword strings (case-insensitive)
      3. Majority class (most common value)
    """
    unique_vals = series.dropna().unique()
    if len(unique_vals) == 0:
        return 1

    # Numeric binary
    if set(unique_vals) <= {0, 1}:
        return 1
    try:
        if set(unique_vals) <= {0.0, 1.0}:
            return 1.0
    except TypeError:
        pass

    # Keyword match
    str_map = {str(v).strip().lower(): v for v in unique_vals}
    for kw in _POSITIVE_KEYWORDS:
        if kw in str_map:
            return str_map[kw]

    # Majority class as last resort
    return series.value_counts().idxmax()


# ── Public API ──────────────────────────────────────────────────────────────

def compute_fairness_metrics(
    csv_text: str,
    protected_attributes: list[str],
    target_column: str,
    prediction_column: str | None = None,
) -> dict:
    """
    Compute fairness metrics on a dataset.

    If a prediction_column is provided, uses model predictions vs actual labels
    (model audit mode). Otherwise treats the target column as the decision
    (pre-model / dataset audit mode).

    Returns a flat dict with metric names and numeric values.
    """
    results: dict = {}

    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        return {"error": f"Could not parse CSV: {e}"}

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    y_true_raw = df[target_column]
    pos_label = _infer_positive_label(y_true_raw)
    y_binary = (y_true_raw == pos_label).astype(int)

    # Determine prediction series
    is_model_audit = prediction_column and prediction_column in df.columns
    if is_model_audit:
        y_pred_raw = df[prediction_column]
        y_pred = (y_pred_raw == pos_label).astype(int)
        results["mode"] = "model_prediction_audit"
    else:
        y_pred = y_binary.copy()
        results["mode"] = "pre_model_decision_audit"

    results["positive_label"] = str(pos_label)
    results["total_rows"] = int(len(df))
    results["overall_positive_rate"] = float(round(y_binary.mean(), 4))

    for attr in protected_attributes:
        if attr not in df.columns:
            results[f"{attr}_error"] = "Column not found"
            continue

        sensitive = df[attr].fillna("Unknown").astype(str)
        groups = sensitive.unique()

        # ── Group-level selection rates ──────────────────────────────
        for grp in groups:
            mask = sensitive == grp
            rate = float(round(y_pred[mask].mean(), 4))
            results[f"{attr}[{grp}]_selection_rate"] = rate

        if not FAIRLEARN_AVAILABLE:
            results["fairlearn_status"] = "not_installed"
            continue

        # ── Demographic Parity Difference ────────────────────────────
        try:
            dp_diff = demographic_parity_difference(
                y_true=y_binary,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )
            results[f"{attr}_demographic_parity_difference"] = round(float(dp_diff), 4)
            results[f"{attr}_disparate_impact_violated"] = bool(abs(dp_diff) > 0.1)
        except Exception as e:
            results[f"{attr}_demographic_parity_difference"] = f"error: {e}"

        # ── Equalized Odds Difference ────────────────────────────────
        # Only meaningful in model audit mode — in pre-model mode y_pred == y_binary
        # so EOD would always be 0, which is misleading.
        if is_model_audit:
            try:
                eo_diff = equalized_odds_difference(
                    y_true=y_binary,
                    y_pred=y_pred,
                    sensitive_features=sensitive,
                )
                results[f"{attr}_equalized_odds_difference"] = round(float(eo_diff), 4)
            except Exception as e:
                results[f"{attr}_equalized_odds_difference"] = f"error: {e}"
        else:
            results[f"{attr}_equalized_odds_difference"] = "N/A (pre-model audit)"

        # ── Per-group accuracy ───────────────────────────────────────
        # Only emit accuracy in model audit mode. Pre-model: y_pred == y_binary
        # so accuracy is identically 1.0 for every group, which is meaningless.
        if is_model_audit:
            try:
                mf = MetricFrame(
                    metrics=accuracy_score,
                    y_true=y_binary,
                    y_pred=y_pred,
                    sensitive_features=sensitive,
                )
                for grp, acc in mf.by_group.items():
                    results[f"{attr}[{grp}]_accuracy"] = round(float(acc), 4)
                results[f"{attr}_accuracy_disparity"] = round(
                    float(mf.difference(method="between_groups")), 4
                )
            except Exception as e:
                results[f"{attr}_accuracy_disparity"] = f"error: {e}"

        # ── Disparate Impact Ratio (80% / 4-fifths rule) ─────────────
        try:
            rates: dict[str, float] = {}
            for grp in groups:
                mask = sensitive == grp
                n = int(mask.sum())
                if n == 0:
                    continue
                rates[grp] = float(y_pred[mask].mean())

            if len(rates) >= 2:
                max_rate = max(rates.values())
                min_rate = min(rates.values())
                if max_rate > 0:
                    di_ratio = min_rate / max_rate
                    results[f"{attr}_disparate_impact_ratio"] = round(di_ratio, 4)
                    results[f"{attr}_4_5ths_rule_violated"] = bool(di_ratio < 0.8)
                else:
                    results[f"{attr}_disparate_impact_ratio"] = 0.0
                    results[f"{attr}_4_5ths_rule_violated"] = True
        except Exception as e:
            results[f"{attr}_disparate_impact_ratio"] = f"error: {e}"

    return results
