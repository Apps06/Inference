"""
Fairness metrics — wraps Fairlearn to compute key bias metrics
across protected groups. Returns a flat dict for LLM context injection.
"""
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
    from sklearn.metrics import accuracy_score, selection_rate
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False


def compute_fairness_metrics(
    csv_text: str,
    protected_attributes: list[str],
    target_column: str,
    prediction_column: str | None = None,
) -> dict:
    """
    Compute fairness metrics on a dataset.

    If a prediction_column is provided, uses model predictions vs. actual labels.
    Otherwise treats the target column as the "decision" (useful for pre-model auditing).

    Returns a flat dict with metric names and values.
    """
    results = {}

    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        return {"error": f"Could not parse CSV: {e}"}

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    # Determine labels and predictions
    y_true = df[target_column]

    # Convert to binary if needed
    unique_vals = y_true.dropna().unique()
    if len(unique_vals) == 2:
        pos_label = sorted(unique_vals)[-1]  # larger value = positive
        y_binary = (y_true == pos_label).astype(int)
    elif set(unique_vals).issubset({0, 1}):
        y_binary = y_true.fillna(0).astype(int)
        pos_label = 1
    else:
        results["note"] = "Target column has > 2 unique values; metrics computed on largest value as positive."
        pos_label = y_true.value_counts().index[0]
        y_binary = (y_true == pos_label).astype(int)

    y_pred = y_binary  # treat target as both truth and "decision" for pre-model audit

    if prediction_column and prediction_column in df.columns:
        y_pred_col = df[prediction_column]
        y_pred = (y_pred_col == pos_label).astype(int)
        results["mode"] = "model_prediction_audit"
    else:
        results["mode"] = "pre_model_decision_audit"

    results["total_rows"] = len(df)
    results["overall_positive_rate"] = float(y_binary.mean().round(4))

    for attr in protected_attributes:
        if attr not in df.columns:
            results[f"{attr}_error"] = "Column not found"
            continue

        sensitive = df[attr].fillna("Unknown")

        # ── Group-level selection rates ──────────────────────────────────
        groups = sensitive.unique()
        for grp in groups:
            mask = sensitive == grp
            rate = float(y_pred[mask].mean().round(4))
            results[f"{attr}[{grp}]_selection_rate"] = rate

        if not FAIRLEARN_AVAILABLE:
            results["fairlearn_status"] = "not_installed"
            continue

        # ── Demographic Parity Difference ────────────────────────────────
        try:
            dp_diff = demographic_parity_difference(
                y_true=y_binary,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )
            results[f"{attr}_demographic_parity_difference"] = round(float(dp_diff), 4)
            results[f"{attr}_disparate_impact_violated"] = abs(dp_diff) > 0.1
        except Exception as e:
            results[f"{attr}_demographic_parity_difference"] = f"error: {e}"

        # ── Equalized Odds Difference ────────────────────────────────────
        try:
            eo_diff = equalized_odds_difference(
                y_true=y_binary,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )
            results[f"{attr}_equalized_odds_difference"] = round(float(eo_diff), 4)
        except Exception as e:
            results[f"{attr}_equalized_odds_difference"] = f"error: {e}"

        # ── Per-group accuracy ───────────────────────────────────────────
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

        # ── Disparate Impact Ratio (80% rule) ────────────────────────────
        try:
            rates = {}
            for grp in groups:
                mask = sensitive == grp
                rates[grp] = float(y_pred[mask].mean())
            if rates:
                max_rate = max(rates.values())
                min_rate = min(rates.values())
                if max_rate > 0:
                    di_ratio = min_rate / max_rate
                    results[f"{attr}_disparate_impact_ratio"] = round(di_ratio, 4)
                    results[f"{attr}_4_5ths_rule_violated"] = di_ratio < 0.8
        except Exception as e:
            results[f"{attr}_disparate_impact_ratio"] = f"error: {e}"

    return results
