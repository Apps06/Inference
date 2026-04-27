"""
Dataset profiler — generates a human-readable statistical summary of the
uploaded CSV. Used to give agents rich context about the data without
sharing the full file.

Fix over v1: positive label is now inferred using the same keyword
allowlist as fairness_metrics.py, so "No"/"Yes" and "Rejected"/"Approved"
datasets are handled correctly.
"""
from __future__ import annotations

import io
import pandas as pd
import numpy as np


# ── Shared positive-label heuristic (mirrors fairness_metrics._infer_positive_label) ─
_POSITIVE_KEYWORDS = (
    "yes", "true", "1", "hired", "approved", "accepted",
    "passed", "qualified", "selected", "positive", "granted",
    "admitted", "success", "good", "high",
)


def _infer_positive_label(series: pd.Series):
    unique_vals = series.dropna().unique()
    if len(unique_vals) == 0:
        return 1
    if set(unique_vals) <= {0, 1}:
        return 1
    try:
        if set(unique_vals) <= {0.0, 1.0}:
            return 1.0
    except TypeError:
        pass
    str_map = {str(v).strip().lower(): v for v in unique_vals}
    for kw in _POSITIVE_KEYWORDS:
        if kw in str_map:
            return str_map[kw]
    return series.value_counts().idxmax()


def profile_dataset(
    csv_text: str,
    protected_attributes: list[str],
    target_column: str,
) -> str:
    """
    Given raw CSV text, return a structured text summary for LLM context.
    """
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        return f"Error reading dataset: {e}"

    lines: list[str] = []
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append(f"Columns: {list(df.columns)}")

    missing = df.isnull().sum()
    if missing.any():
        lines.append(f"Missing values: {missing[missing > 0].to_dict()}")
    else:
        lines.append("Missing values: none")

    # Target column distribution
    if target_column in df.columns:
        pos_label = _infer_positive_label(df[target_column])
        lines.append(f"\nTarget column '{target_column}' (positive label: {pos_label!r}):")
        vc = df[target_column].value_counts(normalize=True)
        for val, pct in vc.items():
            lines.append(f"  {val}: {pct:.1%}")
    else:
        lines.append(f"\nTarget column '{target_column}' not found in dataset.")

    # Protected attribute distributions + outcome rates
    for attr in protected_attributes:
        if attr not in df.columns:
            lines.append(f"\nProtected attribute '{attr}' not found in columns.")
            continue

        lines.append(f"\nProtected attribute: '{attr}'")
        group_counts = df[attr].value_counts()
        for grp, cnt in group_counts.items():
            pct = cnt / len(df)
            lines.append(f"  {grp}: {cnt} ({pct:.1%})")

        if target_column in df.columns:
            pos_label = _infer_positive_label(df[target_column])
            lines.append(f"  Outcome rates by '{attr}' (positive = {pos_label!r}):")
            grouped = df.groupby(attr)[target_column].apply(
                lambda x: (x == pos_label).mean()  # noqa: B023
            )
            for grp, rate in grouped.items():
                lines.append(f"    {grp}: {rate:.1%} positive outcome rate")

    # Numeric column summaries (top 5, excluding protected attributes)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    relevant_numeric = [c for c in numeric_cols if c not in protected_attributes][:5]
    if relevant_numeric:
        lines.append("\nNumeric feature summaries (top 5):")
        for col in relevant_numeric:
            mn, mx, med = df[col].min(), df[col].max(), df[col].median()
            lines.append(f"  {col}: min={mn:.2f}, max={mx:.2f}, median={med:.2f}")

    return "\n".join(lines)
