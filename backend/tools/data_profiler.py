"""
Dataset profiler — generates a human-readable statistical summary of the uploaded CSV.
Used to give agents rich context about the data without sharing the full file.
"""
import io
import pandas as pd
import numpy as np


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

    lines = []
    lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    lines.append(f"Columns: {list(df.columns)}")
    lines.append(f"Missing values per column: {df.isnull().sum().to_dict()}")

    # Target column distribution
    if target_column in df.columns:
        lines.append(f"\nTarget column '{target_column}' distribution:")
        vc = df[target_column].value_counts(normalize=True)
        for val, pct in vc.items():
            lines.append(f"  {val}: {pct:.1%}")

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
            lines.append(f"  Outcome rates by '{attr}':")
            # Handle binary (0/1) or categorical targets
            target_vals = df[target_column].dropna().unique()
            positive_label = 1 if 1 in target_vals else target_vals[0]
            grouped = df.groupby(attr)[target_column].apply(
                lambda x: (x == positive_label).mean()
            )
            for grp, rate in grouped.items():
                lines.append(f"    {grp}: {rate:.1%} positive outcome rate")

    # Numeric column stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    relevant_numeric = [c for c in numeric_cols if c not in protected_attributes][:5]
    if relevant_numeric:
        lines.append(f"\nNumeric feature summaries (top 5):")
        for col in relevant_numeric:
            mn, mx, med = df[col].min(), df[col].max(), df[col].median()
            lines.append(f"  {col}: min={mn:.2f}, max={mx:.2f}, median={med:.2f}")

    return "\n".join(lines)
