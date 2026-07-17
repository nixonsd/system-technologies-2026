"""Quick check of the exported model via the standalone module.

Loads model_export/kt10_xgb.json through kt10_model.init_model, then runs
predictValue on a spread of rows from the dataset and prints the real vs.
predicted KT10valueMax along with the difference.

Run:  python test_kt10_model.py
"""

import numpy as np
import pandas as pd

import kt10_model

DATASET = "samples/dataset.xlsx"
MODEL = "model_export/kt10_xgb.json"

# Feature order expected by predictValue (same as notebook feature_cols).
FEATURES = list(kt10_model.FEATURE_ORDER)


def load_dataset(path):
    """Load + clean the dataset the same way as notebook Cell 1."""
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all").reset_index(drop=True)

    numeric_cols = FEATURES + ["KT10valueMax"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    # European decimal commas -> dots, then coerce to numbers.
    df[numeric_cols] = df[numeric_cols].astype(str).map(
        lambda x: x.replace(",", ".") if isinstance(x, str) else x
    )
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    return df


def pick_sample_rows(df, n=8):
    """Pick n rows evenly spread across the KT10valueMax range (low -> high)."""
    order = df["KT10valueMax"].sort_values().index.to_numpy()
    idx = np.linspace(0, len(order) - 1, n).round().astype(int)
    return df.loc[order[idx]]


def main():
    if not kt10_model.init_model(MODEL):
        raise SystemExit(
            f"Could not load model from {MODEL!r}. "
            f"Run the export cell in system_technologies.ipynb first."
        )
    print(f"Loaded model: {MODEL}\n")

    df = load_dataset(DATASET)
    sample = pick_sample_rows(df, n=100)

    print(f"{'real':>10} {'predicted':>12} {'diff':>10} {'abs_diff':>10} {'err_%':>8}")
    print("-" * 54)

    abs_diffs = []
    for _, row in sample.iterrows():
        features = [row[c] for c in FEATURES]          # in the required order
        real = float(row["KT10valueMax"])
        pred = kt10_model.predictValue(*features)      # 8 separate args
        diff = pred - real
        abs_diff = abs(diff)
        err_pct = (abs_diff / real * 100.0) if real != 0 else float("nan")
        abs_diffs.append(abs_diff)
        print(f"{real:10.3f} {pred:12.3f} {diff:10.3f} {abs_diff:10.3f} {err_pct:7.1f}%")

    abs_diffs = np.array(abs_diffs)
    print("-" * 54)
    print(f"Sample size : {len(abs_diffs)}")
    print(f"MAE         : {abs_diffs.mean():.3f}")
    print(f"Max abs diff: {abs_diffs.max():.3f}")


if __name__ == "__main__":
    main()
