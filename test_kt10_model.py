"""Quick check of both exported models via the standalone modules.

Loads model_export/kt10_xgb.json through kt10_model and model_export/kt10_rf.joblib
through kt10_model_rf, then runs predictValue on a spread of rows from the dataset
and prints real vs. predicted KT10valueMax for both, side by side.

Run:  python test_kt10_model.py
"""

import numpy as np
import pandas as pd

import kt10_model
import kt10_model_rf

DATASET = "samples/dataset.xlsx"

# (label, module, weights file) — both modules expose the same init_model/predictValue API.
MODELS = [
    ("XGBoost", kt10_model, "model_export/kt10_xgb.json"),
    ("RandomForest", kt10_model_rf, "model_export/kt10_rf.joblib"),
]

# Feature order expected by predictValue (same as notebook feature_cols).
FEATURES = list(kt10_model.FEATURE_ORDER)
assert FEATURES == list(kt10_model_rf.FEATURE_ORDER), "modules disagree on feature order"


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
    for label, module, path in MODELS:
        if not module.init_model(path):
            raise SystemExit(
                f"Could not load {label} model from {path!r}. "
                f"Run the export cell in system_technologies.ipynb first."
            )
        print(f"Loaded {label:<13} from {path}")
    print()

    df = load_dataset(DATASET)
    sample = pick_sample_rows(df, n=100)

    header = f"{'real':>10}" + "".join(
        f"{label[:9] + '_pred':>16}{label[:9] + '_diff':>16}" for label, _, _ in MODELS
    )
    print(header)
    print("-" * len(header))

    abs_diffs = {label: [] for label, _, _ in MODELS}
    for _, row in sample.iterrows():
        features = [row[c] for c in FEATURES]      # in the required order
        real = float(row["KT10valueMax"])
        line = f"{real:10.3f}"
        for label, module, _ in MODELS:
            pred = module.predictValue(*features)  # 7 separate args
            diff = pred - real
            abs_diffs[label].append(abs(diff))
            line += f"{pred:16.3f}{diff:16.3f}"
        print(line)

    print("-" * len(header))
    print(f"Sample size : {len(sample)}\n")
    print(f"{'model':<15}{'MAE':>10}{'max_abs_diff':>15}")
    for label, _, _ in MODELS:
        d = np.array(abs_diffs[label])
        print(f"{label:<15}{d.mean():10.3f}{d.max():15.3f}")


if __name__ == "__main__":
    main()
