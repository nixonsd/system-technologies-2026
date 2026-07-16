# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A single-notebook ML regression study (`system_technologies.ipynb`, Python 3.10) that predicts
`KT10valueMax` from industrial sensor measurements, then uses **learning-curve scaling-law
analysis** to estimate how much more data is needed to hit a target RMSE.

The data is `samples/dataset.xlsx` (~4161 rows). Raw columns: `Label`, `DateTime`, `SensorId`,
`Volume`, `Area`, `MaxHeight`, `CenterDiviation`, `UsedCoilArea`, `SensorTemp`, `SensorValueRAW`,
`KT10valueCenterDiviation`, `KT10valueMax` (target). Note the source-data spelling `CenterDiviation`
(not "Deviation") — match it exactly when referencing columns. (`plan.xlsx` in the repo root is a
side planning sheet, not model input.)

## Environment

No `requirements.txt` / lockfile exists. The notebook depends on: `pandas`, `numpy`, `matplotlib`,
`scikit-learn` (1.7), `xgboost` (3.1), `optuna` (4.8), `scipy`, plus `openpyxl` to read the `.xlsx`.
Run it through the Jupyter kernel `Python 3 (ipykernel)`.

## Notebook structure & flow

Cells are meant to be run top to bottom; later cells depend on variables defined earlier
(`df`, `numeric_cols`, `pipeline`, `X`/`y`/`groups`, `train_sizes_abs`, `val_rmse`).

1. **Cell 0** — all imports.
2. **Cell 1** — load + clean: strips column whitespace, converts European decimal commas to dots,
   coerces numerics, parses `DateTime` (`%Y-%m-%d_%H-%M-%S`). Engineers `AspectRatio`, `Density`,
   `NormDeviation`. Defines `numeric_cols` (the raw numerics plus `KT10valueCenterDiviation` and the
   three engineered features). **These engineered features are for EDA only — the model in Cell 7
   does not use them** (see Working conventions).
3. **Cells 2–5** — EDA: `describe`, correlation against the target, scatter matrix, log-feature
   histograms.
4. **Cell 6** — `LogFeaturesTransformer` (sklearn `BaseEstimator`/`TransformerMixin`) for
   `log1p`-transforming skewed columns inside a pipeline.
5. **Cell 7** — the model: a `Pipeline` of `SimpleImputer` → `TransformedTargetRegressor`
   (`log1p`/`expm1` on the target) wrapping `XGBRegressor` with hard-coded tuned params.
   `feature_cols` is the **8 raw columns** (`SensorId`, `Volume`, `Area`, `MaxHeight`,
   `CenterDiviation`, `UsedCoilArea`, `SensorTemp`, `SensorValueRAW`) — not the engineered ones.
   Evaluated with **`GroupKFold(n_splits=10)` grouped on `Label`** (prevents leakage across repeated
   parts) via `cross_validate` (R², RMSE).
6. **Cell 8** — Optuna hyperparameter search (`xgb_search_space`, `objective`); params are set on
   the pipeline as `model__regressor__<param>`. The tuned output from here is what gets pasted into
   Cell 7.
7. **Cells 9–10** — **NLS scaling law**: fits `rmse(n) = a + b·(n/n_ref)^(-c)` to learning-curve
   RMSE via `scipy.optimize.curve_fit` (weighted by per-size std). Reports `a` as the asymptotic
   RMSE, computes the `N` needed for `rmse_target` (=23), the point of diminishing returns, and
   Monte-Carlo confidence bands from the parameter covariance.
8. **Cell 11** — **Gaussian-process variant**: same NLS fit, then a `GaussianProcessRegressor`
   (ConstantKernel·RBF + WhiteKernel) models residuals in `log(n)` space for a probabilistic
   extrapolation.
9. **Cells 12–13** — **inference-latency benchmark**: clones + fits `pipeline`, pulls the bare fitted
   `XGBRegressor` (`named_steps["model"].regressor_`, `n_jobs=1`), and times single-row `predict` over
   100k repeats — the "raw model, no DataFrame/imputer overhead" path. Remember the raw output is in
   `log1p` space, so it applies `np.expm1` to recover real `KT10valueMax`.

## Working conventions

- **Always group by `Label` in any CV/split.** Rows sharing a `Label` are the same physical
  sample; ungrouped CV leaks and inflates scores.
- The target is modeled in log space (`TransformedTargetRegressor` with `log1p`/`expm1`); keep
  metrics in the original units.
- When you re-run Optuna (Cell 8), copy the resulting best params into the `XGBRegressor(...)`
  constructor in Cell 7 — the two are linked only by manual copy, not programmatically.
- `numeric_cols` and `feature_cols` are filtered with `[c for c in ... if c in df.columns]`, so
  renaming/removing a column degrades silently rather than erroring — verify columns exist.
