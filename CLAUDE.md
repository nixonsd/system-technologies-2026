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
`scikit-learn` (1.7), `xgboost` (3.1), `optuna` (4.8), `scipy`, `joblib` (to export the
RandomForest), plus `openpyxl` to read the `.xlsx`. Run it through the Jupyter kernel
`Python 3 (ipykernel)`.

## Notebook structure & flow

Cells are meant to be run top to bottom; later cells depend on variables defined earlier
(`df`, `numeric_cols`, `pipelines`, `build_pipeline`, `X`/`y`/`groups`, `train_sizes_abs`,
`val_rmse`, `fits`, `f`, `first_n`).

**Two models are trained side by side throughout: XGBoost and RandomForest.** Everything from
Cell 7 on is keyed by model name — `pipelines`, `val_rmse`, `fits` and `gp_fits` are all dicts
`{"XGBoost": ..., "RandomForest": ...}`, and each is exported to its own weights file.

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
5. **Cell 7** — the models. `build_pipeline(regressor)` wraps any regressor in the shared
   `SimpleImputer` → `TransformedTargetRegressor` (`log1p`/`expm1` on the target) pipeline; the
   `pipelines` dict holds an `XGBRegressor` and a `RandomForestRegressor`, each with hard-coded
   tuned params (`XGB_PARAMS` / `RF_PARAMS`).
   `feature_cols` is the **7 raw sensor columns** (`Volume`, `Area`, `MaxHeight`,
   `CenterDiviation`, `UsedCoilArea`, `SensorTemp`, `SensorValueRAW`) — not the engineered ones.
   `Label`, `DateTime` and `SensorId` are **metadata and must never be model inputs** (the sibling
   benchmark `../model_comparison/ore_sorting_ml_comparison.ipynb` excludes the same three via its
   `META_COLS`). `SensorId` was a feature until it was removed: a *shuffled* `SensorId` reproduced
   its entire apparent gain, so the effect was `colsample_bytree` drawing from one more column,
   not sensor information.
   Both are evaluated with **`GroupKFold(n_splits=10)` grouped on `Label`** (prevents leakage across
   repeated parts) via `cross_validate` (R², RMSE), then a learning curve is built per model into
   `val_rmse[name]`. RandomForest currently wins (R² 0.887 / RMSE 13.46 vs XGBoost 0.866 / 15.22).
6. **Cell 8** — Optuna search for **both** models. `SEARCH_SPACES` maps each name to its
   `(search_space, regressor_factory)` pair; `tune(name)` runs 100 trials and applies the **1-SE
   rule** on R² (RMSE as tiebreaker). Spaces mirror the sibling benchmark's. The output is pasted
   back into `XGB_PARAMS` / `RF_PARAMS` in Cell 7. RandomForest tuning is slow (~16 min) because it
   is not parallelised at the CV level — RF already parallelises across trees, and nesting both
   layers oversubscribes the CPU.
7. **Cells 9–10** — **NLS scaling law**, fitted per model. `fit_scaling_law(curve, a_lower_frac)`
   fits `rmse(n) = a + b·(n/n_ref)^(-c)` to a learning curve via `scipy.optimize.curve_fit`
   (weighted by per-size std); results land in the `fits` dict. Reports `a` as the asymptotic RMSE,
   the `N` needed for `rmse_target` (=23), the point of diminishing returns, and Monte-Carlo
   confidence bands from the parameter covariance. Both models are overlaid on one figure.
8. **Cell 11** — **Gaussian-process variant**, also per model: refits the power law with
   `a_lower_frac=0.8` (Cell 9 uses `0.3` — deliberately different floors on the asymptote), then a
   `GaussianProcessRegressor` (ConstantKernel·RBF + WhiteKernel) models residuals in `log(n)` space
   for a probabilistic extrapolation.
9. **Cells 12–13** — **inference-latency benchmark** for both models: clones + fits each pipeline,
   pulls the bare fitted regressor (`named_steps["model"].regressor_`, `n_jobs=1`), and times
   single-row `predict` — the "raw model, no DataFrame/imputer overhead" path. RandomForest gets
   fewer repeats (10k vs 100k) because it is far slower per call. Remember the raw output is in
   `log1p` space, so it applies `np.expm1` to recover real `KT10valueMax`.
10. **Cells 14–15** — **export both models**. Each fitted regressor is saved to the format its
   standalone module expects: `model_export/kt10_xgb.json` (native XGBoost JSON, loaded by
   `kt10_model.py`) and `model_export/kt10_rf.joblib` (`joblib.dump`, loaded by `kt10_model_rf.py`).

## Standalone inference modules

`kt10_model.py` (XGBoost) and `kt10_model_rf.py` (RandomForest) are deliberately **separate
modules with an identical API** — `init_model(model_path, base_dir=None) -> bool` and
`predictValue(x0…x6) -> float`, 7 positional args in `FEATURE_ORDER`. Keeping them separate keeps
the XGBoost runtime at `xgboost` + `numpy`; only the RandomForest path pulls in `scikit-learn` +
`joblib`. Both redo the `expm1` inversion themselves, so neither needs the notebook's pipeline.

- Any change to `feature_cols` in Cell 7 must be mirrored in **both** modules' `FEATURE_ORDER` and
  in their `predictValue` signatures — nothing links them programmatically.
- `test_kt10_model.py` exercises both exports side by side and asserts the two `FEATURE_ORDER`
  tuples agree.
- The `.joblib` file is tied to the scikit-learn version that wrote it; re-export after upgrading.

## Working conventions

- **Always group by `Label` in any CV/split.** Rows sharing a `Label` are the same physical
  sample; ungrouped CV leaks and inflates scores.
- The target is modeled in log space (`TransformedTargetRegressor` with `log1p`/`expm1`); keep
  metrics in the original units.
- When you re-run Optuna (Cell 8), copy the resulting best params into `XGB_PARAMS` / `RF_PARAMS`
  in Cell 7 — the two are linked only by manual copy, not programmatically.
- Re-tuning matters after a feature-set change: dropping `SensorId` while keeping the old params
  cost ~0.02 R², all of which came back once Cell 8 was re-run on the 7-feature set.
- `numeric_cols` and `feature_cols` are filtered with `[c for c in ... if c in df.columns]`, so
  renaming/removing a column degrades silently rather than erroring — verify columns exist.
