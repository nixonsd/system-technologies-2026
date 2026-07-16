"""Standalone KT10valueMax inference module.

Exposes two functions required by the integration spec (plan.xlsx, sheet "Вимоги"):

    init_model(model_path, base_dir=None) -> bool
    predictValue(x0, x1, x2, x3, x4, x5, x6, x7) -> float

The model file is the bare fitted XGBoost regressor exported from
``system_technologies.ipynb`` (native ``save_model`` JSON). The regressor is trained
on the target in log1p space, so ``predictValue`` inverts it with ``expm1`` to return
KT10valueMax in real units.

Runtime dependencies: xgboost, numpy (plus the standard-library ``os``).
"""

import os

import numpy as np
import xgboost as xgb

# Feature order per the "Вимоги" sheet / notebook Cell 7 feature_cols:
# SensorId, Volume, Area, MaxHeight, CenterDiviation, UsedCoilArea, SensorTemp, SensorValueRAW
FEATURE_ORDER = (
    "SensorId",
    "Volume",
    "Area",
    "MaxHeight",
    "CenterDiviation",
    "UsedCoilArea",
    "SensorTemp",
    "SensorValueRAW",
)

_model = None  # loaded XGBRegressor, set by init_model()


def init_model(model_path, base_dir=None):
    """Load the model weights file.

    Args:
        model_path: path to the exported model file (e.g. "kt10_xgb.json").
        base_dir: optional base directory prepended to ``model_path`` so model
            files can live anywhere without changing the program's working
            directory.

    Returns:
        True if the model was loaded successfully, False if the file is missing
        or any other error occurred.
    """
    global _model
    try:
        full_path = os.path.join(base_dir, model_path) if base_dir else model_path
        if not os.path.isfile(full_path):
            return False
        model = xgb.XGBRegressor()
        model.load_model(full_path)
        _model = model
        return True
    except Exception:
        return False


def predictValue(x0, x1, x2, x3, x4, x5, x6, x7):
    """Predict KT10valueMax from the 8 raw sensor features.

    Arguments are passed as separate values (not a struct) in the order:
    SensorId, Volume, Area, MaxHeight, CenterDiviation, UsedCoilArea,
    SensorTemp, SensorValueRAW.

    Returns:
        Predicted KT10valueMax as a float (already inverted from log1p space).

    Raises:
        RuntimeError: if called before a successful init_model().
    """
    if _model is None:
        raise RuntimeError("Model not initialized. Call init_model() first.")
    features = np.array([[x0, x1, x2, x3, x4, x5, x6, x7]], dtype=np.float32)
    raw_pred = _model.predict(features, validate_features=False)
    return float(np.expm1(raw_pred[0]))


if __name__ == "__main__":
    # Smoke test against the exported model.
    default_path = os.path.join("model_export", "kt10_xgb.json")
    if not init_model(default_path):
        raise SystemExit(f"Could not load model from {default_path!r} — run the "
                         f"export cell in system_technologies.ipynb first.")
    # Example feature vector (SensorId, Volume, Area, MaxHeight, CenterDiviation,
    # UsedCoilArea, SensorTemp, SensorValueRAW).
    sample = (1, 1000.0, 500.0, 20.0, 5.0, 300.0, 25.0, 1500.0)
    print("Loaded model:", default_path)
    print("Predicted KT10valueMax:", predictValue(*sample))
