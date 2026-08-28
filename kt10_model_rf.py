"""Standalone KT10valueMax inference module — RandomForest variant.

Mirrors the API of ``kt10_model.py`` exactly, so the two are drop-in
interchangeable for a caller:

    init_model(model_path, base_dir=None) -> bool
    predictValue(x0, x1, x2, x3, x4, x5, x6) -> float

The model file is the bare fitted ``RandomForestRegressor`` exported from
``system_technologies.ipynb`` via ``joblib.dump`` to ``model_export/kt10_rf.joblib``.
The regressor is trained on the target in log1p space, so ``predictValue`` inverts it
with ``expm1`` to return KT10valueMax in real units.

Runtime dependencies: scikit-learn, joblib, numpy. Note this is a heavier runtime than
``kt10_model.py`` (xgboost + numpy), and RandomForest inference is markedly slower per
call — see the latency benchmark in the notebook before choosing between them.

A joblib/pickle model file is tied to the scikit-learn version that wrote it; loading it
under a different version may warn or fail. Re-export from the notebook after upgrading.
"""

import os

import joblib
import numpy as np

# Feature order per notebook Cell 7 feature_cols — identical to kt10_model.py.
# Label, DateTime and SensorId are metadata rather than sensor measurements and are
# deliberately not model inputs.
FEATURE_ORDER = (
    "Volume",
    "Area",
    "MaxHeight",
    "CenterDiviation",
    "UsedCoilArea",
    "SensorTemp",
    "SensorValueRAW",
)

_model = None  # loaded RandomForestRegressor, set by init_model()


def init_model(model_path, base_dir=None):
    """Load the model weights file.

    Args:
        model_path: path to the exported model file (e.g. "kt10_rf.joblib").
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
        _model = joblib.load(full_path)
        return True
    except Exception:
        return False


def predictValue(x0, x1, x2, x3, x4, x5, x6):
    """Predict KT10valueMax from the 7 raw sensor features.

    Arguments are passed as separate values (not a struct) in the order:
    Volume, Area, MaxHeight, CenterDiviation, UsedCoilArea, SensorTemp,
    SensorValueRAW.

    Returns:
        Predicted KT10valueMax as a float (already inverted from log1p space).

    Raises:
        RuntimeError: if called before a successful init_model().
    """
    if _model is None:
        raise RuntimeError("Model not initialized. Call init_model() first.")
    features = np.array([[x0, x1, x2, x3, x4, x5, x6]], dtype=np.float32)
    raw_pred = _model.predict(features)
    return float(np.expm1(raw_pred[0]))


if __name__ == "__main__":
    # Smoke test against the exported model.
    default_path = os.path.join("model_export", "kt10_rf.joblib")
    if not init_model(default_path):
        raise SystemExit(f"Could not load model from {default_path!r} — run the "
                         f"export cell in system_technologies.ipynb first.")
    # Example feature vectors (Volume, Area, MaxHeight, CenterDiviation,
    # UsedCoilArea, SensorTemp, SensorValueRAW) — the three UL17.1.0 measurements
    # from 2026-5-28_9-3-40, all with an actual KT10valueMax of 6.5.
    samples = [
        (13.4982, 15.7059, 1.29118, 14.8655, 0.565217, 29, 200.31),
        (13.4982, 15.7059, 1.29118, 6.06901, 0.731225, 29, 118.31),
        (13.4982, 15.7059, 1.29118, 9.9104, 0.660079, 29, 58.3097),
        (13.532123565673828, 15.226237297058105, 1.3174400329589844,
         6.104642391204834, 0.7588932514190674, 33.5, 254.85765075683594),
    ]
    print("Loaded model:", default_path)
    for sample in samples:
        print("Predicted KT10valueMax:", predictValue(*sample))
