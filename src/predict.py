import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features import add_features, clean

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "credit_model.joblib"
META_PATH = MODEL_PATH.with_name("credit_model_meta.json")

# Break-even cutoff from the expected-profit analysis (margin 0.10, LGD 0.45).
# These are teaching assumptions, not real lender economics.
APPROVAL_CUTOFF = 0.10 / (0.10 + 0.45)


@lru_cache(maxsize=1)
def load_model():
    """Load the trained pipeline once and reuse it for every prediction."""
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    """Training-time facts about the model, written by src/train.py."""
    return json.loads(META_PATH.read_text())


def missing_required(application: dict) -> list[str]:
    """Required input columns that are absent or null in one application."""
    return [c for c in load_metadata()["required_input_columns"] if application.get(c) is None]


def predict(applications: pd.DataFrame) -> pd.DataFrame:
    """Score raw application rows. Missing fields become NaN; unknown fields are ignored."""
    model = load_model()
    model_cols = list(model.named_steps["xgb"].feature_names_in_)
    cat_cols = model.named_steps["categories"].cat_cols_

    X = applications.reindex(columns=model_cols)            # exact training columns, in order
    num_cols = [c for c in model_cols if c not in cat_cols]
    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")
    X = add_features(clean(X))[model_cols]

    pd_default = model.predict_proba(X)[:, 1]
    return pd.DataFrame({
        "SK_ID_CURR": applications["SK_ID_CURR"].values if "SK_ID_CURR" in applications else None,
        "pd_default": pd_default,
        "decision": ["approve" if p < APPROVAL_CUTOFF else "reject" for p in pd_default],
    })
