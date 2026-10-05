from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features import add_features, clean

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "credit_model.joblib"

# Break-even cutoff from the expected-profit analysis (margin 0.10, LGD 0.45).
# These are teaching assumptions, not real lender economics.
APPROVAL_CUTOFF = 0.10 / (0.10 + 0.45)


@lru_cache(maxsize=1)
def load_model():
    """Load the trained pipeline once and reuse it for every prediction."""
    return joblib.load(MODEL_PATH)


def predict(applications: pd.DataFrame) -> pd.DataFrame:
    """Score raw application rows (same columns as application_train.csv, without TARGET)."""
    features = add_features(clean(applications)).drop(columns=["SK_ID_CURR"], errors="ignore")
    pd_default = load_model().predict_proba(features)[:, 1]
    return pd.DataFrame({
        "SK_ID_CURR": applications["SK_ID_CURR"].values if "SK_ID_CURR" in applications else None,
        "pd_default": pd_default,
        "decision": ["approve" if p < APPROVAL_CUTOFF else "reject" for p in pd_default],
    })
