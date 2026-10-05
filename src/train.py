"""Train the final credit model and save it with its metadata.

Run from the project root:  python -m src.train
Reproduces notebooks/03_final_model.ipynb: same cleaning, split, settings and seed.
"""
import json

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import joblib

from src.features import CategoryEncoder, add_features, clean
from src.predict import META_PATH, MODEL_PATH

DATA_PATH = MODEL_PATH.parent.parent / "data" / "raw" / "application_train.csv"

# Chosen by 5-fold CV on the training split (see notebook 03).
BEST_PARAMS = dict(max_depth=4, learning_rate=0.03, n_estimators=600)
SEED = 42


def make_model(**xgb_params) -> Pipeline:
    params = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, enable_categorical=True,
                  eval_metric="auc", n_jobs=-1, random_state=SEED)
    params.update(xgb_params)
    return Pipeline([
        ("categories", CategoryEncoder()),
        ("xgb", XGBClassifier(**params)),
    ])


def main():
    raw = pd.read_csv(DATA_PATH)
    raw_train, raw_test = train_test_split(raw, test_size=0.2, stratify=raw["TARGET"], random_state=SEED)

    # Raw application fields that were never blank in training. The model has never seen
    # them missing, so its output for a blank one is unreliable; the API requires them.
    input_cols = raw_train.columns.drop(["TARGET", "SK_ID_CURR"])
    required = [c for c in input_cols if raw_train[c].notna().all()]

    def to_xy(d):
        X = add_features(clean(d)).drop(columns=["TARGET", "SK_ID_CURR"])
        return X, d["TARGET"]

    X_train, y_train = to_xy(raw_train)
    X_test, y_test = to_xy(raw_test)

    model = make_model(**BEST_PARAMS).fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, p_test), 4),
        "pr_auc": round(average_precision_score(y_test, p_test), 4),
        "log_loss": round(log_loss(y_test, p_test), 4),
        "baseline_log_loss": round(log_loss(y_test, np.full(len(y_test), y_train.mean())), 4),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    META_PATH.write_text(json.dumps({
        "model": "CategoryEncoder + XGBClassifier",
        "params": BEST_PARAMS,
        "seed": SEED,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": list(X_train.columns),
        "required_input_columns": required,
        "test_metrics": metrics,
        "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "pandas": pd.__version__},
    }, indent=2))

    print("test metrics:", metrics)
    print(f"{len(required)} required input columns")
    print("saved", MODEL_PATH.name, "and", META_PATH.name)


if __name__ == "__main__":
    main()
