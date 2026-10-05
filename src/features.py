import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DAYS_EMPLOYED_SENTINEL = 365243


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data problems. Works row by row, so it is safe before the split."""
    df = df.copy()
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered ratios. Uses only application-time fields, so no leakage."""
    df = df.copy()
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["EMPLOYED_AGE_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    return df


class CategoryEncoder(BaseEstimator, TransformerMixin):
    """Converts text columns to pandas 'category' with categories learned on fit.

    Categories unseen during fit become NaN, so new values can't leak in or crash.
    """

    def fit(self, X, y=None):
        self.cat_cols_ = X.select_dtypes(exclude="number").columns.tolist()
        self.categories_ = {c: X[c].astype("category").cat.categories for c in self.cat_cols_}
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cat_cols_:
            X[c] = pd.Categorical(X[c], categories=self.categories_[c])
        return X
