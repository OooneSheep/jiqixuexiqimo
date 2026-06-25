from typing import Optional
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def detect_missing(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def impute_numeric(df: pd.DataFrame, strategy: str = 'median', use_knn: bool = False, knn_k: int = 5) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    out = df.copy()
    if use_knn:
        imputer = KNNImputer(n_neighbors=knn_k)
        out[num_cols] = imputer.fit_transform(out[num_cols])
        return out

    for c in num_cols:
        if strategy == 'median':
            out[c] = out[c].fillna(out[c].median())
        elif strategy == 'mean':
            out[c] = out[c].fillna(out[c].mean())
        else:
            out[c] = out[c].fillna(0)
    return out


def iqr_clip(df: pd.DataFrame, cols: Optional[list] = None, factor: float = 1.5) -> pd.DataFrame:
    out = df.copy()
    if cols is None:
        cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for c in cols:
        q1 = out[c].quantile(0.25)
        q3 = out[c].quantile(0.75)
        iqr = q3 - q1
        low = q1 - factor * iqr
        high = q3 + factor * iqr
        out[c] = out[c].clip(lower=low, upper=high)
    return out


def clean_df(df: pd.DataFrame, impute_strategy: str = 'median', use_knn: bool = False, knn_k: int = 5, iqr: bool = True) -> pd.DataFrame:
    out = df.copy()
    out = drop_duplicates(out)

    # coerce numeric columns
    for c in out.columns:
        if out[c].dtype == object:
            # try to coerce numeric-like strings safely: convert where possible and keep originals otherwise
            s = out[c].astype(str).str.replace(',', '')
            tmp = pd.to_numeric(s, errors='coerce')
            mask = tmp.notna()
            if mask.any():
                out.loc[mask, c] = tmp[mask]

    out = impute_numeric(out, strategy=impute_strategy, use_knn=use_knn, knn_k=knn_k)
    if iqr:
        out = iqr_clip(out)

    return out
