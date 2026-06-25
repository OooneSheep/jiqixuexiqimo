import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def _clean_class_name(s: str) -> str:
    if pd.isna(s):
        return s
    s = str(s).strip().upper()
    s = s.translate(str.maketrans({'0': 'O', '3': 'E', '1': 'I', '5': 'S'}))
    s = re.sub(r'[^A-Za-z]', '', s)
    return s


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Clean numeric columns: coerce errors then fill medians
    num_cols = df.columns.drop('Class')
    for c in num_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')

    # Basic cleaning
    df = df.drop_duplicates()

    # Fill numeric missing values with median
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # Clean class labels
    df['Class'] = df['Class'].apply(_clean_class_name)

    # Drop rows where Class is missing after cleaning
    df = df[~df['Class'].isna()].reset_index(drop=True)

    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=['Class']).values
    y = df['Class'].values
    return X, y


def scale_fit_transform(X_train: np.ndarray, X_val: np.ndarray = None):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val) if X_val is not None else None
    return scaler, X_train_s, X_val_s


def add_gaussian_noise(X: np.ndarray, std: float):
    if std <= 0:
        return X.copy()
    noise = np.random.normal(scale=std, size=X.shape)
    return X + noise


def add_label_noise(y: np.ndarray, fraction: float, classes=None, random_state=None):
    rng = np.random.default_rng(random_state)
    n = len(y)
    k = int(n * fraction)
    idx = rng.choice(n, size=k, replace=False)
    if classes is None:
        classes = np.unique(y)
    for i in idx:
        others = [c for c in classes if c != y[i]]
        y[i] = rng.choice(others)
    return y
