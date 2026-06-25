import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from src.data.cleaning import clean_df, detect_missing


def test_clean_imputes_and_clips():
    df = pd.DataFrame({
        'a': [1, 2, None, 10000],
        'b': [10, None, 30, 40],
        'Class': ['x', 'y', 'x', 'x']
    })
    df2 = clean_df(df, impute_strategy='median', use_knn=False, iqr=True)
    # no missing in numeric columns
    num_cols = df2.select_dtypes(include=[np.number]).columns
    assert not df2[num_cols].isna().any().any()

    # extreme outlier clipped: max should be less than original extreme
    assert df2['a'].max() < 10000
