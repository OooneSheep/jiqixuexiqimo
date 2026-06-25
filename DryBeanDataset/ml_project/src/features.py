from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from typing import Optional
import joblib


def build_feature_pipeline(polynomial_degree: Optional[int] = None, var_thresh: float = 0.0, model_select: bool = False):
    steps = []
    steps.append(('scaler', StandardScaler()))
    if polynomial_degree and polynomial_degree > 1:
        steps.append(('poly', PolynomialFeatures(degree=polynomial_degree, include_bias=False)))
    if var_thresh and var_thresh > 0.0:
        steps.append(('var', VarianceThreshold(threshold=var_thresh)))
    if model_select:
        selector = SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42))
        steps.append(('select', selector))

    pipeline = Pipeline(steps)
    return pipeline


def save_pipeline(pipe, path: str):
    joblib.dump(pipe, path)


def load_pipeline(path: str):
    return joblib.load(path)
