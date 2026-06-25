import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def get_model_instances(random_state=42):
    models = {}
    models['RandomForest'] = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    models['MLP'] = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=400, random_state=random_state)
    if _HAS_XGB:
        models['XGBoost'] = xgb.XGBClassifier(n_estimators=200, eval_metric='mlogloss', random_state=random_state)
    return models


def train_model(name: str, model, X_train, y_train, X_val=None, y_val=None, record_training=False):
    history = {}
    if name == 'XGBoost' and _HAS_XGB and X_val is not None:
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        history['evals_result'] = model.evals_result()
    else:
        model.fit(X_train, y_train)
        if hasattr(model, 'loss_curve_'):
            history['loss_curve'] = getattr(model, 'loss_curve_')

    return model, history


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)


def inference_timing(model, X: np.ndarray, repeat=3) -> float:
    # return average time per sample in milliseconds
    n = X.shape[0]
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = model.predict(X)
        t1 = time.perf_counter()
        times.append((t1 - t0) / n * 1000.0)
    return float(np.mean(times))


def evaluate(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    return acc
