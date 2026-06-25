import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(model, scaler, X, y):
    Xs = scaler.transform(X)
    y_pred = model.predict(Xs)
    acc = accuracy_score(y, y_pred)
    return {'accuracy': acc, 'report': classification_report(y, y_pred, zero_division=0)}


def plot_loss(history: dict, out_path: Path):
    # history may contain 'loss_curve' or xgboost evals_result
    if 'loss_curve' in history:
        plt.figure()
        plt.plot(history['loss_curve'])
        plt.title('Loss Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
    elif 'evals_result' in history:
        res = history['evals_result']
        # try to plot train and validation mlogloss
        try:
            train_loss = res['validation_0']['mlogloss']
            val_loss = res['validation_1']['mlogloss']
            plt.figure()
            plt.plot(train_loss, label='train')
            plt.plot(val_loss, label='val')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('mlogloss')
            plt.savefig(out_path)
            plt.close()
        except Exception:
            pass


def time_inference(model, scaler, X, runs=3):
    Xs = scaler.transform(X)
    n = Xs.shape[0]
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model.predict(Xs)
        t1 = time.perf_counter()
        times.append((t1 - t0) / n * 1000.0)
    return float(np.mean(times))
