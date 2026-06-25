import yaml
import time
import json
import logging
from pathlib import Path
import joblib
from ..data_processing import load_and_clean, split_xy
from ..features import build_feature_pipeline
from ..models.core import get_model_instances
from sklearn.model_selection import train_test_split


def _setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('trainer')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(str(log_path))
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def train_from_config(config_path: str):
    with open(config_path, 'r', encoding='utf8') as f:
        cfg = yaml.safe_load(f)

    logger = _setup_logger(Path(cfg.get('out_dir', 'models')) / 'train.log')
    logger.info('Loaded config %s', config_path)

    df = load_and_clean(cfg['data']['train_csv'])
    X, y = split_xy(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg['data'].get('val_split', 0.2), random_state=cfg.get('random_state', 42), stratify=y)

    pipe = build_feature_pipeline(polynomial_degree=cfg.get('feature', {}).get('poly_degree', None), var_thresh=cfg.get('feature', {}).get('var_thresh', 0.0), model_select=cfg.get('feature', {}).get('model_select', False))
    pipe.fit(X_train)
    X_train_t = pipe.transform(X_train)
    X_val_t = pipe.transform(X_val)

    models = get_model_instances(cfg.get('random_state', 42))
    model_name = cfg['model']['name']
    model = models.get(model_name)
    if model is None:
        raise ValueError('Unknown model: ' + model_name)

    # set params if provided
    params = cfg['model'].get('params', {})
    if params:
        model.set_params(**params)

    t0 = time.perf_counter()
    model.fit(X_train_t, y_train)
    t1 = time.perf_counter()
    train_time = t1 - t0

    metrics = {
        'train_time_s': train_time,
        'train_accuracy': float(model.score(X_train_t, y_train)),
        'val_accuracy': float(model.score(X_val_t, y_val)),
    }

    out_dir = Path(cfg.get('out_dir', 'models'))
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'pipeline': pipe}, out_dir / f'{model_name}.joblib')

    with open(out_dir / 'metrics.json', 'w', encoding='utf8') as f:
        json.dump(metrics, f, indent=2)

    logger.info('Training finished. Metrics: %s', metrics)
    return metrics
