import argparse
from pathlib import Path
import joblib
from .data_processing import load_and_clean, split_xy, scale_fit_transform
from .models.core import get_model_instances, train_model, save_model
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='训练集 CSV 文件路径')
    parser.add_argument('--model', default='RandomForest', choices=['RandomForest', 'MLP', 'XGBoost'])
    parser.add_argument('--out', default='models', help='模型保存目录')
    args = parser.parse_args()

    df = load_and_clean(args.csv)
    X, y = split_xy(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler, X_train_s, X_val_s = scale_fit_transform(X_train, X_val)

    models = get_model_instances()
    model = models[args.model]

    model, history = train_model(args.model, model, X_train_s, y_train, X_val_s, y_val, record_training=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model({'model': model, 'scaler': scaler}, out_dir / f'{args.model}.joblib')
    print('Saved model to', out_dir / f'{args.model}.joblib')


if __name__ == '__main__':
    main()
