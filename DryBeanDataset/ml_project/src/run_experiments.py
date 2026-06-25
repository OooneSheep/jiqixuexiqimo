"""Run the full Dry Bean machine-learning project pipeline.

Usage:
python -m src.run_experiments --data-dir .. --out experiments/full_run --site-dir docs
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from .data_processing import (
    add_gaussian_noise,
    add_label_noise,
    load_and_clean,
    scale_fit_transform,
    split_xy,
)
from .evaluate import plot_loss, time_inference
from .models.core import evaluate, get_model_instances, save_model, train_model


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dirty_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for col in df.columns:
        if col == "Class":
            continue
        out[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    return out


def _save_class_distribution(clean_train: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(9, 4.8))
    order = clean_train["Class"].value_counts().index
    sns.countplot(data=clean_train, x="Class", order=order, color="#3b82f6")
    plt.title("Class Distribution")
    plt.xlabel("Bean class")
    plt.ylabel("Samples")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(fig_dir / "class_distribution.png", dpi=160)
    plt.close()


def _save_missing_plot(raw_train: pd.DataFrame, fig_dir: Path) -> None:
    missing = raw_train.isna().sum().sort_values(ascending=False)
    if missing.sum() == 0:
        missing = _dirty_numeric_frame(raw_train).isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        missing = pd.Series({"No missing cells": 0})
    plt.figure(figsize=(9, 4.8))
    missing.plot(kind="bar", color="#f97316")
    plt.title("Missing or Invalid Numeric Values")
    plt.xlabel("Feature")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(fig_dir / "missing_values.png", dpi=160)
    plt.close()


def _save_correlation_heatmap(clean_train: pd.DataFrame, fig_dir: Path) -> None:
    corr = clean_train.drop(columns=["Class"]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, square=True, cbar_kws={"shrink": 0.78})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(fig_dir / "correlation_heatmap.png", dpi=160)
    plt.close()


def _save_feature_histograms(clean_train: pd.DataFrame, fig_dir: Path) -> None:
    cols = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundness", "Compactness"]
    clean_train[cols].hist(figsize=(11, 7), bins=32, color="#14b8a6", edgecolor="white")
    plt.suptitle("Representative Feature Distributions")
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_histograms.png", dpi=160)
    plt.close()


def _save_pca_plot(clean_train: pd.DataFrame, fig_dir: Path) -> None:
    X, y = split_xy(clean_train)
    _, Xs, _ = scale_fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    pts = pca.fit_transform(Xs)
    plot_df = pd.DataFrame({"PC1": pts[:, 0], "PC2": pts[:, 1], "Class": y})
    plt.figure(figsize=(8.8, 6.2))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Class", s=18, alpha=0.78, linewidth=0)
    plt.title("PCA 2D Projection")
    plt.tight_layout()
    plt.savefig(fig_dir / "pca2d.png", dpi=160)
    plt.close()


def run_eda(csv_train: Path, csv_val: Path, csv_test: Path, out_dir: Path) -> dict:
    fig_dir = _ensure_dir(out_dir / "figs")
    raw_train = pd.read_csv(csv_train)
    raw_val = pd.read_csv(csv_val)
    raw_test = pd.read_csv(csv_test)
    clean_train = load_and_clean(str(csv_train))

    numeric_dirty = _dirty_numeric_frame(raw_train)
    raw_numeric_missing = raw_train.drop(columns=["Class"]).isna().sum().sum()
    invalid_numeric = int(numeric_dirty.isna().sum().sum() - raw_numeric_missing)
    summary = {
        "train_rows_raw": int(len(raw_train)),
        "val_rows_raw": int(len(raw_val)),
        "test_rows_raw": int(len(raw_test)),
        "train_rows_clean": int(len(clean_train)),
        "feature_count": int(clean_train.shape[1] - 1),
        "class_count": int(clean_train["Class"].nunique()),
        "duplicate_rows_removed": int(len(raw_train) - len(raw_train.drop_duplicates())),
        "missing_cells_raw_train": int(raw_train.isna().sum().sum()),
        "invalid_numeric_cells_train": max(invalid_numeric, 0),
        "class_distribution_train": clean_train["Class"].value_counts().to_dict(),
    }

    _save_class_distribution(clean_train, fig_dir)
    _save_missing_plot(raw_train, fig_dir)
    _save_correlation_heatmap(clean_train, fig_dir)
    _save_feature_histograms(clean_train, fig_dir)
    _save_pca_plot(clean_train, fig_dir)

    with open(out_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _save_metric_plots(summary: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    sns.barplot(data=summary, x="model", y="acc_test", color="#22c55e")
    plt.ylim(0, 1)
    plt.title("Test Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_compare.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    sns.barplot(data=summary, x="model", y="infer_ms_per_sample", color="#8b5cf6")
    plt.title("Inference Latency Comparison")
    plt.xlabel("Model")
    plt.ylabel("Milliseconds per sample")
    plt.tight_layout()
    plt.savefig(out_dir / "inference_speed_compare.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    sns.barplot(data=summary, x="model", y="overfit_gap", color="#ef4444")
    plt.title("Overfitting Gap")
    plt.xlabel("Model")
    plt.ylabel("Train accuracy - test accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "overfit_gap_compare.png", dpi=160)
    plt.close()


def run_once(csv_train: Path, csv_test: Path, out_dir: Path) -> pd.DataFrame:
    df_train = load_and_clean(str(csv_train))
    df_test = load_and_clean(str(csv_test))
    X_train, y_train = split_xy(df_train)
    X_test, y_test = split_xy(df_test)

    scaler, X_train_s, X_test_s = scale_fit_transform(X_train, X_test)
    le = LabelEncoder()
    y_train_e = le.fit_transform(y_train)
    y_test_e = le.transform(y_test)

    models = get_model_instances()
    results = []
    combined_loss = {}

    for name, model in models.items():
        print("Training", name)
        fitted, history = train_model(name, model, X_train_s, y_train_e, X_test_s, y_test_e, record_training=True)
        save_model({"model": fitted, "scaler": scaler, "label_encoder": le}, out_dir / f"{name}.joblib")

        acc_train = evaluate(fitted, X_train_s, y_train_e)
        acc_test = evaluate(fitted, X_test_s, y_test_e)
        infer_ms = time_inference(fitted, scaler, X_test)
        y_pred = fitted.predict(X_test_s)
        report = classification_report(
            y_test_e,
            y_pred,
            target_names=le.classes_,
            zero_division=0,
            output_dict=True,
        )
        with open(out_dir / f"{name}_classification_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        cm = confusion_matrix(y_test_e, y_pred)
        plt.figure(figsize=(7.5, 6.2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_confusion_matrix.png", dpi=160)
        plt.close()

        plot_loss(history, out_dir / f"{name}_loss.png")
        if "loss_curve" in history:
            combined_loss[name] = history["loss_curve"]
        elif "evals_result" in history:
            result = history["evals_result"]
            if "validation_1" in result and "mlogloss" in result["validation_1"]:
                combined_loss[name] = result["validation_1"]["mlogloss"]

        results.append(
            {
                "model": name,
                "acc_train": acc_train,
                "acc_test": acc_test,
                "overfit_gap": acc_train - acc_test,
                "infer_ms_per_sample": infer_ms,
                "macro_f1": report["macro avg"]["f1-score"],
            }
        )

    if combined_loss:
        plt.figure(figsize=(8, 5))
        for name, values in combined_loss.items():
            plt.plot(values, label=name)
        plt.title("Comparable Loss Curves")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curves_compare.png", dpi=160)
        plt.close()

    summary = pd.DataFrame(results)
    _save_metric_plots(summary, out_dir)
    return summary


def run_robustness(csv_train: Path, csv_test: Path, out_dir: Path) -> pd.DataFrame:
    df_train = load_and_clean(str(csv_train))
    df_test = load_and_clean(str(csv_test))
    X_train, y_train = split_xy(df_train)
    X_test, y_test = split_xy(df_test)

    records = []
    le = LabelEncoder()
    y_train_e = le.fit_transform(y_train)
    y_test_e = le.transform(y_test)

    noise_settings = [
        ("gaussian_feature", 0.0),
        ("gaussian_feature", 0.01),
        ("gaussian_feature", 0.05),
        ("gaussian_feature", 0.10),
        ("label_flip", 0.01),
        ("label_flip", 0.05),
        ("label_flip", 0.10),
    ]

    for noise_type, strength in noise_settings:
        if noise_type == "gaussian_feature":
            X_noisy = add_gaussian_noise(X_train, std=strength)
            y_noisy = y_train_e.copy()
        else:
            X_noisy = X_train.copy()
            y_noisy = add_label_noise(y_train_e.copy(), fraction=strength, classes=np.unique(y_train_e), random_state=42)

        scaler, X_noisy_s, X_test_s = scale_fit_transform(X_noisy, X_test)
        for name, model in get_model_instances().items():
            fitted, _ = train_model(name, model, X_noisy_s, y_noisy, X_test_s, y_test_e)
            acc = evaluate(fitted, X_test_s, y_test_e)
            records.append({"model": name, "noise_type": noise_type, "strength": strength, "acc_test": acc})

    robustness = pd.DataFrame(records)
    plt.figure(figsize=(9, 5.4))
    sns.lineplot(
        data=robustness,
        x="strength",
        y="acc_test",
        hue="model",
        style="noise_type",
        markers=True,
        dashes=False,
    )
    plt.ylim(0, 1)
    plt.title("Robustness Under Feature and Label Noise")
    plt.xlabel("Noise strength")
    plt.ylabel("Test accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "robustness_compare.png", dpi=160)
    plt.close()
    return robustness


def _df_to_html_table(df: pd.DataFrame, float_format="{:.4f}") -> str:
    return df.to_html(index=False, classes="data-table", border=0, float_format=lambda x: float_format.format(x))


def write_site(site_dir: Path, out_dir: Path, eda_summary: dict, summary: pd.DataFrame, robustness: pd.DataFrame) -> None:
    _ensure_dir(site_dir)
    assets_dir = _ensure_dir(site_dir / "assets")
    for png in list((out_dir / "figs").glob("*.png")) + list(out_dir.glob("*.png")):
        (assets_dir / png.name).write_bytes(png.read_bytes())

    best = summary.sort_values("acc_test", ascending=False).iloc[0]
    rob_drop = (
        robustness.groupby(["model", "noise_type"])["acc_test"]
        .agg(["max", "min"])
        .assign(drop=lambda d: d["max"] - d["min"])
        .reset_index()
        .sort_values("drop", ascending=False)
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dry Bean 多分类机器学习工程展示</title>
  <style>
    :root {{
      --ink: #162033;
      --muted: #5b677a;
      --line: #d8dee9;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --blue: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.65;
    }}
    header {{ background: #ffffff; border-bottom: 1px solid var(--line); }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(28px, 4vw, 48px); }}
    h2 {{ margin: 42px 0 14px; font-size: 26px; }}
    h3 {{ margin: 22px 0 10px; font-size: 18px; }}
    p {{ margin: 0 0 12px; color: var(--muted); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 20px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric strong {{ display: block; font-size: 24px; color: var(--blue); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
      gap: 16px;
    }}
    figure {{
      margin: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }}
    figure img {{ width: 100%; display: block; border-radius: 4px; }}
    figcaption {{ padding: 8px 4px 2px; color: var(--muted); font-size: 14px; }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      font-size: 14px;
      margin: 12px 0 18px;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
    }}
    .data-table th {{ background: #eef3fb; }}
    code {{ background: #e8edf6; padding: 2px 5px; border-radius: 4px; }}
    ul {{ padding-left: 20px; }}
    li {{ margin-bottom: 6px; color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>Dry Bean Dataset 多分类机器学习工程</h1>
      <p>覆盖数据分析、数据清洗、特征工程、多算法实验、鲁棒性评估、系统展示和课程总结。</p>
      <div class="metrics">
        <div class="metric"><strong>{eda_summary["train_rows_clean"]}</strong>清洗后训练样本</div>
        <div class="metric"><strong>{eda_summary["feature_count"]}</strong>数值形态特征</div>
        <div class="metric"><strong>{eda_summary["class_count"]}</strong>干豆类别</div>
        <div class="metric"><strong>{best["model"]}</strong>测试集最优模型：{best["acc_test"]:.4f}</div>
      </div>
    </div>
  </header>
  <main class="wrap">
    <section>
      <h2>数据描述与污染观察</h2>
      <p>数据由干豆图像提取的几何与形态学特征组成，目标是预测 <code>Class</code> 中的干豆品种。训练集中发现的主要问题包括重复行、缺失值、数值列中的非法字符、类别标签格式不统一，以及强相关特征带来的冗余。</p>
      <div class="grid">
        <figure><img src="assets/class_distribution.png" alt="类别分布"><figcaption>类别分布：可观察类别不完全均衡，模型评价需要同时看宏平均 F1。</figcaption></figure>
        <figure><img src="assets/missing_values.png" alt="缺失值"><figcaption>缺失/非法数值统计：指导后续中位数填充和类型转换。</figcaption></figure>
        <figure><img src="assets/correlation_heatmap.png" alt="相关性热力图"><figcaption>相关性热力图：面积、周长、轴长等特征存在明显相关。</figcaption></figure>
        <figure><img src="assets/pca2d.png" alt="PCA"><figcaption>PCA 二维投影：部分类别边界重叠，说明任务不是简单线性可分。</figcaption></figure>
      </div>
    </section>
    <section>
      <h2>数据处理与特征工程</h2>
      <ul>
        <li>列名标准化，所有数值特征统一转为浮点数，逗号和非法字符通过强制类型转换识别。</li>
        <li>删除重复样本，降低训练集泄漏和重复记忆风险。</li>
        <li>数值缺失值使用训练数据中位数填充，避免均值受异常值拉偏。</li>
        <li>类别标签去除非字母字符并转为大写，保证训练、测试标签编码一致。</li>
        <li>所有模型输入使用 <code>StandardScaler</code> 标准化，保证 MLP 与 XGBoost/随机森林处在统一评估流程下。</li>
      </ul>
    </section>
    <section>
      <h2>多算法实验结果</h2>
      {_df_to_html_table(summary)}
      <div class="grid">
        <figure><img src="assets/accuracy_compare.png" alt="准确率"><figcaption>测试集准确率对比。</figcaption></figure>
        <figure><img src="assets/loss_curves_compare.png" alt="loss 曲线"><figcaption>训练型模型 loss 曲线对比，随机森林不适用。</figcaption></figure>
        <figure><img src="assets/inference_speed_compare.png" alt="推理速度"><figcaption>平均单样本推理延迟对比。</figcaption></figure>
        <figure><img src="assets/overfit_gap_compare.png" alt="过拟合差距"><figcaption>训练集与测试集准确率差距，用于过拟合分析。</figcaption></figure>
      </div>
    </section>
    <section>
      <h2>鲁棒性分析</h2>
      <p>在训练数据中加入两类噪声：数值特征高斯噪声和标签翻转噪声。下表与曲线展示模型随噪声强度增加的测试集精度变化。</p>
      {_df_to_html_table(robustness)}
      <div class="grid">
        <figure><img src="assets/robustness_compare.png" alt="鲁棒性"><figcaption>鲁棒性曲线：标签噪声通常比小幅特征噪声更容易破坏泛化性能。</figcaption></figure>
        <figure><img src="assets/MLP_confusion_matrix.png" alt="MLP 混淆矩阵"><figcaption>MLP 混淆矩阵：观察易混类别。</figcaption></figure>
      </div>
    </section>
    <section>
      <h2>工程运行方式</h2>
      <p>统一命令：<code>python -m src.run_experiments --data-dir .. --out experiments/full_run --site-dir docs</code>。算法运行阶段无 UI，所有表格、模型文件、图片和展示页面自动写入工程目录。</p>
      <h3>补充维度：最大精度下降</h3>
      {_df_to_html_table(rob_drop)}
    </section>
  </main>
</body>
</html>
"""
    (site_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="..", help="Directory containing train/val/test CSV files.")
    parser.add_argument("--out", default="experiments/full_run", help="Experiment output directory.")
    parser.add_argument("--site-dir", default="docs", help="Static GitHub Pages directory.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = _ensure_dir(Path(args.out))
    site_dir = Path(args.site_dir)

    csv_train = data_dir / "Dry_Bean_Dataset_Dirty_train.csv"
    csv_val = data_dir / "Dry_Bean_Dataset_Dirty_val.csv"
    csv_test = data_dir / "Dry_Bean_Dataset_Dirty_test.csv"

    eda_summary = run_eda(csv_train, csv_val, csv_test, out_dir)
    summary = run_once(csv_train, csv_test, out_dir)
    summary.to_csv(out_dir / "summary.csv", index=False)

    robustness = run_robustness(csv_train, csv_test, out_dir)
    robustness.to_csv(out_dir / "robustness.csv", index=False)

    write_site(site_dir, out_dir, eda_summary, summary, robustness)

    print("Experiments finished.")
    print("Results:", out_dir.resolve())
    print("Website:", (site_dir / "index.html").resolve())


if __name__ == "__main__":
    main()
