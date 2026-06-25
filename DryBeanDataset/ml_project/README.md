# Dry Bean Dataset 多分类机器学习工程

本项目基于教师提供的 Dry Bean Dataset 脏训练集、验证集和测试集，完成从数据分析、数据清洗、特征工程、多算法实验、鲁棒性分析到静态网页展示的完整机器学习工程流程。

## 项目亮点

- 数据分析：输出类别分布、缺失/非法值统计、特征直方图、相关性热力图、PCA 二维投影。
- 数据处理：统一列名、数值强制转换、缺失值中位数填充、重复样本删除、类别标签清洗、特征标准化。
- 多算法对比：RandomForest、MLP、XGBoost，其中 XGBoost 作为课堂外补充算法。
- 实验维度：测试集准确率、宏平均 F1、loss 曲线、推理速度、鲁棒性、过拟合差距、混淆矩阵。
- 系统展示：`docs/index.html` 为可直接部署到 GitHub Pages 的静态展示页面。

## 目录结构

```text
ml_project/
  src/
    data_processing.py        # 数据加载、清洗、划分、标准化、噪声注入
    run_experiments.py        # 一键运行 EDA、训练、评估、图表和网页生成
    evaluate.py               # 评估、loss 绘图、推理计时
    models/core.py            # RandomForest、MLP、XGBoost 模型定义
  experiments/full_run/       # 实验表格、模型文件、图表和分类报告
  docs/                       # GitHub Pages 静态展示页面
  report/report.md            # 课程论文 Markdown 版
  tests/                      # 数据清洗单元测试
```

## 快速运行

在 `DryBeanDataset/ml_project` 目录下执行：

```bash
pip install -r ../../requirements.txt
python -m src.run_experiments --data-dir .. --out experiments/full_run --site-dir docs
```

运行结束后会生成：

- `experiments/full_run/summary.csv`：主实验指标表。
- `experiments/full_run/robustness.csv`：两类噪声鲁棒性实验表。
- `experiments/full_run/*.png`：精度、速度、loss、鲁棒性、混淆矩阵等图表。
- `docs/index.html`：展示页面，上传 GitHub 后可用 GitHub Pages 访问。

## 当前实验结果

| 模型 | 训练准确率 | 测试准确率 | 过拟合差距 | 推理耗时 ms/样本 | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| RandomForest | 1.0000 | 0.9236 | 0.0764 | 0.0358 | 0.9330 |
| MLP | 0.9458 | 0.9266 | 0.0193 | 0.0016 | 0.9352 |
| XGBoost | 1.0000 | 0.9207 | 0.0793 | 0.0049 | 0.9334 |

结论简述：MLP 在测试集准确率、宏平均 F1、推理速度和过拟合控制上表现最好；RandomForest 和 XGBoost 训练集准确率达到 1.0，但测试集低于 MLP，存在更明显的过拟合迹象。

## GitHub Pages 展示

将本项目上传到 GitHub 后，在仓库设置中启用 Pages，Source 选择 `docs/` 目录，即可得到展示链接：

```text
https://<你的用户名>.github.io/<仓库名>/
```

本地也可直接打开：

```text
DryBeanDataset/ml_project/docs/index.html
```
