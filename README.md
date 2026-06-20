# 材料介电常数预测 — ML 管线

Matbench 介电常数数据集 | 决策树 → 随机森林 → XGBoost → SHAP → GPR

## 学习进度

| 周 | 天 | 内容 | 最佳 R² |
|---|---|---|---|
| Week 1 | Day 1-7 | 决策树（基础→过拟合→调参→特征→管线） | 0.584 |
| Week 2 | Day 8-12 | 随机森林（基础→调参→OOB→Permutation） | 0.704 |
| Week 3 | Day 15-21 | XGBoost（未开始） | — |
| Week 4 | Day 22-28 | SHAP + GPR（未开始） | — |

## 关键里程碑

```
决策树单棵：      R² = 0.58
随机森林 100 棵：  R² = 0.70 (+20%)
随机森林调优后：  R² = 0.70（默认够好）
Permutation 验证：密度 > 电负性（Gini 排名不可全信）
```

## 项目结构

```
matbench-dielectric/
├── data/
│   └── dielectric_cleaned.csv    # 特征矩阵 + 目标变量
├── notebooks/                     # 每日 notebook
│   ├── day01_decision_tree_basic.ipynb
│   ├── day02_overfitting.ipynb
│   ├── day03_pruning.ipynb
│   ├── day04_feature_importance.ipynb
│   ├── day05_feature_selection.ipynb
│   ├── day08_random_forest_basic.ipynb
│   ├── day09_rf_tuning.ipynb
│   ├── day10_oob.ipynb
│   └── day12_permutation_importance.ipynb
├── models/                        # 管线脚本 + 模型文件
│   ├── decision_tree_pipeline.py
│   └── random_forest_pipeline.py
├── figures/                       # 输出图片
└── README.md
```

## 数据来源

Matbench v0.1 dielectric benchmark，4764 种材料。
目标变量：`n_dielectric`（折射率相关介电常数）。
特征：matminer 元素属性 + 密度特征，135 维。

## 环境

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap matminer joblib
```
