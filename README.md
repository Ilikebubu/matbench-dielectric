# 材料介电常数预测 — ML 管线

Matbench 介电常数数据集 | XGBoost + SHAP + GPR

## 项目结构

```
matbench-dielectric/
├── data/
│   └── dielectric_cleaned.csv    # 特征矩阵 + 目标变量
├── notebooks/                     # 阶段二的每日 notebook
├── src/                           # 可复用 Python 脚本
├── figures/                       # 输出图片
├── models/                        # 训练好的模型文件
└── README.md
```

## 数据来源

Matbench v0.1 dielectric benchmark（材料科学标准基准），4764 种材料。
目标变量：`n_dielectric`（折射率相关介电常数）。
特征：用 matminer 从晶体结构提取的 ~68 个元素属性 + 密度特征。

## 进度

- [x] 阶段一：EDA
- [ ] 阶段二：模型实战（进行中）
- [ ] 阶段三：项目整合
- [ ] 阶段四：面试准备

## 环境

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap matminer joblib
```
