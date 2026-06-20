"""
随机森林完整建模管线
作者: [你的名字]
日期: 2024-01-XX
用途: Week 2 (Day 8-14) 的完整流程整合

使用示例:
    from random_forest_pipeline import RandomForestPipeline
    
    pipeline = RandomForestPipeline('data/dielectric_cleaned.csv')
    pipeline.run()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import joblib
import json
import os
os.chdir('D:/MY_Learning/matbench-dielectric/notebooks')
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')


class RandomForestPipeline:
    """
    随机森林完整建模管线
    
    主要功能:
    1. 数据加载和预处理
    2. 超参数自动调优（含 OOB 评估）
    3. 特征重要性分析（三种方法对比）
    4. 部分依赖图（PDP）
    5. 结果可视化和保存
    """
    
    def __init__(self, data_path, target_column='n_dielectric', 
                 test_size=0.2, random_state=42):
        """
        初始化管线
        
        参数:
            data_path (str): 数据文件路径
            target_column (str): 目标变量列名
            test_size (float): 测试集比例
            random_state (int): 随机种子
        """
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        
        # 初始化属性
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.best_params = None
        self.grid_search = None
        
        # 特征重要性结果
        self.importance_builtin = None
        self.importance_permutation = None
        self.importance_comparison = None
        
        # 评估结果
        self.results = {}
        
        # 创建输出目录
        os.makedirs('../figures', exist_ok=True)
        os.makedirs('../models', exist_ok=True)
        
        print("="*70)
        print("随机森林建模管线已初始化")
        print("="*70)
    
    
    def load_data(self):
        """
        加载数据并划分训练集/测试集
        """
        print("\n[1/8] 加载数据...")
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"  ✓ 数据加载成功: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")
        except Exception as e:
            raise Exception(f"数据加载失败: {str(e)}")
        
        # 分离特征和目标
        if self.target_column not in self.df.columns:
            raise ValueError(f"目标列 '{self.target_column}' 不在数据中")
        
        self.X = self.df.drop(self.target_column, axis=1)
        self.y = np.log1p(self.df[self.target_column])
        
        print(f"  ✓ 特征数: {self.X.shape[1]}")
        print(f"  ✓ 目标变量范围: [{self.y.min():.2f}, {self.y.max():.2f}]")
        
        # 划分数据
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        print(f"  ✓ 训练集: {self.X_train.shape[0]} 样本")
        print(f"  ✓ 测试集: {self.X_test.shape[0]} 样本")
    
    
    def tune_hyperparameters(self, param_grid=None, cv=3, use_oob=False):
        """
        超参数调优
        
        参数:
            param_grid (dict): 参数网格，默认使用预设
            cv (int): 交叉验证折数
            use_oob (bool): 是否使用 OOB 快速调参
        """
        print("\n[2/8] 超参数调优...")
        
        # 默认参数网格
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [15, 20, None],
                'max_features': ['sqrt', 0.5],
                'min_samples_leaf': [1, 5]
            }
        
        n_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"  ✓ 参数组合数: {n_combinations}")
        print(f"  ✓ 交叉验证折数: {cv}")
        print(f"  ✓ 总计训练模型数: {n_combinations * cv}")
        
        if use_oob:
            print("\n  → 使用 OOB 快速调参...")
            self._tune_with_oob(param_grid)
        else:
            print("\n  → 使用网格搜索 + 交叉验证...")
            self._tune_with_cv(param_grid, cv)
    
    
    def _tune_with_cv(self, param_grid, cv):
        """使用交叉验证调参"""
        self.grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=self.random_state, n_jobs=1),
            param_grid=param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=-1,         # GridSearchCV 并行
            verbose=1
        )
        
        self.grid_search.fit(self.X_train, self.y_train)
        
        self.model = self.grid_search.best_estimator_
        self.best_params = self.grid_search.best_params_
        
        print(f"\n  ✓ 最佳参数: {self.best_params}")
        print(f"  ✓ 最佳交叉验证 R²: {self.grid_search.best_score_:.4f}")
        
        self.results['tuning'] = {
            'method': 'GridSearchCV',
            'best_params': self.best_params,
            'best_cv_score': float(self.grid_search.best_score_)
        }
    
    
    def _tune_with_oob(self, param_grid):
        """使用 OOB 快速调参"""
        from itertools import product
        
        # 生成所有参数组合
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        best_oob_score = -np.inf
        best_params = None
        
        for params in combinations:
            rf = RandomForestRegressor(**params, oob_score=True, 
                                       random_state=self.random_state, n_jobs=-1)
            rf.fit(self.X_train, self.y_train)
            
            if rf.oob_score_ > best_oob_score:
                best_oob_score = rf.oob_score_
                best_params = params
        
        self.best_params = best_params
        self.model = RandomForestRegressor(**best_params, oob_score=True,
                                           random_state=self.random_state, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)
        
        print(f"\n  ✓ 最佳参数: {self.best_params}")
        print(f"  ✓ 最佳 OOB R²: {best_oob_score:.4f}")
        
        self.results['tuning'] = {
            'method': 'OOB',
            'best_params': self.best_params,
            'best_oob_score': float(best_oob_score)
        }
    
    
    def train(self, params=None):
        """
        训练模型（如果已调参则跳过）
        
        参数:
            params (dict): 模型参数
        """
        if self.model is not None:
            print("\n[3/8] 使用调优后的模型...")
            return
        
        print("\n[3/8] 训练模型...")
        
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 20,
                'max_features': 'sqrt',
                'oob_score': True,
                'random_state': self.random_state,
                'n_jobs': -1
            }
        
        self.model = RandomForestRegressor(**params)
        self.model.fit(self.X_train, self.y_train)
        self.best_params = params
        
        print(f"  ✓ 模型训练完成")
    
    
    def evaluate(self):
        """评估模型性能"""
        print("\n[4/8] 评估模型...")
        
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 训练集评估
        y_train_pred = self.model.predict(self.X_train)
        train_r2 = r2_score(self.y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        
        # 测试集评估
        y_test_pred = self.model.predict(self.X_test)
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        # OOB 评估（如果可用）
        oob_r2 = self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None
        
        print(f"\n  训练集:")
        print(f"    R²:   {train_r2:.4f}")
        print(f"    RMSE: {train_rmse:.4f}")
        
        if oob_r2 is not None:
            print(f"\n  OOB:")
            print(f"    R²:   {oob_r2:.4f}")
        
        print(f"\n  测试集:")
        print(f"    R²:   {test_r2:.4f}")
        print(f"    RMSE: {test_rmse:.4f}")
        
        # 过拟合检查
        gap = train_r2 - test_r2
        print(f"\n  过拟合分析:")
        print(f"    R² 差距: {gap:.4f}")
        if gap < 0.05:
            print(f"    状态: ✓ 泛化能力良好")
        elif gap < 0.10:
            print(f"    状态: ⚠ 轻微过拟合")
        else:
            print(f"    状态: ✗ 过拟合")
        
        # 保存结果
        self.results['performance'] = {
            'train_r2': float(train_r2),
            'train_rmse': float(train_rmse),
            'test_r2': float(test_r2),
            'test_rmse': float(test_rmse),
            'oob_r2': float(oob_r2) if oob_r2 is not None else None,
            'overfitting_gap': float(gap)
        }
        
        return test_r2
    
    
    def analyze_importance(self, use_permutation=True):
        """
        特征重要性分析
        
        参数:
            use_permutation (bool): 是否计算 Permutation Importance
        """
        print("\n[5/8] 特征重要性分析...")
        
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 内置重要性
        self.importance_builtin = pd.DataFrame({
            'feature': self.X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        print(f"\n  Top 5 特征（内置方法）:")
        for i, row in self.importance_builtin.head(5).iterrows():
            print(f"    {i+1}. {row['feature']:30s} {row['importance']:.4f}")
        
        # Permutation Importance
        if use_permutation:
            print(f"\n  计算 Permutation Importance（可能需要 1-2 分钟）...")
            perm_imp = permutation_importance(
                self.model, self.X_test, self.y_test,
                n_repeats=10, random_state=self.random_state, n_jobs=-1
            )
            
            self.importance_permutation = pd.DataFrame({
                'feature': self.X.columns,
                'importance_mean': perm_imp.importances_mean,
                'importance_std': perm_imp.importances_std
            }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
            
            print(f"\n  Top 5 特征（Permutation）:")
            for i, row in self.importance_permutation.head(5).iterrows():
                print(f"    {i+1}. {row['feature']:30s} "
                      f"{row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
            
            # 对比
            self.importance_comparison = self.importance_builtin.merge(
                self.importance_permutation[['feature', 'importance_mean']], 
                on='feature'
            ).rename(columns={'importance': 'builtin', 'importance_mean': 'permutation'})
            
            # 保存
            self.results['feature_importance'] = {
                'top_5_builtin': self.importance_builtin.head(5).to_dict('records'),
                'top_5_permutation': self.importance_permutation.head(5).to_dict('records')
            }
    
    
    def plot_importance(self, top_n=15, save=True):
        """绘制特征重要性对比图"""
        if self.importance_builtin is None:
            print("  ⚠️  请先运行 analyze_importance()")
            return
        
        print("\n[6/8] 可视化特征重要性...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左图：内置重要性
        ax1 = axes[0]
        top_builtin = self.importance_builtin.head(top_n)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_builtin)))
        ax1.barh(range(len(top_builtin)), top_builtin['importance'], 
                 color=colors, edgecolor='black')
        # 截断长特征名
        short_names = [f[:40] + '...' if len(f) > 40 else f for f in top_builtin['feature']]
        ax1.set_yticks(range(len(top_builtin)))
        ax1.set_yticklabels(short_names, fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax1.set_title('Feature Importance (Built-in)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # 右图：Permutation（如果有）
        if self.importance_permutation is not None:
            ax2 = axes[1]
            top_perm = self.importance_permutation.head(top_n)
            colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_perm)))
            ax2.barh(range(len(top_perm)), top_perm['importance_mean'],
                     xerr=top_perm['importance_std'],
                     color=colors, edgecolor='black', capsize=5)
            short_perm = [f[:40] + '...' if len(f) > 40 else f for f in top_perm['feature']]
            ax2.set_yticks(range(len(top_perm)))
            ax2.set_yticklabels(short_perm, fontsize=9)
            ax2.invert_yaxis()
            ax2.set_xlabel('Importance (Mean ± Std)', fontsize=12, fontweight='bold')
            ax2.set_title('Feature Importance (Permutation)', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
        else:
            axes[1].text(0.5, 0.5, 'Permutation Importance\nNot Calculated', 
                        ha='center', va='center', fontsize=14)
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f'../figures/rf_importance_{timestamp}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  ✓ 特征重要性图已保存: {filepath}")
        
        plt.show()
    
    
    def plot_pdp(self, top_n=3, save=True):
        """
        绘制部分依赖图（Partial Dependence Plot）
        
        参数:
            top_n (int): 绘制前 N 个重要特征的 PDP
            save (bool): 是否保存图片
        """
        print("\n[7/8] 绘制部分依赖图...")
        
        if self.importance_builtin is None:
            print("  ⚠️  请先运行 analyze_importance()")
            return
        
        # 选择 Top N 特征
        top_features = self.importance_builtin.head(top_n)['feature'].tolist()
        feature_indices = [list(self.X.columns).index(f) for f in top_features]
        
        print(f"  → 计算前 {top_n} 个特征的 PDP...")
        
        fig, ax = plt.subplots(figsize=(15, 5))
        
        display = PartialDependenceDisplay.from_estimator(
            self.model,
            self.X_train,
            features=feature_indices,
            feature_names=self.X.columns,
            n_cols=top_n,
            grid_resolution=50,
            ax=ax
        )
        
        fig.suptitle(f'Partial Dependence Plots (Top {top_n} Features)', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f'../figures/rf_pdp_{timestamp}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  ✓ PDP 已保存: {filepath}")
        
        plt.show()
    
    
    def plot_results(self, save=True):
        """综合可视化"""
        print("\n绘制综合结果图...")
        
        fig = plt.figure(figsize=(16, 10))
        
        # 子图 1：真实值 vs 预测值
        ax1 = plt.subplot(2, 3, 1)
        y_pred = self.model.predict(self.X_test)
        ax1.scatter(self.y_test, y_pred, alpha=0.5, s=30)
        ax1.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('True Values', fontweight='bold')
        ax1.set_ylabel('Predictions', fontweight='bold')
        ax1.set_title('Predictions vs True Values', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 子图 2：残差分布
        ax2 = plt.subplot(2, 3, 2)
        residuals = self.y_test - y_pred
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Residual Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 子图 3：残差 vs 预测值
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(y_pred, residuals, alpha=0.5, s=30)
        ax3.axhline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Predicted Values', fontweight='bold')
        ax3.set_ylabel('Residuals', fontweight='bold')
        ax3.set_title('Residual Plot', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 子图 4：特征重要性（Top 10）
        ax4 = plt.subplot(2, 3, 4)
        if self.importance_builtin is not None:
            top10 = self.importance_builtin.head(10)
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top10)))
            ax4.barh(range(len(top10)), top10['importance'], color=colors, edgecolor='black')
            short_f = [f[:35] + '...' if len(f) > 35 else f for f in top10['feature']]
            ax4.set_yticks(range(len(top10)))
            ax4.set_yticklabels(short_f, fontsize=8)
            ax4.set_xlabel('Importance', fontweight='bold')
            ax4.set_title('Top 10 Feature Importance', fontweight='bold')
            ax4.invert_yaxis()
            ax4.grid(True, alpha=0.3, axis='x')
        
        # 子图 5：树的数量（如果有网格搜索结果）
        ax5 = plt.subplot(2, 3, 5)
        if self.grid_search is not None:
            results_df = pd.DataFrame(self.grid_search.cv_results_)
            if 'param_n_estimators' in results_df.columns:
                grouped = results_df.groupby('param_n_estimators')['mean_test_score'].mean()
                ax5.plot(grouped.index, grouped.values, marker='o', linewidth=2)
                ax5.set_xlabel('n_estimators', fontweight='bold')
                ax5.set_ylabel('Mean CV R²', fontweight='bold')
                ax5.set_title('Performance vs n_estimators', fontweight='bold')
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'Grid Search\nNot Available', 
                        ha='center', va='center', fontsize=12)
                ax5.axis('off')
        else:
            ax5.text(0.5, 0.5, 'Grid Search\nNot Performed', 
                    ha='center', va='center', fontsize=12)
            ax5.axis('off')
        
        # 子图 6：性能指标对比
        ax6 = plt.subplot(2, 3, 6)
        if 'performance' in self.results:
            metrics = ['R²', 'RMSE']
            train_vals = [
                self.results['performance']['train_r2'],
                self.results['performance']['train_rmse']
            ]
            test_vals = [
                self.results['performance']['test_r2'],
                self.results['performance']['test_rmse']
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            ax6.bar(x - width/2, train_vals, width, label='Train', color='skyblue', edgecolor='black')
            ax6.bar(x + width/2, test_vals, width, label='Test', color='lightcoral', edgecolor='black')
            ax6.set_xticks(x)
            ax6.set_xticklabels(metrics)
            ax6.set_ylabel('Score', fontweight='bold')
            ax6.set_title('Performance Metrics', fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f'../figures/rf_results_{timestamp}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  ✓ 综合结果图已保存: {filepath}")
        
        plt.show()
    
    
    def save_model(self, filename=None):
        """保存模型和结果"""
        if self.model is None:
            raise ValueError("模型尚未训练")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存模型
        if filename is None:
            filename = f'random_forest_model_{timestamp}.pkl'

        model_path = f'../models/{filename}'
        joblib.dump(self.model, model_path)
        print(f"\n  Model saved: {model_path}")

        # 转换 numpy 类型
        def to_python(val):
            if isinstance(val, (int, np.integer)): return int(val)
            if isinstance(val, (float, np.floating)): return float(val)
            if isinstance(val, dict): return {k: to_python(v) for k, v in val.items()}
            if isinstance(val, list): return [to_python(v) for v in val]
            return str(val)

        # 保存结果 JSON
        results_path = f'../models/rf_results_{timestamp}.json'
        self.results['timestamp'] = timestamp
        self.results['model_params'] = to_python(self.best_params) if self.best_params else None

        with open(results_path, 'w') as f:
            json.dump(to_python(self.results), f, indent=2)
        print(f"  Results saved: {results_path}")

        # 保存特征重要性
        if self.importance_builtin is not None:
            imp_path = f'../models/rf_importance_{timestamp}.csv'
            if self.importance_permutation is not None:
                self.importance_comparison.to_csv(imp_path, index=False)
            else:
                self.importance_builtin.to_csv(imp_path, index=False)
            print(f"  Importance saved: {imp_path}")

        return model_path
    
    
    def run(self, tune=True, use_oob=False, plot_pdp=True, save=True):
        """
        运行完整流程
        
        参数:
            tune (bool): 是否调参
            use_oob (bool): 是否用 OOB 快速调参
            plot_pdp (bool): 是否绘制 PDP
            save (bool): 是否保存模型和结果
        """
        print("\n" + "="*70)
        print("开始运行随机森林建模完整流程")
        print("="*70)
        
        self.load_data()
        
        if tune:
            self.tune_hyperparameters(use_oob=use_oob)
        else:
            self.train()
        
        test_r2 = self.evaluate()
        self.analyze_importance(use_permutation=True)
        self.plot_importance(save=save)
        
        if plot_pdp:
            self.plot_pdp(save=save)
        
        self.plot_results(save=save)
        
        if save:
            self.save_model()
        
        print("\n" + "="*70)
        print(f"✓ 流程完成！测试集 R² = {test_r2:.4f}")
        print("="*70 + "\n")
        
        return self.model


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建管线实例
    pipeline = RandomForestPipeline(
        data_path='../data/dielectric_cleaned.csv',
        target_column='n_dielectric',
        test_size=0.2,
        random_state=42
    )
    
    # 运行完整流程
    # 选项 1：完整流程（包括调参）
    model = pipeline.run(tune=True, use_oob=False, plot_pdp=True, save=True)
    
    # 选项 2：快速模式（用 OOB 调参）
    # model = pipeline.run(tune=True, use_oob=True, plot_pdp=False, save=True)
    
    # 选项 3：不调参，用默认参数
    # model = pipeline.run(tune=False, plot_pdp=False, save=False)
    
    # 也可以分步运行：
    # pipeline.load_data()
    # pipeline.tune_hyperparameters(use_oob=True)
    # pipeline.evaluate()
    # pipeline.analyze_importance()
    # pipeline.plot_importance()
    # pipeline.save_model()