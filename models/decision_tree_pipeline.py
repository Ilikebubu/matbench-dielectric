"""
决策树完整建模管线
作者: ilikebubu 
日期: 2026-06-16
用途: 快速训练和评估决策树模型

使用示例:
    python decision_tree_pipeline.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
os.chdir('D:/MY_Learning/matbench-dielectric/notebooks')


class DecisionTreePipeline:
    """
    决策树建模完整流程类
    
    主要功能:
    1. 数据加载和划分
    2. 超参数自动调优
    3. 模型训练和评估
    4. 特征重要性分析
    5. 结果可视化和保存
    """
    
    def __init__(self, data_path, target_column='n_dielectric', test_size=0.2, random_state=42):
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
        self.feature_importance = None
        self.results = {}
        
        # 创建输出目录
        os.makedirs('../figures', exist_ok=True)
        os.makedirs('../models', exist_ok=True)
        
        print("="*70)
        print("决策树建模管线已初始化")
        print("="*70)
    
    
    def load_data(self):
        """
        加载数据并划分训练集/测试集
        """
        print("\n[1/7] 加载数据...")
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"  ✓ 数据加载成功: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到文件: {self.data_path}")
        except Exception as e:
            raise Exception(f"数据加载失败: {str(e)}")
        
        # 分离特征和目标
        if self.target_column not in self.df.columns:
            raise ValueError(f"目标列 '{self.target_column}' 不在数据中")
        
        self.X = self.df.drop(self.target_column, axis=1)
        self.y = np.log1p(self.df[self.target_column])
        
        print(f"  ✓ 特征数: {self.X.shape[1]}")
        print(f"  ✓ 目标变量: {self.target_column}")
        print(f"  ✓ 目标范围: [{self.y.min():.2f}, {self.y.max():.2f}]")
        
        # 划分数据
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        print(f"  ✓ 训练集: {self.X_train.shape[0]} 样本")
        print(f"  ✓ 测试集: {self.X_test.shape[0]} 样本")
    
    
    def explore_data(self):
        """
        快速数据探索（可选）
        """
        print("\n[2/7] 数据探索...")
        
        # 检查缺失值
        missing = self.df.isnull().sum().sum()
        if missing > 0:
            print(f"  ⚠ 警告: 发现 {missing} 个缺失值")
        else:
            print("  ✓ 无缺失值")
        
        # 检查特征类型
        numeric_features = self.X.select_dtypes(include=[np.number]).columns
        print(f"  ✓ 数值型特征: {len(numeric_features)} 个")
        
        # 保存基本统计
        self.results['data_summary'] = {
            'n_samples': len(self.df),
            'n_features': len(self.X.columns),
            'n_missing': int(missing),
            'target_mean': float(self.y.mean()),
            'target_std': float(self.y.std())
        }
    
    
    def tune_hyperparameters(self, param_grid=None, cv=5, verbose=1):
        """
        超参数调优（网格搜索 + 交叉验证）
        
        参数:
            param_grid (dict): 参数网格，默认为 None（使用预设）
            cv (int): 交叉验证折数
            verbose (int): 显示详细程度
        """
        print("\n[3/7] 超参数调优...")
        
        # 默认参数网格
        if param_grid is None:
            param_grid = {
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [10, 20, 50],
                'min_samples_leaf': [5, 10, 20]
            }
        
        n_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"  ✓ 参数组合数: {n_combinations}")
        print(f"  ✓ 交叉验证折数: {cv}")
        print(f"  ✓ 总计训练模型数: {n_combinations * cv}")
        
        # 网格搜索
        dt = DecisionTreeRegressor(random_state=self.random_state)
        grid_search = GridSearchCV(
            estimator=dt,
            param_grid=param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=verbose
        )
        
        print("  → 开始网格搜索...")
        grid_search.fit(self.X_train, self.y_train)
        
        # 保存最佳模型和参数
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"\n  ✓ 最佳参数: {self.best_params}")
        print(f"  ✓ 最佳交叉验证 R²: {grid_search.best_score_:.4f}")
        
        # 保存网格搜索结果
        self.results['grid_search'] = {
            'best_params': {k: int(v) for k, v in grid_search.best_params_.items()},
            'best_cv_score': float(grid_search.best_score_),
            'n_combinations_tried': int(n_combinations)
        }
    
    
    def train(self, params=None):
        """
        训练模型（如果已经调参，则跳过；否则用默认参数）
        
        参数:
            params (dict): 模型参数，默认为 None
        """
        if self.model is not None:
            print("\n[4/7] 使用调优后的模型...")
            return
        
        print("\n[4/7] 训练模型...")
        
        if params is None:
            params = {'max_depth': 10, 'min_samples_leaf': 5, 'random_state': self.random_state}
        
        self.model = DecisionTreeRegressor(**params)
        self.model.fit(self.X_train, self.y_train)
        self.best_params = params
        
        print(f"  ✓ 模型训练完成")
        print(f"  ✓ 参数: {params}")
    
    
    def evaluate(self):
        """
        评估模型性能
        """
        print("\n[5/7] 评估模型...")
        
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 train() 或 tune_hyperparameters()")
        
        # 训练集评估
        y_train_pred = self.model.predict(self.X_train)
        train_r2 = r2_score(self.y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        
        # 测试集评估
        y_test_pred = self.model.predict(self.X_test)
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        # 打印结果
        print("\n  训练集:")
        print(f"    R²:   {train_r2:.4f}")
        print(f"    RMSE: {train_rmse:.4f}")
        print(f"    MAE:  {train_mae:.4f}")
        
        print("\n  测试集:")
        print(f"    R²:   {test_r2:.4f}")
        print(f"    RMSE: {test_rmse:.4f}")
        print(f"    MAE:  {test_mae:.4f}")
        
        # 过拟合检查
        gap = train_r2 - test_r2
        print(f"\n  过拟合分析:")
        print(f"    R² 差距: {gap:.4f}")
        if gap < 0.05:
            print(f"    状态: ✓ 泛化能力良好")
        elif gap < 0.15:
            print(f"    状态: ⚠ 轻微过拟合")
        else:
            print(f"    状态: ✗ 严重过拟合")
        
        # 保存结果
        self.results['performance'] = {
            'train_r2': float(train_r2),
            'train_rmse': float(train_rmse),
            'train_mae': float(train_mae),
            'test_r2': float(test_r2),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'overfitting_gap': float(gap)
        }
        
        return test_r2
    
    
    def analyze_features(self, top_n=10):
        """
        分析特征重要性
        
        参数:
            top_n (int): 显示前 N 个重要特征
        """
        print(f"\n[6/7] 特征重要性分析...")
        
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 获取特征重要性
        importances = self.model.feature_importances_
        self.feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        print(f"\n  Top {top_n} 重要特征:\n")
        for i, row in self.feature_importance.head(top_n).iterrows():
            print(f"    {i+1:2d}. {row['feature']:30s} {row['importance']:.4f} ({row['importance']*100:.1f}%)")
        
        # 保存特征重要性
        self.results['feature_importance'] = {
            'top_features': self.feature_importance.head(top_n).to_dict('records')
        }
    
    
    def plot_results(self, save=True):
        """
        可视化结果
        
        参数:
            save (bool): 是否保存图片
        """
        print("\n[7/7] 可视化结果...")
        
        fig = plt.figure(figsize=(16, 10))
        
        # 子图 1: 真实值 vs 预测值
        ax1 = plt.subplot(2, 3, 1)
        y_pred = self.model.predict(self.X_test)
        ax1.scatter(self.y_test, y_pred, alpha=0.5, s=30)
        ax1.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 
                 'r--', lw=2)
        ax1.set_xlabel('True Values', fontweight='bold')
        ax1.set_ylabel('Predictions', fontweight='bold')
        ax1.set_title('Predictions vs True Values', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 子图 2: 残差分布
        ax2 = plt.subplot(2, 3, 2)
        residuals = self.y_test - y_pred
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Residual Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 子图 3: 残差 vs 预测值
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(y_pred, residuals, alpha=0.5, s=30)
        ax3.axhline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Predicted Values', fontweight='bold')
        ax3.set_ylabel('Residuals', fontweight='bold')
        ax3.set_title('Residual Plot', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 子图 4: 特征重要性（Top 10）
        ax4 = plt.subplot(2, 3, 4)
        top10 = self.feature_importance.head(10)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top10)))
        ax4.barh(range(len(top10)), top10['importance'], color=colors, edgecolor='black')
        ax4.set_yticks(range(len(top10)))
        ax4.set_yticklabels(top10['feature'])
        ax4.set_xlabel('Importance', fontweight='bold')
        ax4.set_title('Top 10 Feature Importance', fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 子图 5: 决策树结构（前 3 层）
        ax5 = plt.subplot(2, 3, 5)
        plot_tree(self.model, max_depth=3, filled=True, 
                  feature_names=self.X.columns, 
                   fontsize=8)
        ax5.set_title('Decision Tree Structure (3 levels)', fontweight='bold')
        
        # 子图 6: 性能指标对比
        ax6 = plt.subplot(2, 3, 6)
        metrics = ['R²', 'RMSE', 'MAE']
        train_vals = [
            self.results['performance']['train_r2'],
            self.results['performance']['train_rmse'],
            self.results['performance']['train_mae']
        ]
        test_vals = [
            self.results['performance']['test_r2'],
            self.results['performance']['test_rmse'],
            self.results['performance']['test_mae']
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
            filepath = f'../figures/decision_tree_results_{timestamp}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  ✓ 结果图已保存: {filepath}")
        
        plt.show()
    
    
    def save_model(self, filename=None):
        """
        保存模型和结果
        
        参数:
            filename (str): 模型文件名，默认为带时间戳的名称
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        if filename is None:
            filename = f'decision_tree_model_{timestamp}.pkl'
        
        model_path = f'../models/{filename}'
        joblib.dump(self.model, model_path)
        print(f"\n  ✓ 模型已保存: {model_path}")
        
        # 保存结果 JSON
        results_path = f'../models/results_{timestamp}.json'
        self.results['model_params'] = self.best_params
        self.results['timestamp'] = timestamp
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"  ✓ 结果已保存: {results_path}")
        
        return model_path
    
    
    def run(self, tune=True, plot=True, save=True):
        """
        运行完整流程（一键执行）
        
        参数:
            tune (bool): 是否进行超参数调优
            plot (bool): 是否绘图
            save (bool): 是否保存模型
        """
        print("\n" + "="*70)
        print("开始运行决策树建模完整流程")
        print("="*70)
        
        self.load_data()
        self.explore_data()
        
        if tune:
            self.tune_hyperparameters()
        else:
            self.train()
        
        test_r2 = self.evaluate()
        self.analyze_features()
        
        if plot:
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
    pipeline = DecisionTreePipeline(
        data_path='../data/dielectric_cleaned.csv',
        target_column='n_dielectric',
        test_size=0.2,
        random_state=42
    )
    
    # 运行完整流程
    model = pipeline.run(tune=True, plot=True, save=True)
    
    # 也可以分步运行：
    # pipeline.load_data()
    # pipeline.tune_hyperparameters()
    # pipeline.evaluate()
    # pipeline.analyze_features()
    # pipeline.plot_results()
    # pipeline.save_model()