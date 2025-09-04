#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 17:38:27 2025

@author: chenjingqi
"""
#经典等回归问题
#随机森林回归算法
#波士顿房价
# 1. 导入必要的库
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing # 波士顿数据集已弃用，改用加州房价
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 2. 加载数据集
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

# 将数据转换为DataFrame以便查看
df = pd.DataFrame(X, columns=feature_names)
df['TargetPrice'] = y
print("数据预览:")
print(df.head())
print(f"\n数据集形状: {X.shape}")

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 选择并训练模型
# 创建随机森林回归模型，设置随机种子以确保结果可复现
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train) # 在训练集上进行训练

# 5. 在测试集上进行预测
y_pred = model.predict(X_test)

# 6. 评估模型性能
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n模型评估结果:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# 7. (可选) 可视化预测结果 vs 真实结果
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # 绘制理想对角线
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs. Predicted House Prices')
plt.show()

# 8. (可选) 分析特征重要性
feature_importance = model.feature_importances_
# 排序并展示
indices = np.argsort(feature_importance)[::-1]
print("\n特征重要性排名:")
for i in range(X.shape[1]):
    print(f"{i+1}. {feature_names[indices[i]]} ({feature_importance[indices[i]]:.4f})")