#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 17:08:31 2025

@author: chenjingqi
"""

#半监督学习
#示例（标签传播）：
# 步骤 1: 导入必要库
import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 步骤 2: 准备数据（生成一个环形数据集，并只给少量点标注）
X, y = datasets.make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=42)

# 模拟真实场景：只有2个点有标签，其余998个点标签未知（标记为-1）
rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(len(y)) < 0.998 # 随机选择998个点
y_train = np.copy(y)
y_train[random_unlabeled_points] = -1 # 将未标注点的标签设置为-1

print(f"有标签的样本数量: {len(y_train[y_train != -1])}") # 查看真正有标签的样本数

# 步骤 3 & 4: 选择模型并进行训练
label_prop_model = LabelPropagation(kernel='knn', n_neighbors=10)
label_prop_model.fit(X, y_train)

# 步骤 5: 模型评估（与所有真实标签比较）
y_pred = label_prop_model.transduction_ # 模型推断出的所有标签
accuracy = accuracy_score(y, y_pred)
print(f"半监督学习 - 在所有数据上的准确率: {accuracy:.2f}")

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis', alpha=0.7)
plt.show()