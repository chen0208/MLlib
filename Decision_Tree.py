#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 11:16:20 2025

@author: chenjingqi
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# 1. 加载数据
data = pd.read_csv('your_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建模型并设置关键参数
# 这些参数是防止过拟合的关键！
model = DecisionTreeClassifier(
    criterion='gini',     # 分裂准则，可以是 'gini' 或 'entropy'
    max_depth=5,          # 树的最大深度，最重要的调参之一
    min_samples_split=10, # 节点至少包含10个样本才允许分裂
    min_samples_leaf=5,   # 叶子节点至少包含5个样本
    random_state=42       # 固定随机种子，保证结果可复现
)

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 预测与评估
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. （可选）可视化树
from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
tree.plot_tree(model, feature_names=X.columns, class_names=True, filled=True)
plt.show()