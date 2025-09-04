#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 15:51:50 2025

@author: chenjingqi
"""
#监督学习
#示例 （鸢尾花分类 - 分类问题）

# 步骤 1: 导入必要库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 步骤 2: 加载数据 (已标注好的)
iris = load_iris()
X, y = iris.data, iris.target

# 步骤 3: 数据预处理 - 划分训练集和测试集，并进行特征缩放
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # 注意：测试集使用训练集的缩放参数

# 步骤 4 & 5: 选择模型并进行训练
model = LogisticRegression(random_state=42, max_iter=200)
model.fit(X_train_scaled, y_train)

# 步骤 6: 模型评估
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"监督学习 - 测试集准确率: {accuracy:.2f}")

# 步骤 7: (可选) 这里为了简洁省略了超参数调优步骤，通常使用GridSearchCV

# 步骤 8: 预测新样本
new_flower = [[5.1, 3.5, 1.4, 0.2]] # 一个新花的特征
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)
print(f"新样本预测为: {iris.target_names[prediction][0]}")
