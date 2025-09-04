#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 17:11:51 2025

@author: chenjingqi
"""

#弱监督学习
#示例（带噪声标签的分类 - 不准确监督）
# 步骤 1: 导入必要库
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# 步骤 2: 生成数据并人为注入标签噪声
X, y_clean = make_classification(n_samples=1000, n_features=20, random_state=42)

# 划分训练测试集
X_train, X_test, y_train_clean, y_test = train_test_split(X, y_clean, test_size=0.3, random_state=42)

# 模拟在20%的训练样本上标签错误
np.random.seed(42)
noise_mask = np.random.rand(len(y_train_clean)) < 0.20 # 20%的训练数据
y_train_noisy = y_train_clean.copy()
y_train_noisy[noise_mask] = 1 - y_train_clean[noise_mask] # 翻转标签（0变1，1变0）

# 步骤 3 & 4: 使用带噪声的标签训练一个标准模型
model_standard = LogisticRegression(random_state=42, max_iter=1000)
model_standard.fit(X_train, y_train_noisy)

# 再训练一个对噪声可能更鲁棒的模型（例如，加强正则化）
# L1正则化可能有助于特征选择，从而在一定程度上抑制噪声影响
model_robust = LogisticRegression(random_state=42, penalty='l1', solver='liblinear', C=0.1, max_iter=1000)
model_robust.fit(X_train, y_train_noisy)

# 步骤 5: 模型评估（在干净的测试集上！）
y_pred_standard = model_standard.predict(X_test)
y_pred_robust = model_robust.predict(X_test)

acc_standard = accuracy_score(y_test, y_pred_standard)
acc_robust = accuracy_score(y_test, y_pred_robust)

print(f"弱监督学习 - 标准模型在干净测试集上的准确率: {acc_standard:.2f}")
print(f"弱监督学习 - 鲁棒模型(L1正则化)在干净测试集上的准确率: {acc_robust:.2f}")
# 通常鲁棒模型的表现会更好，这展示了处理弱监督的一种简单策略