#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 16:39:50 2025

@author: chenjingqi
"""
#我们将继续使用鸢尾花（Iris）数据集，这是一个3分类问题（Setosa, Versicolour, Virginica），包含4个特征。LDA最多可以将其降至2维。

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#数据准备与标准化：
#虽然LDA内部计算会处理散度矩阵，对scale不像PCA那么敏感，但标准化仍然是一个好习惯，可以确保所有特征在同等基础上被评估。
# 1. 加载和准备数据
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
feature_names = iris.feature_names

df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y
df['Target_Name'] = df['Target'].map({i: name for i, name in enumerate(target_names)})

print("原始数据形状:", X.shape)
print("类别标签:", np.unique(y))

# 2. 数据标准化 (虽然不是LDA必须的，但良好的实践习惯)
# LDA本身会处理方差，但标准化可以避免数值范围差异过大的问题
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#执行LDA：
#关键区别：lda.fit_transform(X_scaled, y)。必须传入标签 y，这是LDA“有监督”的核心体现。
#降维后的维度 n_components 不能大于 n_classes - 1。
# 3. 执行LDA
# 我们知道是3分类问题，所以最多只能降到2维 (n_components <= 2)
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y) # 注意！这里传入了标签 y

print(f"\n降维后的数据形状: {X_lda.shape}")
print("LDA模型解释方差比: ", lda.explained_variance_ratio_)
# LDA没有`explained_variance_ratio_`属性，正确的属性是：
# 在scikit-learn中，LDA没有直接提供解释方差比，但我们可以通过其`scalings_`或计算特征值来近似
# 更常见的做法是查看判别能力
print(f"各判别方向的判别能力: {lda.explained_variance_ratio_}")

#可视化：
#将LDA降维后的结果在二维平面画出来。你会看到三个类别被清晰地分在了三个不同的区域，类间距离大，类内聚集度高。这正是LDA优化目标的可视化体现。
#创建降维后的DataFrame用于可视化
lda_columns = [f'LDA{i+1}' for i in range(X_lda.shape[1])]
df_lda = pd.DataFrame(X_lda, columns=lda_columns)
df_lda['Target'] = y
df_lda['Target_Name'] = df_lda['Target'].map({i: name for i, name in enumerate(target_names)})

# 5. 可视化LDA结果
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_lda, x='LDA1', y='LDA2', hue='Target_Name', palette='viridis', s=100, alpha=0.8)
plt.title('LDA: iris data (4D -> 2D)')
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.legend(title='Species')
plt.grid(True)
plt.show()


