#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 16:45:04 2025

@author: chenjingqi
"""

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

# 4. 创建降维后的DataFrame用于可视化
lda_columns = [f'LDA{i+1}' for i in range(X_lda.shape[1])]
df_lda = pd.DataFrame(X_lda, columns=lda_columns)
df_lda['Target'] = y
df_lda['Target_Name'] = df_lda['Target'].map({i: name for i, name in enumerate(target_names)})

# 5. 可视化LDA结果
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_lda, x='LDA1', y='LDA2', hue='Target_Name', palette='viridis', s=100, alpha=0.8)
plt.title('LDA投影: 鸢尾花数据集 (4D -> 2D)')
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.legend(title='Species')
plt.grid(True)
plt.show()

# 6. 与PCA结果对比 (强化理解)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 创建对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# PCA图
scatter_pca = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8)
ax1.set_title(f'PCA投影 (总方差: {pca.explained_variance_ratio_.sum()*100:.2f}%)')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
ax1.grid(True)
legend1 = ax1.legend(*scatter_pca.legend_elements(), title='Species')
ax1.add_artist(legend1)

# LDA图
scatter_lda = ax2.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis', alpha=0.8)
ax2.set_title('LDA投影 (有监督)')
ax2.set_xlabel('LD1')
ax2.set_ylabel('LD2')
ax2.grid(True)
legend2 = ax2.legend(*scatter_lda.legend_elements(), title='Species')
ax2.add_artist(legend2)

plt.tight_layout()
plt.show()

# 7. 【高级】使用LDA作为分类器的预处理步骤，并评估性能
# 这是一个更贴近真实工程的用例

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 方法A: 直接使用原始特征（4维）进行分类
lr_raw = LogisticRegression(random_state=42)
lr_raw.fit(X_train, y_train)
y_pred_raw = lr_raw.predict(X_test)
accuracy_raw = accuracy_score(y_test, y_pred_raw)

# 方法B: 先使用LDA降维（2维），再用分类器
# 重要：LDA模型必须在训练集上拟合，然后转换训练集和测试集
lda_for_clf = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda_for_clf.fit_transform(X_train, y_train)
X_test_lda = lda_for_clf.transform(X_test) # 注意：对测试集只进行transform，不要fit！

lr_lda = LogisticRegression(random_state=42)
lr_lda.fit(X_train_lda, y_train)
y_pred_lda = lr_lda.predict(X_test_lda)
accuracy_lda = accuracy_score(y_test, y_pred_lda)

print(f"\n分类器性能对比:")
print(f"原始特征 (4D) 准确率: {accuracy_raw:.4f}")
print(f"LDA降维后特征 (2D) 准确率: {accuracy_lda:.4f}")

# 分析：LDA降维后虽然特征更少，但准确率很可能相近甚至更高，因为它去除了对分类无用的噪声，突出了判别信息。


'''与PCA对比：
对比图非常能说明问题。PCA的投影可能保留了更多的全局方差，但三个类别的点可能会混在一起。而LDA的投影牺牲了全局方差，却换来了极佳的类别分离度。这张图完美诠释了无监督学习和有监督学习的根本差异。
工程应用：作为预处理提升分类性能：
这是LDA在真实项目中最主要的用途。代码演示了一个完整的流程：
将数据分为训练集和测试集。
在训练集上拟合（fit）LDA模型，学习判别方向。
用学习到的LDA模型分别转换（transform）训练集和测试集。
在降维后的低维数据上训练分类器（如逻辑回归）。
结果往往会发现，使用LDA降维到2维后的数据，训练出的分类器性能与使用原始4维数据相当甚至更好。这是因为LDA过滤掉了与分类无关的噪声和冗余信息，只保留了最具有判别力的特征，有时反而能防止过拟合，提升模型的泛化能力。同时，计算效率也因为维度降低而得到提升。'''