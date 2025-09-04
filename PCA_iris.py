#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 15:59:54 2025

@author: chenjingqi
"""
#我们将使用经典的鸢尾花（Iris）数据集进行演示。这个数据集有4个特征（萼片长宽、花瓣长宽），150个样本。我们将把它从4维降到2维。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#数据加载与标准化：
#我们首先加载数据并对其进行标准化。输出会显示标准化后的均值几乎为0，方差为1，确认标准化成功。

# 1. 加载数据
iris = load_iris()
X = iris.data  # 特征矩阵 (150, 4)
y = iris.target # 目标变量（三种鸢尾花）
feature_names = iris.feature_names
target_names = iris.target_names

# 创建一个DataFrame以便于查看
df = pd.DataFrame(X, columns=feature_names)
print("原始数据形状:", X.shape)
print("\n原始数据前5行:")
print(df.head())

# 2. 数据标准化 (至关重要！)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # 转换为均值为0，标准差为1的数据

# 查看标准化后的均值和标准差，确认标准化成功
print(f"\n标准化后均值: {np.mean(X_scaled, axis=0).round(2)}")
print(f"标准化后方差: {np.var(X_scaled, axis=0).round(2)}")

#初步PCA与方差分析：
#我们首次执行PCA时不指定维度，目的是分析所有主成分。
#解释方差比 (Explained Variance Ratio)：每个主成分所保留的原始数据方差的百分比。例如，PC1可能保留了73%的方差。
#累计解释方差比 (Cumulative Explained Variance Ratio)：前K个主成分保留的方差总和。
#碎石图 (Scree Plot)：帮助选择K值的经典工具。我们寻找"拐点"（Elbow），即解释方差突然变小的点。之后的主成分贡献很小，可以舍弃。
#累计方差图：更直观地显示选择K个成分能保留多少信息。工程上常见的阈值是95%。

# 3. 执行PCA
# 首先，我们初始化PCA对象。
# 不指定n_components，用于分析所有成分，以便我们决定降维到几维
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# 4. 分析解释方差比 (Explained Variance Ratio)
# 这是决定降维到多少维（K）的关键指标。
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print("\n每个主成分的解释方差比:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

print("\n累计解释方差比:")
for i, cum_ratio in enumerate(cumulative_variance_ratio):
    print(f"前{i+1}个主成分: {cum_ratio:.4f} ({cum_ratio*100:.2f}%)")

# 绘制碎石图 (Scree Plot) 和累计方差图
plt.figure(figsize=(10, 4))

# 碎石图
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, alpha=0.7, align='center')
plt.step(range(1, len(cumulative_variance_ratio)+1), cumulative_variance_ratio, where='mid')
plt.xlabel('primary count')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.legend(['Cumulative Variance', 'Single Variance'])

# 累计方差图
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance_ratio)+1), cumulative_variance_ratio, 'o-')
plt.xlabel('primary count')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance') # 标记95%的线
plt.legend()

plt.tight_layout()
plt.show()

#最终降维与可视化：
#基于分析（例如，前两个主成分已保留超过95%的方差），我们选择 n_components=2 进行最终降维。
#将4维数据降为2维后，我们可以轻松地将其在散点图上可视化。不同颜色的点代表不同种类的花。你会看到，降维后的数据依然能很好地分离出三个类别，这证明了PCA的有效性。
# 5. 基于分析，选择K值并执行最终PCA
# 从图上可以看出，前两个主成分已经保留了超过95%的方差，我们决定降到2维
k = 2
pca_final = PCA(n_components=k)
X_pca = pca_final.fit_transform(X_scaled) # 拟合模型并应用降维

print(f"\n降维后的数据形状: {X_pca.shape}")

# 6. 可视化降维结果
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=70)
plt.xlabel(f'primary component (PC1: {explained_variance_ratio[0]*100:.2f}%)')
plt.ylabel(f'secondary component (PC2: {explained_variance_ratio[1]*100:.2f}%)')
plt.title(f'PCA_result (2D, {cumulative_variance_ratio[1]*100:.2f}%)')
plt.colorbar(scatter, label='iris species', ticks=range(3))
plt.clim(-0.5, 2.5)
# 为颜色条添加标签
cb = plt.colorbar(scatter, label='iris species', ticks=range(3))
cb.set_ticklabels(target_names)
plt.show()

#主成分载荷 (Component Loadings)：
#components_ 属性显示了每个主成分是如何由原始特征线性组合而成的。
#例如，如果PC1在"花瓣长度"上的载荷值很大（例如0.8），而其他特征载荷小，说明第一主成分主要代表了花瓣长度的信息。这帮助我们解释主成分的实际意义。
# （可选）7. 查看主成分的构成（哪些原始特征贡献大）
# 主成分是原始特征的线性组合
pca_components = pca_final.components_
components_df = pd.DataFrame(pca_components.T, # 转置以便于阅读
                             columns=[f'PC{i+1}' for i in range(k)],
                             index=feature_names)
print("\n Component Loadings:")
print(components_df)