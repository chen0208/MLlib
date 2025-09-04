#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 17:06:15 2025

@author: chenjingqi
"""

#非监督学习
#示例（客户细分 - 聚类问题）
# 步骤 1: 导入必要库
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# 步骤 2: 生成模拟的无标签数据
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
# 注意：我们假装不知道y_true，它是用来后期验证的

# 步骤 3: 数据预处理 (本例中数据已模拟好，无需额外处理)

# 步骤 4 & 5: 选择模型并进行训练
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X) # 只传入X，没有y！

# 步骤 6: 结果解释与分析
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.xlabel("特征 1")
plt.ylabel("特征 2")
plt.show()

# 分析每个簇的中心点，可以为每个簇赋予业务含义
print("簇中心坐标（代表该簇的典型特征）：")
print(centers)
# 例如，中心点数值大的簇可能是“高价值客户”，小的可能是“低价值客户”