#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 22:17:29 2025

@author: chenjingqi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# 生成样本数据
n_samples = 1500
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)

# 添加一些噪声
noise = np.random.normal(0, 0.5, (50, 2))
X = np.vstack([X, noise])

# 生成非球形数据集
X_varied, y_varied = datasets.make_blobs(n_samples=n_samples,
                                         cluster_std=[1.0, 2.5, 0.5],
                                         random_state=random_state)

# 生成各向异性分布的数据
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X_varied, transformation)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_varied_scaled = scaler.fit_transform(X_varied)
X_aniso_scaled = scaler.fit_transform(X_aniso)

# 可视化数据集
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], s=10)
plt.title("globular dataset")

plt.subplot(132)
plt.scatter(X_varied_scaled[:, 0], X_varied_scaled[:, 1], s=10)
plt.title("dataset with different variation")

plt.subplot(133)
plt.scatter(X_aniso_scaled[:, 0], X_aniso_scaled[:, 1], s=10)
plt.title("Anisotropic dataset")

plt.tight_layout()
plt.show()

from sklearn.cluster import KMeans
# 应用K-Means
kmeans = KMeans(n_clusters=3, random_state=random_state)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 可视化结果
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, s=10, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, alpha=0.8)
plt.title("K-Means")
plt.show()

from sklearn.cluster import DBSCAN

# 应用DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)

# 可视化结果
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, s=10, cmap='viridis')
plt.title("DBSCAN (noise is -1)")
plt.show()

from sklearn.cluster import AgglomerativeClustering

# 应用层次聚类
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# 可视化结果
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=hierarchical_labels, s=10, cmap='viridis')
plt.title("AgglomerativeClustering")
plt.show()

from sklearn.mixture import GaussianMixture

# 应用GMM
gmm = GaussianMixture(n_components=3, random_state=random_state)
gmm_labels = gmm.fit_predict(X_scaled)

# 可视化结果
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels, s=10, cmap='viridis')
plt.title("GMM")
plt.show()

from minisom import MiniSom

# 创建并训练SOM
som = MiniSom(x=10, y=10, input_len=2, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train_random(X_scaled, 100)

# 获取每个样本的获胜神经元
winners = np.array([som.winner(x) for x in X_scaled])

# 将二维网格位置转换为一维标签
# 首先创建一个映射，将每个唯一的获胜神经元位置映射到一个标签
unique_winners = np.unique(winners, axis=0)
winner_to_label = {}
som_labels = np.zeros(len(X_scaled))
for idx,winner in enumerate(unique_winners):
    winner_to_label[tuple(winner)] = idx

# 为每个样本分配标签
labels = np.array([winner_to_label[tuple(winner)] for winner in winners])

n_clusters = 3
# 如果获得的标签数量多于所需的聚类数，使用K-Means进行进一步聚类
if len(np.unique(labels)) > n_clusters:
    # 获取所有获胜神经元的位置
    winner_positions = np.array(list(winner_to_label.keys()))
    
    # 使用K-Means对获胜神经元位置进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    winner_clusters = kmeans.fit_predict(winner_positions)
    
    # 创建一个从原始获胜神经元标签到新聚类标签的映射
    position_to_cluster = {}
    for pos, cluster in zip(winner_positions, winner_clusters):
        position_to_cluster[tuple(pos)] = cluster
    
    # 更新样本标签
    labels = np.array([position_to_cluster[tuple(winner)] for winner in winners])


# 可视化结果
plt.figure(figsize=(10, 8))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=som_labels, s=10, cmap='viridis')
plt.title("SOM")
plt.show()

import skfuzzy as fuzz

# 应用FCM
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_scaled.T, 3, 2, error=0.005, maxiter=1000, init=None)

# 获取聚类标签
fcm_labels = np.argmax(u, axis=0)

# 可视化结果
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=fcm_labels, s=10, cmap='viridis')
plt.title("FCM")
plt.show()

