#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 22:41:36 2025

@author: chenjingqi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from minisom import MiniSom
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score
import time

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 生成样本数据
n_samples = 1500
random_state = 170

# 生成三种不同类型的数据集
X_blobs, y_blobs = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
X_varied, y_varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X_varied, transformation)

# 添加一些噪声点
noise = np.random.normal(0, 0.5, (50, 2))
X_blobs = np.vstack([X_blobs, noise])
X_varied = np.vstack([X_varied, noise])
X_aniso = np.vstack([X_aniso, noise])

# 标准化数据
scaler = StandardScaler()
X_blobs_scaled = scaler.fit_transform(X_blobs)
X_varied_scaled = scaler.fit_transform(X_varied)
X_aniso_scaled = scaler.fit_transform(X_aniso)

# 可视化数据集
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.scatter(X_blobs_scaled[:, 0], X_blobs_scaled[:, 1], s=10)
plt.title("globular dataset")

plt.subplot(132)
plt.scatter(X_varied_scaled[:, 0], X_varied_scaled[:, 1], s=10)
plt.title("diffirent variation dataset")

plt.subplot(133)
plt.scatter(X_aniso_scaled[:, 0], X_aniso_scaled[:, 1], s=10)
plt.title("Anisotropic dataset")

plt.tight_layout()
plt.show()

# 定义聚类算法函数
def apply_kmeans(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    start_time = time.time()
    labels = kmeans.fit_predict(X)
    end_time = time.time()
    return labels, end_time - start_time

def apply_dbscan(X, eps=0.3, min_samples=10):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    start_time = time.time()
    labels = dbscan.fit_predict(X)
    end_time = time.time()
    return labels, end_time - start_time

def apply_hierarchical(X, n_clusters=3):
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    start_time = time.time()
    labels = hierarchical.fit_predict(X)
    end_time = time.time()
    return labels, end_time - start_time

def apply_gmm(X, n_clusters=3):
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    start_time = time.time()
    labels = gmm.fit_predict(X)
    end_time = time.time()
    return labels, end_time - start_time

def apply_som(X, n_clusters=3, grid_size=10):
    som = MiniSom(grid_size, grid_size, X.shape[1], sigma=1.0, learning_rate=0.5, random_seed=random_state)
    som.random_weights_init(X)
    start_time = time.time()
    som.train_random(X, 100, verbose=False)
    
    # 获取每个样本的获胜神经元
    winners = np.array([som.winner(x) for x in X])
    
    # 将二维网格位置转换为一维标签
    # 首先创建一个映射，将每个唯一的获胜神经元位置映射到一个标签
    unique_winners = np.unique(winners, axis=0)
    winner_to_label = {}
    for idx, winner in enumerate(unique_winners):
        winner_to_label[tuple(winner)] = idx
    
    # 为每个样本分配标签
    labels = np.array([winner_to_label[tuple(winner)] for winner in winners])
    
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
    
    end_time = time.time()
    return labels, end_time - start_time

def apply_fcm(X, n_clusters=3):
    start_time = time.time()
    # 转置数据以适应skfuzzy的输入格式
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)
    labels = np.argmax(u, axis=0)
    end_time = time.time()
    return labels, end_time - start_time

# 准备数据集和算法
datasets = {
    "globular dataset": X_blobs_scaled,
    "dataset with different variation": X_varied_scaled,
    "Anisotropic dataset": X_aniso_scaled
}

algorithms = {
    "K-Means": apply_kmeans,
    "DBSCAN": apply_dbscan,
    "hierarchical": apply_hierarchical,
    "GMM": apply_gmm,
    "SOM": apply_som,
    "FCM": apply_fcm
}

# 运行所有算法并收集结果
results = {}

for dataset_name, X in datasets.items():
    results[dataset_name] = {}
    
    for algo_name, algo_func in algorithms.items():
        try:
            if algo_name == "DBSCAN":
                # 对于DBSCAN，尝试不同的参数组合
                eps_values = [0.2, 0.3, 0.4]
                min_samples_values = [5, 10, 15]
                best_labels = None
                best_sil_score = -1
                
                for eps in eps_values:
                    for min_samples in min_samples_values:
                        labels, time_taken = algo_func(X, eps=eps, min_samples=min_samples)
                        
                        # 计算轮廓系数（对于DBSCAN，排除噪声点）
                        non_noise = labels != -1
                        if sum(non_noise) > 1 and len(np.unique(labels[non_noise])) > 1:
                            sil_score = silhouette_score(X[non_noise], labels[non_noise])
                            if sil_score > best_sil_score:
                                best_sil_score = sil_score
                                best_labels = labels
                                best_time = time_taken
                
                if best_labels is None:
                    # 如果没有找到合适的参数，使用默认参数
                    labels, time_taken = algo_func(X, eps=0.3, min_samples=10)
                    non_noise = labels != -1
                    if sum(non_noise) > 1 and len(np.unique(labels[non_noise])) > 1:
                        sil_score = silhouette_score(X[non_noise], labels[non_noise])
                    else:
                        sil_score = -1
                else:
                    labels = best_labels
                    time_taken = best_time
                    sil_score = best_sil_score
            else:
                # 对于其他算法，直接调用
                labels, time_taken = algo_func(X, n_clusters=3)
                
                # 计算轮廓系数
                if len(np.unique(labels)) > 1:
                    sil_score = silhouette_score(X, labels)
                else:
                    sil_score = -1
            
            results[dataset_name][algo_name] = {
                "labels": labels,
                "time": time_taken,
                "silhouette": sil_score
            }
            
        except Exception as e:
            print(f"算法 {algo_name} 在数据集 {dataset_name} 上出错: {e}")
            results[dataset_name][algo_name] = None

# 可视化所有结果
fig, axes = plt.subplots(3, 6, figsize=(20, 12))

for i, (dataset_name, X) in enumerate(datasets.items()):
    for j, algo_name in enumerate(algorithms.keys()):
        ax = axes[i, j]
        
        if results[dataset_name][algo_name] is not None:
            labels = results[dataset_name][algo_name]["labels"]
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap='viridis')
            
            # 添加标题和性能指标
            sil_score = results[dataset_name][algo_name]["silhouette"]
            time_taken = results[dataset_name][algo_name]["time"]
            ax.set_title(f"{algo_name}\nSil: {sil_score:.3f}, Time: {time_taken:.3f}s")
        else:
            ax.text(0.5, 0.5, "Error", ha='center', va='center')
            ax.set_title(algo_name)
        
        if j == 0:
            ax.set_ylabel(dataset_name)

plt.tight_layout()
plt.show()

# 打印性能比较表
print("算法性能比较:")
print("数据集\t\t算法\t\t时间(秒)\t轮廓系数")
print("-" * 50)

for dataset_name in datasets:
    for algo_name in algorithms:
        if results[dataset_name][algo_name] is not None:
            res = results[dataset_name][algo_name]
            print(f"{dataset_name}\t{algo_name:12}\t{res['time']:.4f}\t\t{res['silhouette']:.4f}")