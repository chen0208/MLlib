#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 15:42:58 2025

@author: chenjingqi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import warnings
warnings.filterwarnings('ignore') # 避免不必要的警告干扰输出

#数据加载：
#我们使用 Fashion-MNIST 数据集，它比手写数字更复杂，包含10类服装物品。
#这能更好地展示 UMAP 处理复杂、真实世界数据的能力。我们抽样5000个点以加快演示速度。
# 1. 加载数据 - 使用Fashion-MNIST，一个更复杂的10分类数据集
# 这里我们使用 sklearn 的 fetch_openml 来获取，也可以从Keras等库加载
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

print("Loading Fashion-MNIST data...")
# 这将下载数据，可能需要一点时间
fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='pandas')
X = fashion_mnist.data
y = fashion_mnist.target.astype(int) # 目标转换为整数

# 为了演示速度，我们随机抽取一个子集（例如5000个样本）
np.random.seed(42)
sample_indices = np.random.choice(len(X), 5000, replace=False)
X_sample = X[sample_indices]
y_sample = y[sample_indices]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Data shape (subsampled): {X_sample.shape}")
print(f"Labels: {np.unique(y_sample)}")

# 2. 数据预处理 - 标准化是必须的
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

#执行UMAP：
#注意关键参数 n_neighbors 和 min_dist 的设置。verbose=True 会让你看到优化的进度条。
#你会注意到运行速度非常快，即使对于数千个样本。
# 3. 运行 UMAP
print("Running UMAP...")

# 初始化UMAP转换器
# 关键参数：
# n_components: 目标维度
# n_neighbors: 平衡局部/全局结构，最重要参数
# min_dist: 控制点聚集的紧密度
# metric: 计算高维空间距离的度量，默认'euclidean'（欧氏距离）通常很好
# random_state: 保证结果可复现
umap = UMAP(n_components=2,
            n_neighbors=15,      # 默认15，可尝试10-200
            min_dist=0.1,        # 默认0.1，可尝试0.01-0.99
            metric='euclidean',
            random_state=42,
            verbose=True)        # 打印进度信息

# 拟合并转换数据
X_umap = umap.fit_transform(X_scaled)

print("UMAP completed!")

#可视化：
#观察结果：你应该能看到10个清晰分离的集群，每个集群对应一种服装类型。特别注意 UMAP 如何很好地保持了全局关系——你可能发现“鞋类”（Sandal, Sneaker, Ankle boot）的集群彼此靠近，而“上衣类”（T-shirt, Pullover, Dress, Coat）也聚集在图的另一区域。这种全局结构的保持是 UMAP 的显著优势。
# 4. 可视化结果
df_umap = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
df_umap['Label'] = y_sample
df_umap['Class'] = df_umap['Label'].map(lambda x: class_names[x])

plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_umap['UMAP1'], df_umap['UMAP2'],
                     c=df_umap['Label'],           # 按数字标签着色
                     cmap='Spectral',              # 使用鲜艳的色谱
                     s=5,                          # 点较小，因为数据多
                     alpha=0.7)
plt.colorbar(scatter, ticks=range(10), label='Class')
plt.clim(-0.5, 9.5)
# 为颜色条设置标签
cbar = plt.colorbar(scatter, ticks=range(10), label='Class')
cbar.set_ticklabels(class_names)
plt.title(f'UMAP Projection of the Fashion-MNIST Dataset (n_neighbors={umap.n_neighbors}, min_dist={umap.min_dist})')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.tight_layout()
plt.show()

#参数探索：
#n_neighbors 影响：小值（5）会产生更多、更碎片化的集群，揭示非常局部的模式；大值（100）会产生更整体、更“平滑”的图谱，集群更大但可能模糊细节。
#min_dist 影响：小值（0.01）让点紧紧地挤在一起，形成非常致密的团块；大值（0.99）让点均匀散布在可用空间中，更容易看到集群内部的子结构。
# 5. 【高级】探索关键参数 n_neighbors 的影响
# 理解这个参数是掌握UMAP的关键
n_neighbors_to_try = [5, 15, 50, 100]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, n_nbrs in enumerate(n_neighbors_to_try):
    # 为每个参数值创建一个UMAP嵌入
    umap_model = UMAP(n_components=2, n_neighbors=n_nbrs, min_dist=0.1, random_state=42)
    X_umap_temp = umap_model.fit_transform(X_scaled)
    
    ax = axes[i]
    scatter = ax.scatter(X_umap_temp[:, 0], X_umap_temp[:, 1], c=y_sample, cmap='Spectral', s=2, alpha=0.7)
    ax.set_title(f'n_neighbors = {n_nbrs}')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('Effect of n_neighbors on UMAP Embedding', fontsize=16)
plt.tight_layout()
plt.show()

# 6. 【高级】探索关键参数 min_dist 的影响
min_dists_to_try = [0.01, 0.1, 0.5, 0.99]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, m_dist in enumerate(min_dists_to_try):
    umap_model = UMAP(n_components=2, n_neighbors=15, min_dist=m_dist, random_state=42)
    X_umap_temp = umap_model.fit_transform(X_scaled)
    
    ax = axes[i]
    scatter = ax.scatter(X_umap_temp[:, 0], X_umap_temp[:, 1], c=y_sample, cmap='Spectral', s=2, alpha=0.7)
    ax.set_title(f'min_dist = {m_dist}')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('Effect of min_dist on UMAP Embedding', fontsize=16)
plt.tight_layout()
plt.show()

#下游应用：聚类：
#这段代码演示了 UMAP 的一个 killer feature：其输出可以作为特征输入给其他算法（如 K-Means）。我们直接在2维的 UMAP 嵌入上进行聚类，并评估其与真实标签的匹配度（ARI 和 NMI 分数）。结果显示，仅在2维数据上就能取得相当不错的聚类效果，这证明了 UMAP 成功地将最具有判别力的信息压缩到了低维空间。这是你绝对无法用 t-SNE 做到的事情。
# 7. 【高级】UMAP用于后续机器学习任务（聚类）
# 演示UMAP结果如何用于下游分析，这是t-SNE难以做到的
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 在UMAP降维后的2维数据上进行聚类
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_umap)

# 评估聚类效果（与真实标签比较）
ari_score = adjusted_rand_score(y_sample, cluster_labels)
nmi_score = normalized_mutual_info_score(y_sample, cluster_labels)

print(f"\nClustering Performance on UMAP-reduced data (2D):")
print(f"Adjusted Rand Index: {ari_score:.4f}") # 越接近1越好
print(f"Normalized Mutual Info: {nmi_score:.4f}") # 越接近1越好

# 可视化聚类结果
plt.figure(figsize=(10, 8))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=cluster_labels, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(label='Cluster ID')
plt.title('K-Means Clustering on UMAP Embedding')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()