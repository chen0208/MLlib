#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 15:28:44 2025

@author: chenjingqi
"""
#我们将使用经典的 MNIST 手写数字数据集，这是一个高维（784维）数据，非常适合展示 t-SNE 的强大可视化能力
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

#数据加载与预览：
#我们使用 MNIST/Digits 数据集，它包含手写数字的8x8图像（展平后是64维）。代码中展示了前10个样本，让你对数据有个直观印象。

# 1. 加载数据
# 使用MNIST数据集，这是一个28x28像素的手写数字图像数据集，共784维
print("Loading MNIST data...")
digits = datasets.load_digits() # 我们先使用sklearn自带的更小的Digits数据集（8x8图像）以便快速演示
# 如果你想用完整的MNIST，可以使用 `from sklearn.datasets import fetch_openml; mnist = fetch_openml('mnist_784', version=1)`
X = digits.data
y = digits.target
target_names = digits.target_names

print(f"Data shape: {X.shape}") # 应该是 (n_samples, 64)
print(f"Labels: {np.unique(y)}")

# 展示一些原始图片
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
plt.suptitle('Sample Images from MNIST/Digits Dataset')
plt.tight_layout()
plt.show()

#预处理与PCA预降维：
#标准化是必须的，因为它直接影响距离计算。
#先用PCA降到50维是标准工程实践，能大幅减少t-SNE的计算时间，同时去除高频噪声。

# 2. 数据预处理
# 标准化对于基于距离的算法（计算相似度）非常重要
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 3. 首先使用PCA进行初步降维（加速t-SNE并去噪）
# 这是一个非常常见的技巧：PCA -> t-SNE
print("Running PCA for initial dimensionality reduction...")
pca = PCA(n_components=50) # 常见选择：30-50
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance ratio by {pca.n_components_} components: {pca.explained_variance_ratio_.sum():.4f}")


#执行t-SNE：
#注意关键参数 perplexity, random_state, n_iter 的设置。
#init='pca' 使用PCA初始化，通常能得到更稳定的结果。
#计时是为了让你感受t-SNE的计算成本。

# 4. 运行 t-SNE
print("Running t-SNE...")
start_time = time.time()
# 初始化 t-SNE 模型
# 关键参数：
# n_components: 目标维度（通常是2）
# perplexity: 困惑度，最重要参数，通常在5-50之间
# random_state: 设置随机种子以保证结果可复现（但记住，不同seed结果会不同）
# n_iter: 优化迭代次数
# learning_rate: 学习率，通常10-1000，默认200较好
tsne = TSNE(n_components=2,
            perplexity=40,        # 尝试30, 40, 50
            random_state=42,      # 保证可复现性
            n_iter=1000,
            learning_rate='auto', # 新版本推荐 'auto'
            init='pca')           # 初始化方式，'pca' 通常比随机初始化更稳定

# 在PCA降维后的数据上拟合和变换
X_tsne = tsne.fit_transform(X_pca)

end_time = time.time()
print(f"t-SNE completed in {end_time - start_time:.2f} seconds.")

#可视化：
#使用 seaborn 绘制高质量的散点图，不同颜色代表不同的数字。
#观察结果：你应该能看到10个清晰的、分离的集群，每个集群大致对应一个数字。这正是t-SNE的强大之处——它成功地将高维像素空间中复杂的模式，映射成了人类肉眼可见的、易于理解的二维结构。你会发现有些数字（如“1”）的集群非常紧密，而有些（如“8”）可能稍微分散，这反映了书写这些数字本身的变体多少。
# 5. 创建结果DataFrame并可视化
df_tsne = pd.DataFrame(X_tsne, columns=['Dimension 1', 'Dimension 2'])
df_tsne['Label'] = y
df_tsne['Label'] = df_tsne['Label'].astype(str) # 转换为字符串类型便于绘图

# 使用Seaborn绘制漂亮的散点图
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_tsne,
                x='Dimension 1',
                y='Dimension 2',
                hue='Label',
                palette='tab10', # 使用清晰的颜色盘
                s=50,           # 点的大小
                alpha=0.7,      # 透明度
                legend='full')
plt.title(f't-SNE Visualization of MNIST/Digits (Perplexity={tsne.perplexity}, PCA preprocessed)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 将图例放在图外
plt.tight_layout()
plt.show()

#探索Perplexity的影响：
#这个对比图至关重要。你会发现：
#Perplexity=5：形成大量的小微集群，局部细节丰富，但可能过度分裂。
#Perplexity=30/50：形成更大、更连贯的集群，全局结构更清晰。通常需要尝试多个值来选择最能揭示数据底层结构的一个。
# 6. 【高级】探索不同Perplexity的影响
# 理解超参数的影响是掌握t-SNE的关键
perplexities = [5, 30, 50]
fig, axes = plt.subplots(1, len(perplexities), figsize=(18, 5))

for i, perp in enumerate(perplexities):
    tsne_model = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
    X_tsne_temp = tsne_model.fit_transform(X_pca)
    
    df_temp = pd.DataFrame(X_tsne_temp, columns=['Dim1', 'Dim2'])
    df_temp['Label'] = y.astype(str)
    
    sc = axes[i].scatter(df_temp['Dim1'], df_temp['Dim2'], c=y, cmap='tab10', s=20, alpha=0.7)
    axes[i].set_title(f'Perplexity = {perp}')
    axes[i].set_xlabel('Dimension 1')
    axes[i].set_ylabel('Dimension 2')

plt.suptitle('Effect of Perplexity on t-SNE Clustering', fontsize=16)
plt.tight_layout()
plt.show()

# 7. 【补充】KL散度 - 评估优化效果
# 最终的KL散度代表了映射后信息损失的程度，值越低越好（但不同数据集之间无法比较）
print(f"Final KL divergence: {tsne.kl_divergence_:.4f}")