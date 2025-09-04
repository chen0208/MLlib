#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 17:47:14 2025

@author: chenjingqi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# 生成示例数据
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X = np.hstack([np.ones((X.shape[0], 1)), X])  # 添加偏置项
y = y.reshape(-1, 1)

# 初始化参数
def initialize_parameters(n_features):
    return np.zeros((n_features, 1))

# 定义损失函数（均方误差）
def compute_loss(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    loss = (1/(2*m)) * np.sum(np.square(predictions - y))
    return loss

# SGD优化器
def sgd_optimizer(X, y, learning_rate=0.01, n_iters=100):
    m, n = X.shape
    theta = initialize_parameters(n)
    losses = []
    
    for i in range(n_iters):
        for j in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            
            gradient = xi.T.dot(xi.dot(theta) - yi)
            theta -= learning_rate * gradient
        
        if i % 10 == 0:
            loss = compute_loss(X, y, theta)
            losses.append(loss)
    
    return theta, losses

# Momentum优化器
def momentum_optimizer(X, y, learning_rate=0.01, beta=0.9, n_iters=100):
    m, n = X.shape
    theta = initialize_parameters(n)
    v = np.zeros_like(theta)
    losses = []
    
    for i in range(n_iters):
        for j in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            
            gradient = xi.T.dot(xi.dot(theta) - yi)
            v = beta * v + (1 - beta) * gradient
            theta -= learning_rate * v
        
        if i % 10 == 0:
            loss = compute_loss(X, y, theta)
            losses.append(loss)
    
    return theta, losses

# AdaGrad优化器
def adagrad_optimizer(X, y, learning_rate=0.01, epsilon=1e-8, n_iters=100):
    m, n = X.shape
    theta = initialize_parameters(n)
    G = np.zeros_like(theta)
    losses = []
    
    for i in range(n_iters):
        for j in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            
            gradient = xi.T.dot(xi.dot(theta) - yi)
            G += gradient ** 2
            adjusted_grad = gradient / (np.sqrt(G) + epsilon)
            theta -= learning_rate * adjusted_grad
        
        if i % 10 == 0:
            loss = compute_loss(X, y, theta)
            losses.append(loss)
    
    return theta, losses

# RMSProp优化器
def rmsprop_optimizer(X, y, learning_rate=0.01, beta=0.9, epsilon=1e-8, n_iters=100):
    m, n = X.shape
    theta = initialize_parameters(n)
    E = np.zeros_like(theta)
    losses = []
    
    for i in range(n_iters):
        for j in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            
            gradient = xi.T.dot(xi.dot(theta) - yi)
            E = beta * E + (1 - beta) * (gradient ** 2)
            adjusted_grad = gradient / (np.sqrt(E) + epsilon)
            theta -= learning_rate * adjusted_grad
        
        if i % 10 == 0:
            loss = compute_loss(X, y, theta)
            losses.append(loss)
    
    return theta, losses

# Adam优化器
def adam_optimizer(X, y, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, n_iters=100):
    m, n = X.shape
    theta = initialize_parameters(n)
    m_t = np.zeros_like(theta)
    v_t = np.zeros_like(theta)
    losses = []
    
    for i in range(1, n_iters+1):
        for j in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            
            gradient = xi.T.dot(xi.dot(theta) - yi)
            
            # 更新一阶和二阶矩估计
            m_t = beta1 * m_t + (1 - beta1) * gradient
            v_t = beta2 * v_t + (1 - beta2) * (gradient ** 2)
            
            # 偏差校正
            m_hat = m_t / (1 - beta1 ** i)
            v_hat = v_t / (1 - beta2 ** i)
            
            # 更新参数
            theta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        if i % 10 == 0:
            loss = compute_loss(X, y, theta)
            losses.append(loss)
    
    return theta, losses

# 比较不同优化器
def compare_optimizers(X, y, n_iters=100):
    optimizers = {
        'SGD': sgd_optimizer,
        'Momentum': momentum_optimizer,
        'AdaGrad': adagrad_optimizer,
        'RMSProp': rmsprop_optimizer,
        'Adam': adam_optimizer
    }
    
    results = {}
    
    plt.figure(figsize=(12, 8))
    
    for name, optimizer in optimizers.items():
        print(f"训练 {name}...")
        theta, losses = optimizer(X, y, n_iters=n_iters)
        results[name] = losses
        plt.plot(losses, label=name)
    
    plt.xlabel('迭代次数 (每10次)')
    plt.ylabel('损失')
    plt.title('不同优化算法的损失曲线比较')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results

# 运行比较
results = compare_optimizers(X, y, n_iters=200)