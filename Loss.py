#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 17:44:06 2025

@author: chenjingqi
"""

import numpy as np

def zero_one_loss(y_true, y_pred):
    """
    0-1损失函数
    用于分类问题，预测错误损失为1，正确为0
    
    参数:
    y_true: 真实标签 (n_samples,)
    y_pred: 预测标签 (n_samples,)
    
    返回:
    loss: 平均0-1损失
    """
    return np.mean(y_true != y_pred)

def absolute_loss(y_true, y_pred):
    """
    绝对值损失函数 (L1损失)
    用于回归问题，计算预测值与真实值之间的绝对差异
    
    参数:
    y_true: 真实值 (n_samples,)
    y_pred: 预测值 (n_samples,)
    
    返回:
    loss: 平均绝对误差
    """
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    """
    均方误差损失函数 (MSE)
    用于回归问题，计算预测值与真实值之间的平方差异
    
    参数:
    y_true: 真实值 (n_samples,)
    y_pred: 预测值 (n_samples,)
    
    返回:
    loss: 均方误差
    """
    return np.mean((y_true - y_pred) ** 2)

def log_loss(y_true, y_pred_proba, epsilon=1e-15):
    """
    对数损失函数 (逻辑损失)
    用于二分类问题，评估概率估计的质量
    
    参数:
    y_true: 真实标签 (0或1) (n_samples,)
    y_pred_proba: 预测为正类的概率 (n_samples,)
    epsilon: 小值用于避免log(0)的情况
    
    返回:
    loss: 对数损失
    """
    # 限制概率值范围，避免log(0)的问题
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))

def cross_entropy_loss(y_true, y_pred_proba, epsilon=1e-15):
    """
    交叉熵损失函数
    用于多分类问题，评估概率分布之间的差异
    
    参数:
    y_true: 真实标签的one-hot编码 (n_samples, n_classes)
    y_pred_proba: 预测概率 (n_samples, n_classes)
    epsilon: 小值用于避免log(0)的情况
    
    返回:
    loss: 交叉熵损失
    """
    # 限制概率值范围，避免log(0)的问题
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred_proba), axis=1))

def exponential_loss(y_true, y_pred):
    """
    指数损失函数
    主要用于AdaBoost算法
    
    参数:
    y_true: 真实标签 (-1或1) (n_samples,)
    y_pred: 预测值 (实数) (n_samples,)
    
    返回:
    loss: 指数损失
    """
    return np.mean(np.exp(-y_true * y_pred))

def hinge_loss(y_true, y_pred):
    """
    Hinge损失函数
    用于支持向量机(SVM)
    
    参数:
    y_true: 真实标签 (-1或1) (n_samples,)
    y_pred: 预测值 (实数) (n_samples,)
    
    返回:
    loss: Hinge损失
    """
    return np.mean(np.maximum(0, 1 - y_true * y_pred))


# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    n_samples = 100
    y_true_reg = np.random.randn(n_samples)  # 回归问题的真实值
    y_pred_reg = y_true_reg + np.random.randn(n_samples) * 0.5  # 回归问题的预测值
    
    y_true_clf = np.random.randint(0, 2, n_samples)  # 二分类真实标签
    y_pred_clf = np.random.randint(0, 2, n_samples)  # 二分类预测标签
    y_pred_proba = np.random.rand(n_samples)  # 二分类预测概率
    
    y_true_multiclass = np.eye(3)[np.random.randint(0, 3, n_samples)]  # 多分类one-hot编码
    y_pred_multiclass = np.random.rand(n_samples, 3)  # 多分类预测概率
    y_pred_multiclass = y_pred_multiclass / np.sum(y_pred_multiclass, axis=1, keepdims=True)  # 归一化
    
    y_true_svm = np.random.choice([-1, 1], n_samples)  # SVM真实标签
    y_pred_svm = np.random.randn(n_samples)  # SVM预测值
    
    # 计算各种损失
    print("0-1损失:", zero_one_loss(y_true_clf, y_pred_clf))
    print("绝对值损失:", absolute_loss(y_true_reg, y_pred_reg))
    print("均方误差:", mean_squared_error(y_true_reg, y_pred_reg))
    print("对数损失:", log_loss(y_true_clf, y_pred_proba))
    print("交叉熵损失:", cross_entropy_loss(y_true_multiclass, y_pred_multiclass))
    print("指数损失:", exponential_loss(y_true_svm, y_pred_svm))
    print("Hinge损失:", hinge_loss(y_true_svm, y_pred_svm))