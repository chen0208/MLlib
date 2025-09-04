#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 17:44:32 2025

@author: chenjingqi
"""

import numpy as np

# 1. 均方误差 (Mean Squared Error, MSE)
def mean_squared_error(y_true, y_pred):
    """
    计算均方误差(MSE)
    
    参数:
    y_true -- 真实值数组
    y_pred -- 预测值数组
    
    返回:
    mse -- 均方误差值
    """
    # 确保输入为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 检查形状是否匹配
    if y_true.shape != y_pred.shape:
        raise ValueError("输入数组形状不匹配")
    
    # 计算均方误差
    mse = np.mean(np.square(y_true - y_pred))
    return mse


# 2. 平均绝对误差 (Mean Absolute Error, MAE)
def mean_absolute_error(y_true, y_pred):
    """
    计算平均绝对误差(MAE)
    
    参数:
    y_true -- 真实值数组
    y_pred -- 预测值数组
    
    返回:
    mae -- 平均绝对误差值
    """
    # 确保输入为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 检查形状是否匹配
    if y_true.shape != y_pred.shape:
        raise ValueError("输入数组形状不匹配")
    
    # 计算平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


# 3. 交叉熵代价函数 (Cross-Entropy Loss)
def cross_entropy_loss(y_true, y_pred, epsilon=1e-12):
    """
    计算交叉熵损失
    
    参数:
    y_true -- 真实标签(one-hot编码或类别索引)
    y_pred -- 预测概率值数组
    epsilon -- 小值防止log(0)的情况
    
    返回:
    cross_entropy -- 交叉熵损失值
    """
    # 确保输入为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 如果y_true是类别索引(非one-hot)，将其转换为one-hot编码
    if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
        # 获取类别数
        n_classes = y_pred.shape[1] if len(y_pred.shape) > 1 else int(np.max(y_true)) + 1
        # 创建one-hot编码
        y_true_onehot = np.eye(n_classes)[y_true.astype(int).flatten()]
    else:
        y_true_onehot = y_true
    
    # 裁剪预测值以避免log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    
    # 计算交叉熵损失
    cross_entropy = -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))
    return cross_entropy


# 4. 二元交叉熵损失 (Binary Cross-Entropy Loss)
def binary_cross_entropy_loss(y_true, y_pred, epsilon=1e-12):
    """
    计算二元交叉熵损失
    
    参数:
    y_true -- 真实二进制标签(0或1)
    y_pred -- 预测概率值(介于0和1之间)
    epsilon -- 小值防止log(0)的情况
    
    返回:
    bce_loss -- 二元交叉熵损失值
    """
    # 确保输入为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 裁剪预测值以避免log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    
    # 计算二元交叉熵损失
    bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce_loss


# 示例使用
if __name__ == "__main__":
    # 示例数据
    y_true_reg = np.array([3, -0.5, 2, 7])
    y_pred_reg = np.array([2.5, 0.0, 2, 8])
    
    # 分类示例数据
    y_true_cls = np.array([0, 1, 2])  # 类别索引
    y_pred_cls = np.array([[0.9, 0.1, 0.0],  # 预测概率
                           [0.2, 0.7, 0.1],
                           [0.1, 0.3, 0.6]])
    
    # 二元分类示例数据
    y_true_binary = np.array([0, 1, 1, 0])
    y_pred_binary = np.array([0.1, 0.9, 0.8, 0.3])
    
    # 计算并输出结果
    print("回归问题示例:")
    print(f"真实值: {y_true_reg}")
    print(f"预测值: {y_pred_reg}")
    print(f"MSE: {mean_squared_error(y_true_reg, y_pred_reg):.4f}")
    print(f"MAE: {mean_absolute_error(y_true_reg, y_pred_reg):.4f}")
    
    print("\n分类问题示例:")
    print(f"真实类别: {y_true_cls}")
    print("预测概率:")
    print(y_pred_cls)
    print(f"交叉熵损失: {cross_entropy_loss(y_true_cls, y_pred_cls):.4f}")
    
    print("\n二元分类问题示例:")
    print(f"真实标签: {y_true_binary}")
    print(f"预测概率: {y_pred_binary}")
    print(f"二元交叉熵损失: {binary_cross_entropy_loss(y_true_binary, y_pred_binary):.4f}")
    
    # 梯度计算示例(MSE)
    print("\nMSE梯度计算示例:")
    def mse_gradient(y_true, y_pred):
        """计算MSE的梯度"""
        return 2 * (y_pred - y_true) / len(y_true)
    
    gradient = mse_gradient(y_true_reg, y_pred_reg)
    print(f"MSE梯度: {gradient}")