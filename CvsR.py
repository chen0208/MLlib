#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 17:41:06 2025

@author: chenjingqi
"""

#让我们用经典的 scikit-learn 库和鸢尾花数据集来直观感受一下分类和回归的区别。
#分类示例：预测鸢尾花种类（离散标签）
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target # y 是离散的类别标签 (0, 1, 2)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 选择并训练一个【分类】模型
clf_model = RandomForestClassifier()
clf_model.fit(X_train, y_train)

# 进行预测并评估
y_pred = clf_model.predict(X_test)
print("分类预测结果:", y_pred)
print("分类准确率:", accuracy_score(y_test, y_pred))

#回归示例：预测花瓣长度（连续值）
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 将花瓣长度作为目标变量y，其他特征作为X
X = iris.data[:, [0, 1, 3]] # 萼片长度、萼片宽度、花瓣宽度
y = iris.data[:, 2] # 花瓣长度（连续值）

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 选择并训练一个【回归】模型
reg_model = RandomForestRegressor()
reg_model.fit(X_train, y_train)

# 进行预测并评估
y_pred = reg_model.predict(X_test)
print("回归预测结果:", y_pred)
print("真实结果:", y_test)
print("均方误差 (MSE):", mean_squared_error(y_test, y_pred))