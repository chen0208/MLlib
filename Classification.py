#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 17:34:58 2025

@author: chenjingqi
"""

#经典的二分类问题
#使用随机森林算法
#乳腺癌预测
# 1. 导入必要的库
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, 
                             ConfusionMatrixDisplay, RocCurveDisplay)
import matplotlib.pyplot as plt

# 2. 加载数据集
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names # ['malignant'（恶性）, 'benign'（良性）]

print(f"数据集形状: {X.shape}")
print(f"类别分布: malignant（恶性）: {sum(y==0)}, benign（良性）: {sum(y==1)}")

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    stratify=y, # 保持类别比例
                                                    random_state=42)

# 4. 选择并训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 在测试集上进行预测
y_pred = model.predict(X_test)
# 也可以得到预测概率
y_pred_proba = model.predict_proba(X_test)[:, 1] # 取正类（良性）的概率

# 6. 全面评估模型性能
print("\n=== 模型性能评估 ===")
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(f"精确率 (Precision): {precision_score(y_test, y_pred):.4f}")
print(f"召回率 (Recall): {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# 7. 绘制混淆矩阵
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=target_names)
disp.plot(cmap='Blues', ax=ax1)
ax1.set_title('Confusion Matrix')

# 8. 绘制ROC曲线
RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax2)
ax2.plot([0, 1], [0, 1], 'k--') # 绘制对角线（随机猜测）
ax2.set_title('ROC Curve')
plt.tight_layout()
plt.show()

# 9. (可选) 分析特征重要性
importances = model.feature_importances_
indices = np.argsort(importances)[::-1] # 从高到低排序

print("\nTop 5 重要特征:")
for i in range(5):
    print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")