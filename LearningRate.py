#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 17:45:12 2025

@author: chenjingqi
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 定义一个简单的神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 训练和测试函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=10, scheduler_name='Fixed'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses = []
    test_accuracies = []
    learning_rates = []  # 记录每个epoch的学习率

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # 更新学习率
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
                scheduler.step()
            else:
                scheduler.step()
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        train_losses.append(epoch_loss)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} LR: {current_lr:.6f}')

        # 测试阶段
        model.eval()
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        epoch_test_acc = running_corrects.double() / len(test_dataset)
        test_accuracies.append(epoch_test_acc.cpu().numpy())

        print(f'Test Acc: {epoch_test_acc:.4f}')

        # 深拷贝模型
        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'训练完成于 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳测试准确率: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, train_losses, test_accuracies, learning_rates

# 初始化模型、损失函数和优化器
def init_model_and_optimizer():
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return model, criterion, optimizer

# 固定学习率
print("=" * 50)
print("固定学习率策略")
print("=" * 50)
model_fixed, criterion, optimizer = init_model_and_optimizer()
# 不设置任何学习率调度器
model_fixed, losses_fixed, accs_fixed, lrs_fixed = train_model(
    model_fixed, criterion, optimizer, None, num_epochs=15, scheduler_name='Fixed'
)

# Step Decay
print("=" * 50)
print("Step Decay策略")
print("=" * 50)
model_step, criterion, optimizer = init_model_and_optimizer()
# 每5个epoch将学习率乘以0.1
scheduler_step = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
model_step, losses_step, accs_step, lrs_step = train_model(
    model_step, criterion, optimizer, scheduler_step, num_epochs=15, scheduler_name='Step'
)

# Exponential Decay
print("=" * 50)
print("Exponential Decay策略")
print("=" * 50)
model_exp, criterion, optimizer = init_model_and_optimizer()
# 每个epoch将学习率乘以0.95
scheduler_exp = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
model_exp, losses_exp, accs_exp, lrs_exp = train_model(
    model_exp, criterion, optimizer, scheduler_exp, num_epochs=15, scheduler_name='Exp'
)

# Cosine Annealing
print("=" * 50)
print("Cosine Annealing策略")
print("=" * 50)
model_cos, criterion, optimizer = init_model_and_optimizer()
# 使用余弦退火调度器
scheduler_cos = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=0.001)
model_cos, losses_cos, accs_cos, lrs_cos = train_model(
    model_cos, criterion, optimizer, scheduler_cos, num_epochs=15, scheduler_name='Cosine'
)

# 绘制结果比较图
plt.figure(figsize=(15, 10))

# 学习率变化曲线
plt.subplot(2, 2, 1)
plt.plot(lrs_fixed, label='Fixed LR')
plt.plot(lrs_step, label='Step Decay')
plt.plot(lrs_exp, label='Exponential Decay')
plt.plot(lrs_cos, label='Cosine Annealing')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedules')
plt.legend()
plt.grid(True)

# 训练损失曲线
plt.subplot(2, 2, 2)
plt.plot(losses_fixed, label='Fixed LR')
plt.plot(losses_step, label='Step Decay')
plt.plot(losses_exp, label='Exponential Decay')
plt.plot(losses_cos, label='Cosine Annealing')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)

# 测试准确率曲线
plt.subplot(2, 2, 3)
plt.plot(accs_fixed, label='Fixed LR')
plt.plot(accs_step, label='Step Decay')
plt.plot(accs_exp, label='Exponential Decay')
plt.plot(accs_cos, label='Cosine Annealing')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy Comparison')
plt.legend()
plt.grid(True)

# 最终性能比较
final_accuracies = [accs_fixed[-1], accs_step[-1], accs_exp[-1], accs_cos[-1]]
strategies = ['Fixed LR', 'Step Decay', 'Exp Decay', 'Cosine Anneal']

plt.subplot(2, 2, 4)
bars = plt.bar(strategies, final_accuracies)
plt.ylabel('Final Test Accuracy')
plt.title('Final Performance Comparison')
# 在柱状图上添加数值标签
for bar, acc in zip(bars, final_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{acc:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('lr_schedule_comparison.png')
plt.show()

# 打印最终结果
print("\n最终测试准确率比较:")
for strategy, acc in zip(strategies, final_accuracies):
    print(f"{strategy}: {acc:.4f}")