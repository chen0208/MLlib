#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 17:48:36 2025

@author: chenjingqi
"""

import torch
import torch.nn as nn
import torch.optim as optim

# 在PyTorch中使用这些优化器
model = nn.Linear(20, 1)

# 各种优化器的初始化
sgd_optimizer = optim.SGD(model.parameters(), lr=0.01)
momentum_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adagrad_optimizer = optim.Adagrad(model.parameters(), lr=0.01)
rmsprop_optimizer = optim.RMSprop(model.parameters(), lr=0.01)
adam_optimizer = optim.Adam(model.parameters(), lr=0.01)

# 在TensorFlow/Keras中的使用示例
"""
from tensorflow.keras.optimizers import SGD, Adagrad, RMSprop, Adam

sgd = SGD(learning_rate=0.01)
momentum = SGD(learning_rate=0.01, momentum=0.9)
adagrad = Adagrad(learning_rate=0.01)
rmsprop = RMSprop(learning_rate=0.01)
adam = Adam(learning_rate=0.01)
"""
