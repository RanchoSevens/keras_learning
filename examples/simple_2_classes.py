# -*-coding:utf-8
from keras.models import Sequential
from keras.layers import Dense, Activation

# 模型搭建
model = Sequential()  # 类初始化
model.add(Dense(32, activation='relu', input_dim=100))
# Dense(32)是一个含32个隐藏单元的全连接层
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Generate dummy data
import numpy as np

data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# train the model,iterating on the data in batches of 32 samples
model.fit(data, labels, nb_epoch=10, batch_size=32)
