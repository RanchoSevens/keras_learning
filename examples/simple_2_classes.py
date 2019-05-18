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
model.fit(data, labels, nb_epoch=150, batch_size=32)
# 根据epoch的不同，可以得到不同的准确率， 理论情况下可以学习到所有特征实现100的acc
# 训练集如此，测试集就不是这样了
