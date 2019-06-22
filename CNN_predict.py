# -*- coding: UTF-8 -*-

# mnist神经网络训练，采用LeNet-5模型

import os
import cv2
import numpy as np
import pydot
import graphviz

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adadelta, Adagrad

from keras.utils import np_utils
from keras.utils import plot_model

import h5py
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
import pickle as p
import matplotlib.image as plimg
from PIL import Image

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)

    print("labels_path: ",labels_path)
    print("images_path: ", images_path)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels
    
# 建立一个Sequential模型
model = Sequential()

# model.add(Conv2D(4, 5, 5, border_mode='valid',input_shape=(28,28,1)))
# 第一个卷积层，4个卷积核，每个卷积核5*5,卷积后24*24，第一个卷积核要申明input_shape(通道，大小) ,激活函数采用“tanh”
model.add(Conv2D(filters=4, kernel_size=(5, 5), padding='valid', input_shape=(28, 28, 1), activation='tanh'))

# model.add(Conv2D(8, 3, 3, subsample=(2,2), border_mode='valid'))
# 第二个卷积层，8个卷积核，不需要申明上一个卷积留下来的特征map，会自动识别，下采样层为2*2,卷完且采样后是11*11
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=8, kernel_size=(3, 3), padding='valid', activation='tanh'))
# model.add(Activation('tanh'))

# model.add(Conv2D(16, 3, 3, subsample=(2,2), border_mode='valid'))
# 第三个卷积层，16个卷积核，下采样层为2*2,卷完采样后是4*4
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='valid', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Activation('tanh'))

model.add(Flatten())
# 把多维的模型压平为一维的，用在卷积层到全连接层的过度

# model.add(Dense(128, input_dim=(16*4*4), init='normal'))
# 全连接层，首层的需要指定输入维度16*4*4,128是输出维度，默认放第一位
model.add(Dense(128, activation='tanh'))

# model.add(Activation('tanh'))

# model.add(Dense(10, input_dim= 128, init='normal'))
# 第二层全连接层，其实不需要指定输入维度，输出为10维，因为是10类
model.add(Dense(10, activation='softmax'))
# model.add(Activation('softmax'))
# 激活函数“softmax”，用于分类

# 训练CNN模型

sgd = SGD(lr=0.05, momentum=0.9, decay=1e-6, nesterov=True)
# 采用随机梯度下降法，学习率初始值0.05,动量参数为0.9,学习率衰减值为1e-6,确定使用Nesterov动量
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# 配置模型学习过程，目标函数为categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列，第18行已转化，优化器为sgd
model.load_weights("CNN.h5")

path = ".\\mnist"
X_train, y_train = load_mnist(path, kind='train')

path = ".\\mnist"
X_test, y_test = load_mnist(path, kind='t10k')

print("X_train: ",X_train.T.shape, X_train.dtype)
print('y_train: ',y_train.T.shape, y_train.dtype)
print("X_test: ",X_test.shape, X_test.dtype)
print("y_test: ",y_test.shape, y_test.dtype)

number_index = 69
x_test1 = X_test[number_index,:]
print(x_test1.shape)

x_test1 = x_test1.reshape(28,28)
print(x_test1.shape)

plt.subplot(1,1,1)
plt.imshow(x_test1, cmap='gray', interpolation='none')

x_test1 = x_test1[np.newaxis, ..., np.newaxis]
x_test1_pred = model.predict(x_test1, batch_size=1, verbose=1)
print("x_test1_pred: ",np.argmax(x_test1_pred))

x_test1_actual = y_test[number_index]
print("x_test1_actual: ", x_test1_actual)