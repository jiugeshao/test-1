from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import matplotlib.pyplot as plt
from keras.utils import plot_model
import numpy

model = Sequential()
model.add(Dense(500,init='glorot_uniform', input_dim=784)) # 输入层，28*28=784

model.add(Activation('tanh')) # 激活函数是tanh
model.add(Dropout(0.5)) # 采用50%的dropout

model.add(Dense(500, init='glorot_uniform')) # 隐层节点500个
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(10, init='glorot_uniform')) # 输出结果是10个类别，所以维度是10
model.add(Activation('softmax')) # 最后一层用softmax

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # 设定学习率（lr）等参数
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy']) # 使用交叉熵作为loss函数

(X_train, y_train), (X_test, y_test) = mnist.load_data() # 使用Keras自带的mnist工具读取数据（第一次需要联网）

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]) # 由于mist的输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维
print("X_train.shape: ",X_train.shape)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
Y_train = (numpy.arange(10) == y_train[:, None]).astype(int) # 参考上一篇文章，这里需要把index转换成一个one hot的矩阵
Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)

# 开始训练，这里参数比较多。batch_size就是batch_size，nb_epoch就是最多迭代的次数， shuffle就是是否把数据随机打乱之后再进行训练
# verbose是屏显模式，官方这么说的：verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
# 就是说0是不屏显，1是显示一个进度条，2是每个epoch都显示一行数据
# show_accuracy就是显示每次迭代后的正确率
# validation_split就是拿出百分之多少用来做交叉验证
model_checkpoint = ModelCheckpoint("MLP.h5", monitor='val_loss', save_best_only=True)
model.fit(X_train, Y_train, batch_size=200, epochs=2, shuffle=True, verbose=1, validation_split=0.3,callbacks=[model_checkpoint])
plot_model(model, to_file='MLP.png', show_shapes=True, show_layer_names=True)
print('test set')
score = model.evaluate(X_test, Y_test, batch_size=200, verbose=1)
json_string = model.to_json()
open('MLP.json','w').write(json_string)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
