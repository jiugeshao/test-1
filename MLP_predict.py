from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
import os
import struct
import numpy as np
from keras.utils import np_utils
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

#模型预测
path = ".\\mnist"
X_train, y_train = load_mnist(path, kind='train')

path = ".\\mnist"
X_test, y_test = load_mnist(path, kind='t10k')

print("X_train: ",X_train.T.shape, X_train.dtype)
print('y_train: ',y_train.T.shape, y_train.dtype)
print("X_test: ",X_test.shape, X_test.dtype)
print("y_test: ",y_test.shape, y_test.dtype)

index=69
x_test1 = X_test[index,:]
x_test1_array= np.empty((1, 784), dtype="float32")
x_test1_array[0,:] = x_test1

print(x_test1.shape)

x_test1 = x_test1.reshape(28,28)
print(x_test1.shape)

plt.subplot(1,1,1)
plt.imshow(x_test1, cmap='gray', interpolation='none')

model = load_model('MLP.h5')
x_test1_pred = model.predict_classes(x_test1_array, verbose=0)
print("x_test1_pred: ", x_test1_pred)

x_test1_actual = y_test[index]
print("x_test1_actual: ", x_test1_actual)
#plt.title("Class {}".format(y_train[i]))
#x_test1_image = Image.fromarray(x_test1)
#x_test1_image.show()

#print(y_train[1:20])

#a = np.array([2,1, 0 ,4 ,1, 4, 9, 2, 9, 0, 6, 9, 0, 1, 3, 9, 7, 8, 4])
#print(a)
#print(a[:3])
#Y_train = (numpy.arange(10) == y_train[:, None])
#Y_test =(numpy.arange(10) == y_test[:, None])
#print("Y_test.shape: ",Y_test.shape)

#model = load_model('keras_train_minist_itsselfdata_train.h5')
#y_test_pred = model.predict_classes(X_test, verbose=0)
#print("y_test_pred.shape: ", y_test_pred.shape)


#test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
#print('Test accuracy: %.8f%%' % (test_acc * 100))

#y_train_pred = model.predict_classes(X_train, verbose=0)
#print(y_train_pred[0:3])
#print('First 3 predictions: ', y_train_pred[:3])

