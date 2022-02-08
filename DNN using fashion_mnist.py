import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def sigmoid_derv(s):
    return s * (1 - s)

def ReLU(x):
  return max(x,0)

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

def to_one_hot(Y):
    n_col = np.amax(Y) + 1
    binarized = np.zeros((len(Y),n_col))
    for i in range(len(Y)):
        binarized[i,Y[i]] = 1.
    # print(binarized)
    return binarized

def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

mnist_dataset = tf.keras.datasets.fashion_mnist.load_data()
digits = datasets.load_digits()

(x_train, y_train), (x_test, y_test) = mnist_dataset

# x_train = x_train.reshape(-1, 28,28, 1)
# x_test = x_test.reshape(-1, 28,28, 1)

x_train = x_train.reshape(60000,-1)
x_test = x_test.reshape(10000,-1)

xx = normalize(x_train) ; xxt = normalize(x_test)
yy = to_one_hot(y_train) ; yyt = to_one_hot(y_test)

class MyNN:
    def __init__(self, x, y):
        self.x = x
        neurons = 64
        self.lr = 0.5
        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        self.w1 = np.random.randn(ip_dim, neurons)
        self.b1 = np.zeros((1, neurons))
        self.w2 = np.random.randn(neurons, neurons)
        self.b2 = np.zeros((1, neurons))
        self.w3 = np.random.randn(neurons, op_dim)
        self.b3 = np.zeros((1, op_dim))
        self.y = y

    def feedforward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)

    def backprop(self):
        loss = error(self.a3, self.y)
        print('Error :', loss)
        a3_delta = cross_entropy(self.a3, self.y) # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.a2) # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derv(self.a1) # w1

        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

    def predict(self, data):
        self.x = data
        self.feedforward()
        # print(f'Predicted values are : {self.a3.argmax()}')
        return self.a3.argmax()

model = MyNN(xx,yy)

epochs = 1000
for x in range(epochs):
    model.feedforward()
    model.backprop()

def get_acc(x, y):
    acc = 0
    for xx,yy in zip(x, y):
        s = model.predict(xx)
        # print(f'The predicted value is {s}')
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100

# xt = np.array([6.4,3.2,5.3,2.3]) ; xt = normalize(xt)
# yt = np.array([2]) ; yt = to_one_hot(yt)

print("Training accuracy : ", get_acc(xx,yy))
print("Test accuracy : ", get_acc(xxt, yyt))
