from numpy import sin, cos
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.integrate as integrate
#import pickle
#from itertools import *
# from math import sqrt
# from numpy import concatenate
# from matplotlib import pyplot
# from pandas import read_csv
# from pandas import DataFrame
# from pandas import concat
# from sklearn.prep rocessing import MinMaxScaler
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from DoublePendulumSimulation import timeSeries

angle1 = 75
angle2 = -45
G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


"""
dt = 0.01
t = np.arange(0.0, 100 , dt)
train_steps = 8000
value = timeSeries(angle1,angle2,0,0,t)
dataset = value.T
train = dataset[:train_steps]
test = dataset[train_steps:]
"""
def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
    dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_) +
               M2*G*sin(state[2])*cos(del_) +
               M2*L2*state[3]*state[3]*sin(del_) -
               (M1 + M2)*G*sin(state[0]))/den1

    dydx[2] = state[3]

    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_) +
               (M1 + M2)*G*sin(state[0])*cos(del_) -
               (M1 + M2)*L1*state[1]*state[1]*sin(del_) -
               (M1 + M2)*G*sin(state[2]))/den2

    return dydx


dt = 0.01
t = np.arange(0.0, 100 , dt)
state = np.radians([angle1, 0, angle2, 0])
y = integrate.odeint(derivs, state, t)
x1 = 1 * sin(y[:, 0])
y1 = -1 * cos(y[:, 0])
x2 = 1 * sin(y[:, 2]) + x1
y2 = -1 * cos(y[:, 2]) + y1

train_time = 9000
trainX = t[:9000]
trainY = y2[:9000]
testX = t[9000:]
testY =y2[9000:]

trainX = np.array(trainX)
testX = np.array(testX)
trainX = trainX.reshape((9000,1,1))
testX = testX.reshape((1000,1,1))
print(trainX)
print(trainX.shape)
print(trainX.shape)
print(trainY.shape)
model = Sequential()
model.add(LSTM(50, input_shape=(1,1 )))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(trainX, trainY, epochs=50, batch_size=72, validation_data=(testX, testY), verbose=2, shuffle=False)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
y_predict = model.predict(testX)

print(y_predict)
print(testY)
plt.plot(testX,testY)
plt.plot(testX,y_predict)
# train
