import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Activation, Flatten
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Sequential

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import os

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from itertools import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#NN for predicting X1

pendulum_NN_x1 = Sequential()

#input layer
pendulum_NN_x1.add(Dense(8, kernel_initializer='normal',input_dim = 8, activation='relu'))

#hidden layers
#pendulum_NN.add(Dense(8, kernel_initializer='normal',activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
pendulum_NN_x1.add(Dense(8, kernel_initializer='normal',activation='relu'))
pendulum_NN_x1.add(Dense(8, kernel_initializer='normal',activation='relu' ))
#pendulum_NN.add(Dense(8 , kernel_initializer='normal',activation='relu'))

# The Output Layer :
pendulum_NN_x1.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
pendulum_NN_x1.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
pendulum_NN_x1.summary()

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}-x1.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

wights_file='Weights-049--0.00099-x1.hdf5'  # choose the best checkpoint
pendulum_NN_x1.load_weights(wights_file)  # load it
pendulum_NN_x1.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


##NN for predicting y1

pendulum_NN_y1 = Sequential()

#input layer
pendulum_NN_y1.add(Dense(8, kernel_initializer='normal',input_dim = 8, activation='relu'))

#hidden layers
#pendulum_NN.add(Dense(8, kernel_initializer='normal',activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
pendulum_NN_y1.add(Dense(8, kernel_initializer='normal',activation='relu'))
pendulum_NN_y1.add(Dense(8, kernel_initializer='normal',activation='relu' ))
#pendulum_NN.add(Dense(8 , kernel_initializer='normal',activation='relu'))

# The Output Layer :
pendulum_NN_y1.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
pendulum_NN_y1.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
pendulum_NN_y1.summary()

checkpoint_name_2 = 'Weights-{epoch:03d}--{val_loss:.5f}-y1.hdf5'
checkpoint_2 = ModelCheckpoint(checkpoint_name_2, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list_2 = [checkpoint_2]

wights_file_2='Weights-086--0.00076-y1.hdf5'  # choose the best checkpoint
pendulum_NN_y1.load_weights(wights_file_2)  # load it
pendulum_NN_y1.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])



##NN for prediction x2

pendulum_NN_x2 = Sequential()

#input layer
pendulum_NN_x2.add(Dense(8, kernel_initializer='normal',input_dim = 8, activation='relu'))

#hidden layers
#pendulum_NN.add(Dense(8, kernel_initializer='normal',activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
pendulum_NN_x2.add(Dense(8, kernel_initializer='normal',activation='relu'))
pendulum_NN_x2.add(Dense(8, kernel_initializer='normal',activation='relu' ))
#pendulum_NN.add(Dense(8 , kernel_initializer='normal',activation='relu'))

# The Output Layer :
pendulum_NN_x2.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
pendulum_NN_x2.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
pendulum_NN_x2.summary()

checkpoint_name_3 = 'Weights-{epoch:03d}--{val_loss:.5f}-x2.hdf5'
checkpoint_3 = ModelCheckpoint(checkpoint_name_3, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list_3 = [checkpoint_3]

wights_file_3='Weights-080--0.00091-x2.hdf5'  # choose the best checkpoint
pendulum_NN_x2.load_weights(wights_file_3)  # load it
pendulum_NN_x2.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])



##NN for prediction y2

pendulum_NN_y2 = Sequential()

#input layer
pendulum_NN_y2.add(Dense(8, kernel_initializer='normal',input_dim = 8, activation='relu'))

#hidden layers
#pendulum_NN.add(Dense(8, kernel_initializer='normal',activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
pendulum_NN_y2.add(Dense(8, kernel_initializer='normal',activation='relu'))
pendulum_NN_y2.add(Dense(8, kernel_initializer='normal',activation='relu' ))
#pendulum_NN.add(Dense(8 , kernel_initializer='normal',activation='relu'))

# The Output Layer :
pendulum_NN_y2.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
pendulum_NN_y2.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
pendulum_NN_y2.summary()

checkpoint_name_4 = 'Weights-{epoch:03d}--{val_loss:.5f}-y2.hdf5'
checkpoint_4 = ModelCheckpoint(checkpoint_name_4, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list_4 = [checkpoint_4]

wights_file_4='Weights-072--0.00071-y2.hdf5'  # choose the best checkpoint
pendulum_NN_y2.load_weights(wights_file_4)  # load it
pendulum_NN_y2.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


##NN for prediction v1x

pendulum_NN_v1x = Sequential()

#input layer
pendulum_NN_v1x.add(Dense(8, kernel_initializer='normal',input_dim = 8, activation='relu'))

#hidden layers
#pendulum_NN.add(Dense(8, kernel_initializer='normal',activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
pendulum_NN_v1x.add(Dense(8, kernel_initializer='normal',activation='relu'))
pendulum_NN_v1x.add(Dense(8, kernel_initializer='normal',activation='relu' ))
#pendulum_NN.add(Dense(8 , kernel_initializer='normal',activation='relu'))

# The Output Layer :
pendulum_NN_v1x.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
pendulum_NN_v1x.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
pendulum_NN_v1x.summary()

checkpoint_name_5 = 'Weights-{epoch:03d}--{val_loss:.5f}-v1x.hdf5'
checkpoint_5 = ModelCheckpoint(checkpoint_name_5, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list_5 = [checkpoint_5]

wights_file_5 = 'Weights-069--0.00082-v1x.hdf5'  # choose the best checkpoint
pendulum_NN_v1x.load_weights(wights_file_5)  # load it
pendulum_NN_v1x.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


##NN for prediction v2x

pendulum_NN_v2x = Sequential()

#input layer
pendulum_NN_v2x.add(Dense(8, kernel_initializer='normal',input_dim = 8, activation='relu'))

#hidden layers
#pendulum_NN.add(Dense(8, kernel_initializer='normal',activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
pendulum_NN_v2x.add(Dense(8, kernel_initializer='normal',activation='relu'))
pendulum_NN_v2x.add(Dense(8, kernel_initializer='normal',activation='relu' ))
#pendulum_NN.add(Dense(8 , kernel_initializer='normal',activation='relu'))

# The Output Layer :
pendulum_NN_v2x.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
pendulum_NN_v2x.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
pendulum_NN_v2x.summary()

checkpoint_name_6 = 'Weights-{epoch:03d}--{val_loss:.5f}-v2x.hdf5'
checkpoint_6 = ModelCheckpoint(checkpoint_name_6, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list_6 = [checkpoint_6]

wights_file_6='Weights-074--0.00078-v2x.hdf5'  # choose the best checkpoint
pendulum_NN_v2x.load_weights(wights_file_6)  # load it
pendulum_NN_v2x.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])



##NN for prediction v1y

pendulum_NN_v1y = Sequential()

#input layer
pendulum_NN_v1y.add(Dense(8, kernel_initializer='normal',input_dim = 8, activation='relu'))

#hidden layers
#pendulum_NN.add(Dense(8, kernel_initializer='normal',activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
pendulum_NN_v1y.add(Dense(8, kernel_initializer='normal',activation='relu'))
pendulum_NN_v1y.add(Dense(8, kernel_initializer='normal',activation='relu' ))
#pendulum_NN.add(Dense(8 , kernel_initializer='normal',activation='relu'))

# The Output Layer :
pendulum_NN_v1y.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
pendulum_NN_v1y.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
pendulum_NN_v1y.summary()

checkpoint_name_7 = 'Weights-{epoch:03d}--{val_loss:.5f}-v1y.hdf5'
checkpoint_7 = ModelCheckpoint(checkpoint_name_7, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list_7 = [checkpoint_7]

wights_file_7='Weights-076--0.00101-v1y.hdf5'  # choose the best checkpoint
pendulum_NN_v1y.load_weights(wights_file_7)  # load it
pendulum_NN_v1y.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])



##NN for prediction v2y

pendulum_NN_v2y = Sequential()

#input layer
pendulum_NN_v2y.add(Dense(8, kernel_initializer='normal',input_dim = 8, activation='relu'))

#hidden layers
#pendulum_NN.add(Dense(8, kernel_initializer='normal',activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
pendulum_NN_v2y.add(Dense(8, kernel_initializer='normal',activation='relu'))
pendulum_NN_v2y.add(Dense(8, kernel_initializer='normal',activation='relu' ))
#pendulum_NN.add(Dense(8 , kernel_initializer='normal',activation='relu'))

# The Output Layer :
pendulum_NN_v2y.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
pendulum_NN_v2y.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
pendulum_NN_v2y.summary()

checkpoint_name_8 = 'Weights-{epoch:03d}--{val_loss:.5f}-v2y.hdf5'
checkpoint_8 = ModelCheckpoint(checkpoint_name_8, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list_8 = [checkpoint_8]

wights_file_8='Weights-041--0.00080-v2y.hdf5'  # choose the best checkpoint
pendulum_NN_v2y.load_weights(wights_file_8)  # load it
pendulum_NN_v2y.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


# test data on random initial angle
theta1 = np.random.randint(-150, 150, size=1)
theta2 = np.random.randint(-150, 150, size=1)

dt = 0.01
t = np.arange(0.0, 20, dt)
# hard code it here,  can change to random
w1 = 0.0
w2 = 0.0


G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return list(zip(a, b))

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

dataset = []
for i in range(0, len(theta1)):
    state = np.radians([theta1[i], w1, theta2[i], w2])
    y = integrate.odeint(derivs, state, t)
    x1 = L1 * sin(y[:, 0])
    y1 = -L1 * cos(y[:, 0])
    x2 = L2 * sin(y[:, 2]) + x1
    y2 = -L2 * cos(y[:, 2]) + y1
    v_x1 = np.diff(x1)
    v_x1 = np.insert(v_x1, obj=0, values=0)
    v_x1 = v_x1 / dt
    v_x2 = np.diff(x2)
    v_x2 = np.insert(v_x2, obj=0, values=0)
    v_x2 = v_x2 / dt
    v_y1 = np.diff(y1)
    v_y1 = np.insert(v_y1, obj=0, values=0)
    v_y1 = v_y1 / dt
    v_y2 = np.diff(y2)
    v_y2 = np.insert(v_y2, obj=0, values=0)
    v_y2 = v_y2 / dt
    matrix = np.array([x1])
    matrix = np.append(matrix, [y1], axis=0)
    matrix = np.append(matrix, [x2], axis=0)
    matrix = np.append(matrix, [y2], axis=0)
    matrix = np.append(matrix, [v_x1], axis=0)
    matrix = np.append(matrix, [v_x2], axis=0)
    matrix = np.append(matrix, [v_y1], axis=0)
    matrix = np.append(matrix, [v_y2], axis=0)
    data = matrix.T
    pair = pairwise(data)
    dataset += pair

time_series_for_angle=[]

num =0
very_first=None
for i in dataset:
    first=i[0]
    second=i[1]
    if num == 0:
        very_first=first
        time_series_for_angle.append(first)
    time_series_for_angle.append(second)
    num += 1

number_times_to_run = len(time_series_for_angle) - 1


# build NN time series for angle
NN_time_series = []
predict_single = np.zeros([1, 8])
predict_single[0] = very_first
NN_time_series.append(very_first)

x_1 = []
for i in range(number_times_to_run):
    predictions_1 = pendulum_NN_x1.predict(predict_single)
    predictions_2 = pendulum_NN_y1.predict(predict_single)
    predictions_3 = pendulum_NN_x2.predict(predict_single)
    predictions_4 = pendulum_NN_y2.predict(predict_single)
    predictions_5 = pendulum_NN_v1x.predict(predict_single)
    predictions_6 = pendulum_NN_v2x.predict(predict_single)
    predictions_7 = pendulum_NN_v1y.predict(predict_single)
    predictions_8 = pendulum_NN_v2y.predict(predict_single)



    predictions_output = [predictions_1[0],predictions_2,predictions_3,predictions_4,predictions_5,predictions_6,predictions_7,predictions_8]
    predict_single[0] = predictions_output
    NN_time_series.append(predictions_output)

print(NN_time_series[1])


# plot both the actual time series vs the NN given one
actual_bob_2_x = []
actual_bob_2_y = []
predicted_bob_2_x = []
predicted_bob_2_y = []
for i in time_series_for_angle:
    bob_2_x = i[2]
    actual_bob_2_x.append(bob_2_x)
    bob_2_y = i[3]
    actual_bob_2_y.append(bob_2_y)



for i in NN_time_series:
    bob_2_x = i[2]
    predicted_bob_2_x.append(bob_2_x)
    bob_2_y = i[3]
    predicted_bob_2_y.append(bob_2_y)




plt.plot(actual_bob_2_x, actual_bob_2_y, label='actual time series')
plt.plot(predicted_bob_2_x, predicted_bob_2_y, label='predicted time series')
plt.legend()
plt.show()


# output MAE
print(mean_absolute_error(time_series_for_angle, NN_time_series))
