from DoublePendulumSimulation import dataset
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

os.environ['KMP_DUPLICATE_LIB_OK']='True'

print(theta1_0)
print(theta2_0)


train = np.zeros([len(dataset),8])
target = np.zeros([len(dataset),8])
iter = 0
for i in dataset:
    first = i[0]
    train[iter] = first
    second = i[1]
    target[iter] = second
    iter += 1

train_data = pd.DataFrame(train)
target_data = pd.DataFrame(target)

train_data.columns = ["x1","y1","x2","y2","v_x1","v_x2","v_y1","v_y2"]
target_data.columns = ["x1_tar","y1_tar","x2_tar","y2_tar","v_x1_tar","v_x2_tar","v_y1_tar","v_y2_tar"]

pendulum_NN = Sequential()

#input layer
pendulum_NN.add(Dense(8, kernel_initializer='normal',input_dim = train_data.shape[1], activation='relu'))

#hidden layers
#pendulum_NN.add(Dense(8, kernel_initializer='normal',activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
pendulum_NN.add(Dense(8, kernel_initializer='normal',activation='relu'))
pendulum_NN.add(Dense(8, kernel_initializer='normal',activation='relu' ))
#pendulum_NN.add(Dense(8 , kernel_initializer='normal',activation='relu'))
# The Output Layer :
pendulum_NN.add(Dense(8, kernel_initializer='normal',activation='linear'))


# Compile the network :
pendulum_NN.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
pendulum_NN.summary()

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}-five1.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

pendulum_NN.fit(train_data, target_data, epochs=100, batch_size=100, validation_split = 0.2, callbacks=callbacks_list)



#pendulum_NN.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


