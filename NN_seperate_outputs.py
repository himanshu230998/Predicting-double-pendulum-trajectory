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


#create 8 different data sets that each have a target value beingi one of the features


train = np.zeros([len(dataset),8])
target_x1 = np.zeros([len(dataset),1])
target_y1 = np.zeros([len(dataset),1])
target_x2 = np.zeros([len(dataset),1])
target_y2 = np.zeros([len(dataset),1])
target_v1x = np.zeros([len(dataset),1])
target_v2x = np.zeros([len(dataset),1])
target_v1y = np.zeros([len(dataset),1])
target_v2y = np.zeros([len(dataset),1])

iter = 0
for i in dataset:
    first = i[0]
    train[iter] = first
    second_1 = i[1][0]
    second_2 = i[1][0]
    second_3 = i[1][0]
    second_4 = i[1][0]
    second_5 = i[1][0]
    second_6 = i[1][0]
    second_7 = i[1][0]
    second_8 = i[1][0]

    target_x1[iter] = second_1
    target_y1[iter] = second_2
    target_x2[iter] = second_3
    target_y2[iter] = second_4
    target_v1x[iter] = second_5
    target_v2x[iter] = second_6
    target_v1y[iter] = second_7
    target_v2y[iter] = second_8

    iter += 1

train_data = pd.DataFrame(train)
target_data_1 = pd.DataFrame(target_x1)
target_data_2 = pd.DataFrame(target_x1)
target_data_3 = pd.DataFrame(target_x1)
target_data_4 = pd.DataFrame(target_x1)
target_data_5 = pd.DataFrame(target_x1)
target_data_6 = pd.DataFrame(target_x1)
target_data_7 = pd.DataFrame(target_x1)
target_data_8 = pd.DataFrame(target_x1)


train_data.columns = ["x1","y1","x2","y2","v_x1","v_x2","v_y1","v_y2"]
target_data_1.columns = ["x1_tar"]
target_data_2.columns = ["y1_tar"]
target_data_3.columns = ["x2_tar"]
target_data_4.columns = ["y2_tar"]
target_data_5.columns = ["v_x1_tar"]
target_data_6.columns = ["v_x2_tar"]
target_data_7.columns = ["v_y1_tar"]
target_data_8.columns = ["v_y2_tar"]

#NN for predicting X1

pendulum_NN_x1 = Sequential()

#input layer
pendulum_NN_x1.add(Dense(8, kernel_initializer='normal',input_dim = train_data.shape[1], activation='relu'))

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

pendulum_NN_x1.fit(train_data, target_data_1, epochs=100, batch_size=100, validation_split = 0.2, callbacks=callbacks_list)

##NN for predicting y1

pendulum_NN_y1 = Sequential()

#input layer
pendulum_NN_y1.add(Dense(8, kernel_initializer='normal',input_dim = train_data.shape[1], activation='relu'))

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

pendulum_NN_y1.fit(train_data, target_data_2, epochs=100, batch_size=100, validation_split = 0.2,
                   callbacks=callbacks_list_2)


##NN for prediction x2

pendulum_NN_x2 = Sequential()

#input layer
pendulum_NN_x2.add(Dense(8, kernel_initializer='normal',input_dim = train_data.shape[1], activation='relu'))

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

pendulum_NN_x2.fit(train_data, target_data_3, epochs=100, batch_size=100, validation_split = 0.2,
                   callbacks=callbacks_list_3)


##NN for prediction y2

pendulum_NN_y2 = Sequential()

#input layer
pendulum_NN_y2.add(Dense(8, kernel_initializer='normal',input_dim = train_data.shape[1], activation='relu'))

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

pendulum_NN_y2.fit(train_data, target_data_4, epochs=100, batch_size=100, validation_split = 0.2, callbacks=callbacks_list_4)


##NN for prediction v1x

pendulum_NN_v1x = Sequential()

#input layer
pendulum_NN_v1x.add(Dense(8, kernel_initializer='normal',input_dim = train_data.shape[1], activation='relu'))

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

pendulum_NN_v1x.fit(train_data, target_data_5, epochs=100, batch_size=100, validation_split = 0.2,
                    callbacks=callbacks_list_5)


##NN for prediction v2x

pendulum_NN_v2x = Sequential()

#input layer
pendulum_NN_v2x.add(Dense(8, kernel_initializer='normal',input_dim = train_data.shape[1], activation='relu'))

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

pendulum_NN_x1.fit(train_data, target_data_6, epochs=100, batch_size=100, validation_split = 0.2,
                   callbacks=callbacks_list_6)


##NN for prediction v1y

pendulum_NN_v1y = Sequential()

#input layer
pendulum_NN_v1y.add(Dense(8, kernel_initializer='normal',input_dim = train_data.shape[1], activation='relu'))

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

pendulum_NN_v1y.fit(train_data, target_data_7, epochs=100, batch_size=100, validation_split = 0.2,
                    callbacks=callbacks_list_7)


##NN for prediction v2y

pendulum_NN_v2y = Sequential()

#input layer
pendulum_NN_v2y.add(Dense(8, kernel_initializer='normal',input_dim = train_data.shape[1], activation='relu'))

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

pendulum_NN_v2y.fit(train_data, target_data_8, epochs=100, batch_size=100, validation_split = 0.2, callbacks=callbacks_list_8)



