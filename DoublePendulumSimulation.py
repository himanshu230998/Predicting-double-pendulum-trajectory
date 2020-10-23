from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pickle
from itertools import *


"""
import matplotlib.animation as animation
import random
import matplotlib.patches as mpatches

from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import r2_score
import pandas as pd
"""
# foe each experiment value of l1,l2,m1,m2 and th1,th2,w1,w2 are same so explicitely add these features after training.

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


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

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=1)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text
# Generating data set

# create a time array from 0..100 sampled at 0.05 second steps




#ani.save('double_pendulum.mp4', fps=15)
#plt.show()


# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)

#state = np.radians([4, 0, -17, 0])
#y = integrate.odeint(derivs, state, t)
#ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                                  #interval=15, blit=True, init_func=init)
#plt.show()
dt = 0.01
t = np.arange(0.0, 20 , dt)
theta1 = np.random.randint(-150,150,size = 5000)
theta2 = np.random.randint(-150,150,size = 5000)
# hard code it here,  can change to random
w1 = 0.0
w2 = 0.0

# list of list storing timeseries for different initial conditions (10 by 10000)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return list(zip(a, b))

dataset = []
for i in range(0,len(theta1)):
    state = np.radians([theta1[i], w1, theta2[i], w2])
    y = integrate.odeint(derivs, state, t)
    x1 = L1*sin(y[:, 0])
    y1 = -L1*cos(y[:, 0])
    x2 = L2*sin(y[:, 2]) + x1
    y2 = -L2*cos(y[:, 2]) + y1
    v_x1 = np.diff(x1)
    v_x1 = np.insert(v_x1,obj=0,values=0)
    v_x1 = v_x1/dt
    v_x2 = np.diff(x2)
    v_x2 = np.insert(v_x2,obj=0,values=0)
    v_x2 = v_x2/dt
    v_y1  = np.diff(y1)
    v_y1 = np.insert(v_y1,obj=0,values=0)
    v_y1 = v_y1/dt
    v_y2 = np.diff(y2)
    v_y2 = np.insert(v_y2,obj=0,values=0)
    v_y2 = v_y2/dt
    matrix = np.array([x1])
    matrix = np.append(matrix,[y1],axis =0 )
    matrix = np.append(matrix, [x2], axis=0)
    matrix = np.append(matrix,[y2] , axis=0)
    matrix = np.append(matrix, [v_x1], axis=0)
    matrix = np.append(matrix,[v_x2], axis=0)
    matrix = np.append(matrix, [v_y1], axis=0)
    matrix = np.append(matrix, [v_y2], axis=0)
    data = matrix.T
    pair = pairwise(data)
    dataset += pair

print(len(dataset))

#dataset is list of tuples that contains pair of two consecutive states
with open('arbitraryangle.pickle', 'wb') as fp:
    pickle.dump(dataset, fp)

# list of list storing timeseries for different initial conditions (10 by 10000)


