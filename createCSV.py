import csv
import scipy.integrate as integrate
from numpy import sin, cos
import numpy as np

angle1 = 120
angle2 = 0
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

dt = 0.01
t = np.arange(0.0, 50 , dt)
state = np.radians([angle1, 0, angle2, 0])
y = integrate.odeint(derivs, state, t)
x1 = 1 * sin(y[:, 0])
y1 = -1 * cos(y[:, 0])
x2 = 1 * sin(y[:, 2]) + x1
y2 = -1 * cos(y[:, 2]) + y1


with open('x_2-120-0.csv', 'w',newline = '') as csvfile:
    filewriter = csv.writer(csvfile,quoting=csv.QUOTE_ALL)
    for i in range(len(t)):
        filewriter.writerow([t[i],x2[i]])


