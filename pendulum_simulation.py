from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import random
import random

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



#ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
#                            interval=15, blit=True, init_func=init)
# ani.save('double_pendulum.mp4', fps=15)
#plt.show()

dt = 0.01
t = np.arange(0.0, 100, dt)
# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)


#theta1 = np.random.randint(-20,20,size = 10)
#theta2 = np.random.randint(-20,20,size = 10)
# hard code it here,  can change to random
theta1 = [ 8 ,-17 ,  4 , -5, -20  , 3 ,  7, -19, -17, -18]
theta2 = [ 13, 12 , -17, -20 , 7 ,  2 ,-14, 10 , -8 , -6]
w1 = 0.0
w2 = 0.0
# list of list storing timeseries for different initial conditions (10 by 10000)
X1 = []
X2 = []
Y1 = []
Y2 = []
for i in range(0,len(theta1)):
    state = np.radians([theta1[i], w1, theta2[i], w2])
    y = integrate.odeint(derivs, state, t)
    x1 = L1*sin(y[:, 0])
    y1 = -L1*cos(y[:, 0])
    x2 = L2*sin(y[:, 2]) + x1
    y2 = -L2*cos(y[:, 2]) + y1
    X1.append(x1)
    X2.append(x2)
    Y1.append(y1)
    Y2.append(y2)


#list of list storing timeseries for different initial conditions (10 by 10000)
X1 = np.array(X1)
X2 = np.array(X2)
Y1 = np.array(Y1)
Y2 = np.array(Y2)
t = np.arange(0.0, 100, dt)

