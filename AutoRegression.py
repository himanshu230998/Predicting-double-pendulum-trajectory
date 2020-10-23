from pendulum_simulation import X1,X2,Y1,Y2,t,theta1,theta2
from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import random
import matplotlib.patches as mpatches

from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import r2_score
import pandas as pd



x1 = X1[0]
t = np.arange(0, 10000, 1)

X_train = x1[0:8000]
date_train = t[0:10000]
model = ARMA(X_train,order=(1,0))
res = model.fit()
res.plot_predict(start=0,end = 10000)
blue_patch = mpatches.Patch(color='blue', label='prediction')
green_patch = mpatches.Patch(color='green', label='simulation')
plt.legend(handles=[blue_patch,green_patch])
plt.plot(t,x1,label = "simulated value")
plt.xlabel('X1')
plt.ylabel('time')
plt.show()
#plot data set