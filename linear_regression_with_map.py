from pendulum_simulation import X1, X2, Y1, Y2, t, theta1, theta2
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import csv


def createXVals(x2, y1, y2, t):
    ret_x = np.zeros([np.shape(x2)[0], 4])
    for i in range(np.shape(ret_x)[0]):
        ret_x[i] = [x2[i], y1[i], y2[i], t[i]]

    return ret_x


results = np.zeros([len(X1), 10])
for i in range(len(X1)):
    # first create X array
    X_vals = createXVals(X2[i], Y1[i], Y2[i], t)
    Y_vals = X1[i]

    print("The inital theta for bob 1 is {0}".format(theta1[i]))
    print("The inital theta for bob 2 is {0}".format(theta2[i]))
    print("The min of the test set is {0}".format(np.amin(Y_vals)))
    print("The max of the test set is {0}".format(np.amax(Y_vals)))

    X_train = X_vals[:9000]
    X_test = X_vals[9000:]
    Y_train = Y_vals[:9000]
    Y_test = Y_vals[9000:]

    #try quadratic and cubic
    poly = PolynomialFeatures(7)

    X_train_poly = poly.fit_transform(X_train)
    print(X_train_poly)
    print(np.shape(X_train_poly))
    X_test_poly = poly.fit_transform(X_test)
    print(X_test_poly)
    print(np.shape(X_test_poly))

    LinReg = LinearRegression()
    LinReg.fit(X_train_poly, Y_train)

    y_pred = LinReg.predict(X_test_poly)
    y_train_pred = LinReg.predict(X_train_poly)

    RMSE_test = mean_squared_error(Y_test, y_pred)
    RMSE_train = mean_squared_error(Y_train, y_train_pred)
    print("The RMSE of the test set is {0}".format(RMSE_test))
    print("The RMSE of the train set is {0}".format(RMSE_train))

    MAE_test = mean_absolute_error(Y_test, y_pred)
    MAE_train = mean_absolute_error(Y_train, y_train_pred)
    print("The MAE of the test set is {0}".format(MAE_test))
    print("The MAE of the train set is {0}".format(MAE_train))

    R2_test = r2_score(Y_test, y_pred)
    R2_train = r2_score(Y_train, y_train_pred)
    print("The R2 Score of the test set is {0}".format(R2_test))
    print("The R2 Score of the train set is {0}".format(R2_train))
    print("\n")

    results[i] = [theta1[i], theta2[i], np.amin(Y_vals), np.amax(Y_vals), RMSE_test, RMSE_train, MAE_test, MAE_train,
                  R2_test, R2_train]

results_df = pd.DataFrame(results)
results_df.columns = ['Theta of Bob 1', 'Theta of Bob 2', 'Min of X1', 'Max of X1', "RMSE of test", "RMSE of train",
                      "MAE of test", "MAE of train", "R2 of test", "R2 of train"]
results_df.to_csv('Linear Regression With Degree 7 Feature Map.csv', index=False)
print(results_df)