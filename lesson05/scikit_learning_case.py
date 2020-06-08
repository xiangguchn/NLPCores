# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:56:30 2020

@author: xiangguchn
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# predict result with reg for x data
def f(x,reg): 
    return reg.coef_ * x + reg.intercept_

# generate train data of X and Y
X = np.linspace(1, 20, 20) + np.random.rand(20)
Y = np.linspace(1, 20, 20)*2 + np.random.rand(20)

# train linear regression model to get coef_ and intercept_
reg = LinearRegression().fit(X.reshape(-1, 1), Y)

# plot X and Y and show the linear regression result
plt.scatter(X, Y)
plt.plot(X, f(X,reg), color='red')
plt.xlabel('X')
plt.ylabel('Y')
# show the regression formular
plt.text(X[1], Y[-2], r'Y = '+str(reg.coef_)+'X + '+str(reg.intercept_))
plt.show()