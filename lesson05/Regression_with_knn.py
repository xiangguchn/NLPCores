# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 21:31:06 2020

@author: xiangguchn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

# generate data for train 
X = np.linspace(1, 20, 20) + np.random.rand(20)
Y = np.linspace(1, 20, 20)*2 + np.random.rand(20)
# 
Xp = np.linspace(1, 21, 500)[:, np.newaxis]

# Fit regression model
n_neighbors = 2

# knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
Yp = knn.fit(X.reshape(-1, 1), Y).predict(Xp)

# plot and show the result
plt.scatter(X, Y, color='darkorange', label='data')
plt.plot(Xp, Yp, color='navy', label='prediction')
plt.legend()
plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                'distance'))
plt.show()