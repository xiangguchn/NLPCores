# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 20:17:57 2020
How to implement a KNN model
@author: xiangguchn
"""

import numpy as np
from scipy.spatial.distance import cosine
from collections import Counter

def model(X, Y):
    return [(Xi, Yi) for Xi, Yi in zip(X, Y)]

# calculate distance between x1 and x2 with cosine
def distance(x1, x2):
    return cosine(x1, x2)

# predict most similars with knn
def predict(x, k=5):
    most_similars = sorted(model(X, Y), key=lambda xi: distance(xi[0], x))[:k]
    #code here
    target_list = [x[1] for x in most_similars]
    label = Counter(target_list).most_common([1][0])
    return label

# generate two random list and classify Y with Y >= 0.5
X = np.random.rand(20)
Y = np.random.rand(20)
Y = (Y >= 0.5)+0

# predict 
a = predict(0.5,5)
print(a)



