# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 09:05:48 2018

@author: shaksuma
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from array import array

#more than 5 is distracted
f_hours = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:1,13:1,14:0,15:0,16:0,17:1,18:0,19:0,20:0,21:0,22:0,23:0}

x_train = np.asarray(f_hours.keys()).reshape(-1,1)
y_train = f_hours.values()

#print x_train

knn = KNeighborsClassifier(n_neighbors=5)
## Fit the model on the training data.
knn.fit(x_train, y_train)
print knn.predict(20)
