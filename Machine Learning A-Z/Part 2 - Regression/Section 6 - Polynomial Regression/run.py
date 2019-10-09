# Polynomial Regression

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# import dataset
dataset = pd.read_csv('Position_Salaries.csv')
# creating matrix of features [lines:lines,columns:columns]
X = dataset.iloc[:, 1:2].values # not [:,1] bc we want a matrix for X!
# creating dependent variable vector
y = dataset.iloc[:, 2].values


# spliitting dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # usually 0.2-0.3

# stanadarisation: xst = x - mean(x)/st dev (x)
# normalisation: xnorm = x - min(x)/max(x) - min(x)
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""



