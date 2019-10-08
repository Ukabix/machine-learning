# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# import dataset
dataset = pd.read_csv('Salary_Data.csv')
# creating matrix of features [lines:lines,columns:columns] - ind variables
X = dataset.iloc[:,:-1].values
# creating dependent variable vector
y = dataset.iloc[:, 1].values


# spliitting dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) # usually 0.2-0.3


#feature scaling - scalar problem and distance in Euclidean math frame of reference
# stanadarisation: xst = x - mean(x)/st dev (x)
# normalisation: xnorm = x - min(x)/max(x) - min(x)
# Feature Scaling - not really needed for linear regr
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# prediction of test set results
