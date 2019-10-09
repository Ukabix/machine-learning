# Polynomial Regression

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# import dataset
dataset = pd.read_csv('Position_Salaries.csv')

# DATA PREPROCESSING
# creating matrix of features [lines:lines,columns:columns]
X = dataset.iloc[:, 1:2].values # not [:,1] bc we want a matrix for X!
# creating dependent variable vector
y = dataset.iloc[:, 2].values


# not enough data to actually train + test and we want to be as precise as possible
# so we are skipping next step
# splitting dataset into Training and Test sets
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # usually 0.2-0.3'''

# stanadarisation: xst = x - mean(x)/st dev (x)
# normalisation: xnorm = x - min(x)/max(x) - min(x)
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
# END DATA PREPROCESSING

# START MODEL DESIGN
# Fitting Multiple Linear Regression

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2) # degree!
X_poly = poly_reg.fit_transform(X) # generates ones at [0] and at [2] X^deg array
# ^ we are mounting our X_poly matrix into LinReg model as below
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y) #38

# END MODEL DESIGN

# START VISUALISATION
# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue') # 2nd arg => y taken from lin_reg
plt.title('truth or bluff (LinReg)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()
# !!!!! pretty interesting - red = real observations, blue = predictions

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
# ^ IT'S A TRAP! poly_reg.fit_transform(X) == X_poly this time, but we can fit any other poly function here -> good practice
plt.title('truth or bluff (PolyReg)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# END VISUALISATION