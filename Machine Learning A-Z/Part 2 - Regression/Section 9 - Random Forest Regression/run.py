# Random Forest Regression

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Position_Salaries.csv')


## DATA PREPROCESSING


# creating matrix of features [lines:lines,columns:columns]
X = dataset.iloc[:, 1:2].values # not [:,1] bc we want a matrix for X!
# creating dependent variable vector
y = dataset.iloc[:, 2].values


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


## END DATA PREPROCESSING


# START MODEL DESIGN


# Fitting Regression Model to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 25000, random_state = 0)
regressor.fit(X, y)


# END MODEL DESIGN

# START RESULT PREDICTION

# Predicting a new result with PolyReg
y_pred = regressor.predict([[6.5]])
# Out[49]: array([158862.45265157])


# START VISUALISATION


# Visualising RFR results
# START HIRES VISUAL # !remember X_grid assignments for plt.plot
X_grid = np.arange(min(X), max(X), 0.01) # output: vector 1-9.0,incrim 0.1
X_grid = X_grid.reshape(len(X_grid), 1) # output: 1 col matrix of ^
# END HIRES VISUAL
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('truth or bluff (RFR)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# noncontinious model again!

# END VISUALISATION

