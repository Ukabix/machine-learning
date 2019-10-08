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
# create vector of predictions of dep var
y_pred = regressor.predict(X_test)

# Visualising the TRAINING set results
plt.scatter(X_train, y_train, color = 'red')
# regressor.predict(X_train) bc we want to predict y for X_train var set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the TEST set results
plt.scatter(X_test, y_test, color = 'red')
# regressor already trained!
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()