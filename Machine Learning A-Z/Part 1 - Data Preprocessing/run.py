# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# import dataset
dataset = pd.read_csv('Data.csv')
# creating matrix of features [lines:lines,columns:columns]
X = dataset.iloc[:,:-1].values
# creating dependent variable vector
y = dataset.iloc[:, 3].values

# handling missing data
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
# put put missing data into matrix of features
missingvalues = missingvalues.fit(X[:, 1:3])
X[:, 1:3] = missingvalues.transform(X[:, 1:3])

print(X)
print(y)