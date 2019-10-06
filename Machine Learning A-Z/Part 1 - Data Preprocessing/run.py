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
# X == array([['France', 44.0, 72000.0],
#        ['Spain', 27.0, 48000.0],
#        ['Germany', 30.0, 54000.0],
#        ['Spain', 38.0, 61000.0],
#        ['Germany', 40.0, 63777.77777777778],
#        ['France', 35.0, 58000.0],
#        ['Spain', 38.77777777777778, 52000.0],
#        ['France', 48.0, 79000.0],
#        ['Germany', 50.0, 83000.0],
#        ['France', 37.0, 67000.0]], dtype=object)
# y == array(['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes'],
#       dtype=object)

# encoding categorical data

# Encoding the Independent Variable
# create OneHotEncoder object from OneHotEncoder class
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
# X == array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.40000000e+01,
#         7.20000000e+04],
#        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 2.70000000e+01,
#         4.80000000e+04],
#        [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 3.00000000e+01,
#         5.40000000e+04],
#        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 3.80000000e+01,
#         6.10000000e+04],
#        [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 4.00000000e+01,
#         6.37777778e+04],
#        [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.50000000e+01,
#         5.80000000e+04],
#        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 3.87777778e+01,
#         5.20000000e+04],
#        [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.80000000e+01,
#         7.90000000e+04],
#        [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 5.00000000e+01,
#         8.30000000e+04],
#        [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.70000000e+01,
#         6.70000000e+04]])
# 1/2/3 col = encoded country values

# Encoding Y data
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
# y == array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])