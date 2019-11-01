# Artificial Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Part 1 - Data Preprocessing

### estimate the problem approach and use an appropriate template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3 : 13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling ! always for ANNs
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### Part 2 - let's make an ANN
# import keras and modules
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
# create ANN object
classifier  = Sequential()

# Add the input layer and 1st hidden layer - 
# step 1: use Dense method to initialise weights ->0
# step 2: input observation of dataset in the input layer, each for one input node
# step 2: here input nodes = 11 (11 indep vars)
# step 3: forward propagation - choose activator function
# step 3: here - rectifier function for I, sigmoid (1 cat) or softmax(2+ cats) for O
# step 4: compare pred results to the actual result. Measure the error.
# step 5: Backpropagation of error, update the weights
# step 6: Repeat 1-5 and update weights: after each observation (reinforcement) or via batch
# step 7: training set done = 1 epoch, redo more epochs

# choose nr of input layers - tip: avg of (input + output) nodes, activation = relu
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# add second hidden layer 
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# add ouput layer - sigmoid
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Part 3 - Making the prediction and model evaluation

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

