# K-NN

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')


## DATA PREPROCESSING


# creating matrix of features [lines:lines,columns:columns]
X = dataset.iloc[:, [2, 3]].values # not [:,1] bc we want a matrix for X!
# creating dependent variable vector
y = dataset.iloc[:, 4].values


# splitting dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) # usually 0.2-0.3

# stanadarisation: xst = x - mean(x)/st dev (x)
# normalisation: xnorm = x - min(x)/max(x) - min(x)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)

## END DATA PREPROCESSING

## MODELLING
# Fitting KNN to the Training Set
# import libraries
from sklearn.neighbors import KNeighborsClassifier
# call classifier 
classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p =2)
# fit classifier into training data, to learn the correlation
classifier.fit(X_train, y_train)

# Predicting Test Set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix - Evaluate if the model works well
from sklearn.metrics import confusion_matrix # function
cm = confusion_matrix(y_test, y_pred)
# call cm from console - 64 cor, 4 uncor, 3 uncor, 29 cor sum =100


## VISUALISATION
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.15, cmap = ListedColormap(('orange', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = [ListedColormap(('orange', 'blue'))(i)], label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.25, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = [ListedColormap(('red', 'green'))(i)], label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()