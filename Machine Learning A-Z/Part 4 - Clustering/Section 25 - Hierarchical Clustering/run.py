# HIERARCHICAL CLUSTERING

%reset -f

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Mall_Customers.csv')


# creating matrix of features [lines:lines,columns:columns]
X = dataset.iloc[:, [3,4]].values # not [:,1] bc we want a matrix for X!


# using the dendrogram method to find the optimal num of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('Euclidean distances')
plt.show()
# answer is 5

# Fitting HC to the dataset - fit_predict
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


# Apllying k-means to the dataset - fit_predict method


# Visualisation of clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'cluster 1 - careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'cluster 2 - standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'cluster 3 - target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'cluster 4 - careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'cluster 5 - sensible')
plt.title('clusters of clients')
plt.xlabel('annual income (k$)')
plt.ylabel('spending score: 1-100')
plt.legend()
plt.show()

