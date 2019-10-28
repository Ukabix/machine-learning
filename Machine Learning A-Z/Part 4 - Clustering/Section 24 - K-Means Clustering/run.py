# K-MEANS CLUSTERING

#%reset -f

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Mall_Customers.csv')


# creating matrix of features [lines:lines,columns:columns]
X = dataset.iloc[:, [3,4]].values # not [:,1] bc we want a matrix for X!


# using the elbow method to find the optimal num of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
# plotting onto graph
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('num of clusters')
plt.ylabel('wcss')
plt.show()


# Apllying k-means to the dataset - fit_predict method
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans= kmeans.fit_predict(X)


# Visualisation of clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'cluster 1 - careful')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'cluster 2 - standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'cluster 3 - target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'cluster 4 - careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'cluster 5 - sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'centroids')
plt.title('clusters of clients')
plt.xlabel('annual income (k$)')
plt.ylabel('spending score: 1-100')
plt.legend()
plt.show()

