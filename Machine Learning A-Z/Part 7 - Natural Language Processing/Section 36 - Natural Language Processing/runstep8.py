# NLP - Bag of words - Mst common NB and RTC

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) # tsv specific, quiting skips ""


# cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    # larger set: review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # joining the words of the review seperated by space
    review = ' '.join(review)
    corpus.append(review)

    
# Creating Bag of Words model
# sparsity reduction
from sklearn.feature_extraction.text import CountVectorizer
# we have 1000 columns and 1564 words, let's take the most common 1500
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
# include indep V to make a vector
y = dataset.iloc[:, 1].values


# !import classification template - NB here
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling - dont need here
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
