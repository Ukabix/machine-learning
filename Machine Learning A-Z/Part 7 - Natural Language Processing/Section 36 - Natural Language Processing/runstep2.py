# NLP - Bag of words

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) # tsv specific, quiting skips ""

# cleaning the texts
# removing irrelevant words
import re
# import lib
import nltk
# dl stopwords list
nltk.download('stopwords')
# import stopwords as a list
from nltk.corpus import stopwords
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
review = review.lower()
# split method
review = review.split()
# list comprehension
review = [word for word in review if not word in stopwords.words('english')]
# for larger texts use below to create a set:
# review = [word for word in review if not word in set(stopwords.words('english'))]