# NLP - Bag of words

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) # tsv specific, quiting skips ""

# cleaning the texts
# stemming
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
# import stemmer
from nltk.stem.porter import PorterStemmer
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
review = review.lower()
review = review.split()
# call stemmer
ps = PorterStemmer()
# update the loop
review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
# for larger texts use below to create a set:
# review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

