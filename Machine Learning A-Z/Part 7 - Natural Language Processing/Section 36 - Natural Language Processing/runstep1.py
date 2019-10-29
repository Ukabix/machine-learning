# NLP - Bag of words

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) # tsv specific, quiting skips ""

# cleaning the texts
import re
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
# ^we won't remove any a-z,A-Z, replace review with space infront
review = review.lower()
# ^making all letters lowercase
import 