# Convolutional Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Part 1 - Building the CNN


# Importing the libraries
# import initialising function
from keras.models import Sequential
# import convolution layer tool
from keras.layers import Conv2D
# import pooling tool
from keras.layers import MaxPooling2D
# import flattening tool
from keras.layers import Flatten
# import layer integrator
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()


# CNN algorithm:
# Steps: 1: Conv -> 2: MaxPool -> 3: Flat -> FullConnection

# Step 1 - Convolution- Conv2D(filters, strides(rows, columns of feature detector), input shape = (h,w,channels), activation = 'af'):
# use relu so we do not get any negative pixels - this problem is nonlinear, ofc
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
