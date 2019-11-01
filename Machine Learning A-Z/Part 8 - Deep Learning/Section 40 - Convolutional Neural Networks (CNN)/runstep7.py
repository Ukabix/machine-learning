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
# Steps: 1: Conv -> 2: MaxPool -> 3: Flat -> 4: FullConnection

# Step 1 - Convolution- Conv2D(filters, strides(rows, columns of feature detector), input shape = (h,w,channels), activation = 'af'):
# use relu so we do not get any negative pixels - this problem is nonlinear, ofc
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling- MaxPooling2D(pool_size = (lin, col))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flattening- Flatten()
classifier.add(Flatten())

# Step 4 - Full Connection - construct an ANN
# add hidden layer, activate by relu
classifier.add(Dense(units = 128, activation = 'relu'))
# add output layer, activate by sigmoid/softmax
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting CNN to images / Watch for overfitting
# Expanding Image Augumentation to make some transformation and widen the pic pool
# Import method - keras.io?
## validate the method:
# import ImageDataGenerator class
from keras.preprocessing.image import ImageDataGenerator
# call ImageDataGenerator class for training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# call ImageDataGenerator class for test set
test_datagen = ImageDataGenerator(rescale=1./255)
# apply ImageDataGenerator class on training set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

# apply ImageDataGenerator class on test set
test_set = validation_generator = test_datagen.flow_from_directory('dataset/test_set',
                                                                    target_size=(64, 64),
                                                                    batch_size=32,
                                                                    class_mode='binary')

# run the CNN
classifier.fit_generator(training_set,
                    steps_per_epoch=2000,
                    epochs=25,
                    validation_data = test_set,
                    validation_steps=2000)
