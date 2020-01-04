#! /usr/bin/python


# Training a CNN from scratch on a small dataset. in this case, we plan to use a dataset containing 4000
# pictures of cats and dogs (2000 cats, 2000 dogs). Also will use 2000 pictures for training, 1000 for validation,
# and finally 1000 for testing.


#Initialise NN sequence of layers
from keras.models import Sequential

#Convolution on images for feature (edge) detection
from keras.layers import Conv2D

#Downsample images keeping high intensity points
from keras.layers import MaxPooling2D

#Convert 2D arrays to flat linear vector
from keras.layers import Flatten

#Perform connection of CNN, provide node-weight architecture
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
#Apply 2D conv using 32, 3x3 kernels per image with dimensions 64x64x3 (RGB)
#Use rectified linear unit activation
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
#Use node-vector to act as input to fully-connected layer
#Expect 128 nodes in hidden layer, range b/w input and output layer sizes
model.add(Dense(units=128, activation='relu'))
#Binary classification, single node (dog/cat)
model.add(Dense(units=1, activation='sigmoid'))
#Optimiser: SGD algorithm, loss: loss func, metrics: performance metric
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Use keras preprocessing to prevent overfitting (great training accuracy, low test)
#Apply regularisation rate (lambda) using either L1 (Least Abs Deviations) or
#L2 (Least Squares Error), L1 reduces weight values of less important features
#Done to fit data with ideal variance and bias so noise/outliers have less effect

#Randomise training data parameters (affects zoom, size, shear)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

# NOTE: The cats and dogs dataset needs to be downloaded first.
# It was made available by Kaggle.com as part of a computer vision competition in late 2013.
# You can download the original dataset at: https://www.kaggle.com/c/dogs-vs-cats/data
# (you will need to create a Kaggle account if you don't already have one).
#
#The pictures are medium-resolution color JPEGs. 
#
# Assume that trainining set is in a folder called 'cats_and_dogs/train' in your home directory

training_set=train_datagen.flow_from_directory('~/cats_and_dogs/train',
                                               target_size=(64,64),
                                               batch_size=64,
                                               class_mode='binary')

# Assume that test set is in a folder called 'cats_and_dogs/test' in your home directory
test_set=train_datagen.flow_from_directory('~/cats_and_dogs/test',
                                           target_size=(64,64),
                                           batch_size=64,
                                           class_mode='binary')


# Assume that validation set is in a folder called 'cats_and_dogs/validation' in your home directory
validation_data_generator = ImageDataGenerator(rescale=1./255)
validation_generator = validation_data_generator.flow_from_directory('~/cats_and_dogs_small/validation',
                                           target_size=(64,64),
                                           batch_size=64,
                                           class_mode='binary')

#Fit data to model, steps/epoch is training images
#Single epoch is single step in training samples of pass
model.fit_generator(training_set,
                         steps_per_epoch=500,
                         epochs=10,
                         verbose=1,
                         validation_data=validation_generator,
                         validation_steps=300)


#Predictions from trained model
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('test_image.jpg', target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=model.predict(test_image)
training_set.class_indices

# print what the test_image represents.
if result[0][0] >= 0.5:
    prediction='dog'
else:
    prediction='cat'
print(prediction)
