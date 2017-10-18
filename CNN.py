# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:09:46 2017

@author: nihal369
"""

#CNN Architecture

#Importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

#Initalziing the CNN
classifier=Sequential()

#Step-1 Convolution
classifier.add(Convolution2D(32,(3,3),activation='relu',input_shape=(64,64,3)))

#Step-2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step-3 Flatenning
classifier.add(Flatten())
