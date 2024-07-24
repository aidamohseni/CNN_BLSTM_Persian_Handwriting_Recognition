# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 23:46:26 2018

@author: aida mohseni

"""
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, TimeDistributed
from keras import backend as K
from keras.layers import Bidirectional, LSTM

# dimensions of our images.
img_width, img_height = 28, 28
pool_size = 2
# train_data_dir = 'D:/uhat/characters_train'
# validation_data_dir = 'D:/uhat/characters_test'

train_data_dir = 'd:/hudc2/train'
validation_data_dir = 'd:/hudc2/test'
# train_data_dir = 'd:/data2/train'
# validation_data_dir = 'd:/data2/test'
nb_train_samples = 500#560
nb_validation_samples = 100#140
epochs = 1
batch_size = 1

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(16, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))

############################################
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('relu'))
###########################################

model.add(Flatten())
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))
###########################################


model.compile(loss='binary_crossentropy',
              #optimizer='rmsprop',
              optimizer='adam',
              metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

# , keras.metrics.Precision(), keras.metrics.Recall()
model.summary()
#===================================
import visualkeras
from tensorflow.keras import layers
from collections import defaultdict
color_map = defaultdict(dict)
color_map[layers.Conv2D]['fill'] = '#00f5d4'
color_map[layers.MaxPooling2D]['fill'] = '#8338ec'
color_map[layers.Dropout]['fill'] = '#03045e'
color_map[layers.Dense]['fill'] = '#fb5607'
color_map[layers.Flatten]['fill'] = '#ffbe0b'
visualkeras.layered_view(model, legend=True, color_map=color_map).show()

#===================================
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
# class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')

# precision, recall, thresholds = precision_recall_curve(y_train, y_prob_train)
# plt.fill_between(recall, precision)
# plt.ylabel("Precision")
# plt.xlabel("Recall")
# plt.title("Train Precision-Recall curve");
# plt.show()