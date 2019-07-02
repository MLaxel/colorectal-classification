# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 22:13:32 2019

@author: tsuja
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

datagen = ImageDataGenerator(
    rescale=1./255)
    
train_gen = datagen.flow_from_directory('Downloads/colorectal/data/Train', class_mode='categorical', target_size=(256,256))
val_gen = datagen.flow_from_directory('Downloads/colorectal/data/Validate', class_mode='categorical', target_size=(256,256))
test_gen = datagen.flow_from_directory('Downloads/colorectal/data/Test', class_mode='categorical', target_size=(256,256))

model = tf.keras.models.Sequential([
#    tf.keras.layers.Con2D(16, (3,3), activation='relu', input_shape(256, 256, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

algo1 = model.fit_generator(
    train_gen,
    epochs = 15
)