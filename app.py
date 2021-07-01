# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 00:40:50 2020

@author: Gokul
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from keras.preprocessing import image
from keras.models import Sequential
from tensorflow.keras import regularizers
from keras.layers import Conv2D, AvgPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout

def import_and_predict(image_data, model):
        size = (64,64)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        x = np.expand_dims(image, axis=0)
        x = np.vstack([x])
        prediction = model.predict(x)
        return prediction

model = Sequential()
model.add(InputLayer(input_shape=(64, 64, 3)))
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(Conv2D(32, kernel_size=3,activation='relu',padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15), bias_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15)))
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(Conv2D(32, kernel_size=3,activation='relu',padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15), bias_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15)))
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(AvgPool2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(Conv2D(64, kernel_size=3,activation='relu',padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15), bias_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15)))
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(Conv2D(64, kernel_size=3,activation='relu',padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15), bias_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15)))
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(AvgPool2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(Conv2D(128, kernel_size=3,activation='relu',padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15), bias_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15)))
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(Conv2D(128, kernel_size=3,activation='relu',padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15), bias_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15)))
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(AvgPool2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(Conv2D(256, kernel_size=3,activation='relu',padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15), bias_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15)))
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(Conv2D(256, kernel_size=3,activation='relu',padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15), bias_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15)))
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(AvgPool2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(Flatten())
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(Dropout(0.6))
model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15), bias_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15)))
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(Dropout(0.6))
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15), bias_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15)))
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(Dropout(0.6))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer="rmsprop",loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights("955.h5")

st.write("""
          # COVID-19 Detector using Chest X-Ray
          """
          )

st.write("This is a image classification web app to predict COVID-19")

file = st.file_uploader("Please upload an image file of Chest X-Ray", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    prediction[0][0]=prediction[0][0]*100
    prediction[0][1]=prediction[0][1]*100
    st.text("Percentage (0: Negative, 1: Postive)")
    st.write(prediction)
    
    if np.argmax(prediction) == 0:
        st.write("COVID Negative!")
    else:
        st.write("COVID Positive!")
        st.write("Please Contact the Authorities")
        st.write("https://www.cowin.gov.in/home")

    
