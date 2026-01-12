
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from tensorflow import keras
from tensorflow.keras import layers , models
import matplotlib.pyplot as plt

(x_train , y_train) , (x_test , y_test)= tf.keras.datasets.mnist.load_data()

model = models.Sequential([
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5)

model.evaluate(x_test,y_test)

prediction =model.predict(x_test)

prediction.shape
