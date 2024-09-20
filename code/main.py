import numpy as np
import pandas as pd
import os
from data_functions import *
from model import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping



from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG19
from keras.applications import VGG16, EfficientNetB7, NASNetMobile, MobileNetV2
import tensorflow as tf 


# Define constants
IMAGE_SIZE = (112, 112)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 5


# # Load training and testing data
train_folder = '../data/train'
test_folder = '../data/test'

# Get data
X_train, y_train = load_train_data(train_folder,IMAGE_SIZE)
X_test, test_filenames = load_test_data(test_folder,IMAGE_SIZE)

# Split the data
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Normalize values
X_train_split = X_train_split / 255.0
X_test_split = X_test_split / 255.0

# Check the shapes of the split data
print('\n\n')
print('####### Shapes splitted data #######')
print("X_train_split.shape:", X_train_split.shape)
print("y_train_split.shape:", y_train_split.shape)
print("X_test_split.shape:", X_test_split.shape)
print("y_test_split.shape:", y_test_split.shape)
print('####################################')


m = Model()
model = m.get_model()
# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(X_train_split, y_train_split,batch_size=BATCH_SIZE, epochs=50, validation_data=(X_test_split, y_test_split))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_split, y_test_split)
print(f'Test accuracy: {accuracy}')

show_figures(history)