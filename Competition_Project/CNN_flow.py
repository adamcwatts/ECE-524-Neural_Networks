import numpy as np
import pandas as pd
import os
import seaborn as sns
from functools import partial

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = keras.models.Sequential()
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=2, activation='relu', padding="SAME")

model = keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=11, input_shape=(640, 480, 3)),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=4),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=4),
    keras.layers.Flatten(),
    keras.layers.Dense(units=20, activation='relu'),
    keras.layers.Dropout(0.50),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(units=10, activation='relu'),
    keras.layers.Dropout(0.50),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(units=5, activation='softmax'),
])

model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

my_dir = os.getcwd()  # grabs current work dir
all_file_paths = os.listdir()  # grabs all items in current work dir

FOLDER_TO_ACCESS = 'traindata_pp_structure'
data_path = [data for data in all_file_paths if
             FOLDER_TO_ACCESS in data.lower()]  # searches for data folder and retrieves
train_folder_path = os.path.join(my_dir, data_path[0])  # joins data folder and current work dir to get picture path
all_classifications = os.listdir(train_folder_path)  # list of classification fodlers

train_datagen = ImageDataGenerator(
    validation_split=0.2)  # set validation split

train_it = train_datagen.flow_from_directory(
    train_folder_path,
    target_size=(640, 480),
    class_mode='sparse',
    batch_size=16)

history = model.fit(train_it,
                    steps_per_epoch=train_it.n // train_it.batch_size,
                    epochs=1, verbose=1,
                    callbacks=None, validation_data=None,
                    validation_steps=None,
                    class_weight=None, max_queue_size=2,
                    workers=1, use_multiprocessing=False,
                    shuffle=False, initial_epoch=0)
