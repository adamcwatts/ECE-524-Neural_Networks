import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator, array_to_img, save_img
import numpy as np
import pandas as pd
import os
import seaborn as sns
from functools import partial
sns.set_context('notebook')
sns.set_style('white')
import matplotlib.pyplot as plt

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# # session = tf.compat.v1.Session(config=config)
# session = tf.compat.v1.InteractiveSession(config=config)


# config.log_device_placement = True  # to log device placement (on which device the operation ran)
#                                     # (nothing gets printed in Jupyter, only if you run it standalone)
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# config.gpu_options.allow_growth = True
my_dir = os.getcwd()  # grabs current work dir
all_file_paths = os.listdir()  # grabs all items in current work dir

data_path = [data for data in all_file_paths if 'traindata' in data.lower()]  # searches for data folder and retrieves

picture_folder_path = os.path.join(my_dir, data_path[0])  # joins data folder and current work dir to get picture path
all_pictures = os.listdir(picture_folder_path)  # list of pictures names in folder

picture_complete_path = [os.path.join(picture_folder_path, pic) for pic in
                         all_pictures]  # complete path to each picture

img_1 = load_img(picture_complete_path[0])
print("FIRST IMAGE DETAILS:", "\nClass: ", type(img_1), "\nFile Format: ", img_1.format,
      "\nImage type: ", img_1.mode, '\nImage Size:', img_1.size)
print()

# grayscale_message = '*' * 15 + ' Gray-scaling Images ' + '*' * 15
Color_message = '*' * 15 + ' IMPORTING RGB Images ' + '*' * 15

print(Color_message)
# grey_scale_photos = [load_img(photo, color_mode='grayscale') for photo in picture_complete_path]
rgb_scale_photos = [load_img(photo, color_mode='rgb', target_size=(240, 240)) for photo in picture_complete_path] # RGB

# convert all images to grayscale

# print()

array_message = '*' * 15 + ' Converting Images to Arrays ' + '*' * 15
print(array_message)

# convert all grayscale images to arrays for ML
# array_photo_list = [img_to_array(photo) for photo in grey_scale_photos]
array_photo_list = [img_to_array(photo) for photo in rgb_scale_photos] # RGB


X_train_all = np.asarray(array_photo_list)
X_train_all[:, :,:,0] /= 255.0  # normalizes images

# del rgb_scale_photos, array_photo_list
# read in CSV as pandas DF
df_y = pd.read_csv('TrainAnnotations.csv')

# strip annotation column and convert to numpy
Y_train_all = df_y['annotation'].to_numpy()

# index position describes numeric values
y_labels = ['No Wilting', 'Leaflets folding inward at secondary pulvinus, no turgor loss in leaflets or petioles',
            'Slight leaflet or petiole turgor loss in upper canopy', 'Moderate turgor loss in upper canopy',
            'Severe turgor loss throughout canopy']

df_y.describe()
sns.distplot(Y_train_all, kde=False, bins=5)
plt.show()


model = keras.models.Sequential([keras.layers.Conv2D(filters=64, kernel_size=11, padding='SAME', input_shape=X_train_all.shape[1:]),
                                 keras.layers.MaxPooling2D(pool_size=2),
                                 keras.layers.Conv2D(filters=128, padding='SAME', kernel_size=5),
#                                  keras.layers.Conv2D(filters=128, kernel_size=3),
                                 keras.layers.MaxPooling2D(pool_size=2),
                                 keras.layers.Conv2D(filters=256, padding='SAME', kernel_size=3),
                                 keras.layers.MaxPooling2D(pool_size=2),
#                                  keras.layers.Conv2D(filters=256, kernel_size=3),
                                 keras.layers.Flatten(),
                                 keras.layers.Dense(units=16, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
                                 keras.layers.Dropout(0.5),
                                 keras.layers.Dense(units=8, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
                                 keras.layers.Dropout(0.5),
                                 keras.layers.Dense(units=5, activation='softmax')
                                 ])

model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_all, Y_train_all, epochs=10, validation_split=0.15, batch_size=32)
pd.DataFrame(history.history).plot(figsize=(12, 6))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()