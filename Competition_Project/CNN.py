import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

my_dir = os.getcwd()  # grabs current work dir
all_file_paths = os.listdir()  # grabs all items in current work dir

data_path = [data for data in all_file_paths if 'data' in data.lower()]  # searches for data folder and retrieves

picture_folder_path = os.path.join(my_dir, data_path[0])  # joins data folder and current work dir to get picture path
all_pictures = os.listdir(picture_folder_path)  # list of pictures names in folder

picture_complete_path = [os.path.join(picture_folder_path, pic) for pic in
                         all_pictures]  # complete path to each picture

img_1 = load_img(picture_complete_path[0])
print("FIRST IMAGE DETAILS:", "\nClass: ", type(img_1), "\nFile Format: ", img_1.format,
      "\nImage type: ", img_1.mode, '\nImage Size:', img_1.size)
print()

grayscale_message = '*' * 15 + ' Gray-scaling Images ' + '*' * 15
print(grayscale_message)
grey_scale_photos = [load_img(photo, color_mode='grayscale') for photo in picture_complete_path]
# convert all images to grayscale

print()

array_message = '*' * 15 + ' Converting Images to Arrays ' + '*' * 15
print(array_message)
array_photo_list = [img_to_array(photo) for photo in grey_scale_photos]
# convert all grayscale images to arrays for ML
X_train_all = np.asarray(array_photo_list)
X_train_all = X_train_all[:, :, :, 0]  # removes empty 4th dimension created from np.asarray()

# read in CSV as pandas DF
df_y = pd.read_csv('TrainAnnotations.csv')

# strip annotation column and convert to numpy
Y_train_all = df_y['annotation'].to_numpy()

# index position describes numeric values
y_labels = ['No Wilting', 'Leaflets folding inward at secondary pulvinus, no turgor loss in leaflets or petioles',
            'Slight leaflet or petiole turgor loss in upper canopy', 'Moderate turgor loss in upper canopy',
            'Severe turgor loss throughout canopy']
