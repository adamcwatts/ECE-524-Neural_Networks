from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import cv2
import numpy as np
import pandas as pd
import os
import seaborn as sns
from functools import partial

sns.set_context('notebook')
import matplotlib.pyplot as plt


def hist_equal(img, L):
    '''takes image and an output space [0,L] as an input and gives an equalized image(in float) as output'''
    img = img.astype('float')
    epsilon = 1e-6  # a small number to remove divide by zero error
    if len(img.shape) == 2:
        return (L * (img - min(img.flatten())) / (max(img.flatten()) - min(img.flatten()) + epsilon))
    else:
        img[:, :, 0] = L * (img[:, :, 0] - min(img[:, :, 0].flatten())) / (
                    max(img[:, :, 0].flatten()) - min(img[:, :, 0].flatten()) + epsilon)
        img[:, :, 1] = L * (img[:, :, 1] - min(img[:, :, 1].flatten())) / (
                    max(img[:, :, 1].flatten()) - min(img[:, :, 1].flatten()) + epsilon)
        img[:, :, 2] = L * (img[:, :, 2] - min(img[:, :, 2].flatten())) / (
                    max(img[:, :, 2].flatten()) - min(img[:, :, 2].flatten()) + epsilon)
        return img
