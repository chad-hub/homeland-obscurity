
# %%
from os.path import join
import os


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timeit

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
keras = tf.keras
AUTOTUNE = tf.data.experimental.AUTOTUNE ## tf.data transformation parameters

matplotlib.style.use('ggplot')

import numpy as np
import pandas as pd
import glob
from PIL import Image
# %%
def image_data_generator(data_dir,
                       data_augment=False,
                       batch_size=32,
                       target_size=(299, 299),
                       color_mode='rgb',
                       class_mode='binary',
                       shuffle=True):
  if data_augment:
      datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   validation_split=0.2,
                                   horizontal_flip=True)
  else:
      datagen = ImageDataGenerator(rescale=1./255)

  train_generator = datagen.flow_from_directory(data_dir,
                                          target_size=(299,299,3),
                                          color_mode='rgb',
                                          batch_size=32,
                                          shuffle=0.2,
                                          class_mode='categorical',
                                          subset='training')
  validation_generator = datagen.flow_from_directory(data_dir,
                                          target_size=(299,299,3),
                                          color_mode='rgb',
                                          batch_size=32,
                                          shuffle=0.2,
                                          class_mode='categorical',
                                          subset='validation')

  return train_generator, validation_generator

def main():
  train_generator, validation_generator = image_data_generator('../data',data_augment=True)
  return train_generator, validation_generator

# %%
if __name__ == '__main__':
  train_generator, validation_generator = main()

# %%
