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

def image_data_generator(data_dir, test_data_dir,
                       data_augment=False,
                       batch_size=32,
                       target_size=(150, 150),
                       color_mode='rgb',
                       class_mode='binary',
                       shuffle=True):
  if data_augment:
      datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=0.005,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.005,
                                   zoom_range=0.2,
                                   validation_split=0.2,
                                   horizontal_flip=True)

      test_datagen = ImageDataGenerator(rescale=1./255)
  else:
      datagen = ImageDataGenerator(rescale=1./255)

  train_generator = datagen.flow_from_directory(data_dir,
                                          target_size=target_size,
                                          color_mode='rgb',
                                          batch_size=batch_size,
                                          shuffle=False,
                                          class_mode='categorical',
                                          subset='training')
  validation_generator = datagen.flow_from_directory(data_dir,
                                          target_size=target_size,
                                          color_mode='rgb',
                                          batch_size=batch_size,
                                          shuffle=False,
                                          class_mode='categorical',
                                          subset='validation')
  test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                    target_size=target_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

  return train_generator, validation_generator, test_generator

# %%
def main():
  train_generator, validation_generator, test_generator = image_data_generator('../data/train/', '../data/test/', batch_size=32,data_augment=True)
  return train_generator, validation_generator, test_generator

# %%
if __name__ == '__main__':
  train_generator, validation_generator, test_generator = main()
