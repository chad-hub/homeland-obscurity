# %%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timeit
import matplotlib

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import activations
from skimage.color import rgb2gray

import tensorflow as tf
import datetime
from tensorflow.keras import layers

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint,LearningRateScheduler, EarlyStopping
from tensorflow.keras import backend as K
from keras.models import Model
keras = tf.keras
AUTOTUNE = tf.data.experimental.AUTOTUNE ## tf.data transformation parameters

matplotlib.style.use('ggplot')

import image_pipeline
import tf_explain
from res_34 import ResidualUnit

# %%
def create_model(num_classes, img_height, img_width, train_ds, lr=1e-3):

  model = Sequential()
	    # layers.experimental.preprocessing.Rescaling(1./255,
                                    # input_shape=(img_height, img_width, 3)),

  model.add(layers.Conv2D(64, 7, strides=2, padding='valid',
                       activation='relu', input_shape=[img_height, img_width, 3]))
  model.add(layers.BatchNormalization())
  model.add(layers.Activation('relu'))
  model.add(layers.MaxPooling2D(pool_size=3, strides=2, padding='valid'))
  prev_filters = 64
  for filters in [64]*3 + [128]*4 + [256]*6 + [512]*3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
  model.add(layers.GlobalAvgPool2D())
  model.add(layers.flatten())
  model.add(layers,Dense(num_classes, activation='softmax'))
  model.compile(optimizer=keras.optimizers.Adam(lr),
	                loss='categorical_crossentropy',
	                metrics=['accuracy'])

  print(model.summary())


  return model

def train_model(model, n_epochs, train_gen, val_gen):
  model = model.fit(train_gen,
                      validation_data = val_gen,
                      validation_steps=val_gen.n // val_gen.batch_size,
                      epochs=n_epochs,
                      steps_per_epoch=train_gen.n//train_gen.batch_size)

  return model


# %%

def plot_training_results(history, n_epochs):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss=history.history['loss']
  val_loss=history.history['val_loss']

  epochs_range = range(n_epochs)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()



# %%
if __name__ == '__main__':
  batch_size = 32
  img_height = 150
  img_width = 150
  num_classes = 5
  n_epochs = 15
  data_dir = '../data'
  lr = 1e-3 #learning rate for optimizer

  train_generator, validation_generator = image_pipeline.main()


  model = create_model(num_classes, img_height,
                          img_width, train_generator)


  history = train_model(model, n_epochs, train_generator, validation_generator, ) #callbacks if it can work

  # %tensorboard --logdir logs/fit

  filename = '../models/cnn_sequential/train_model'
  tf.saved_model.save(model, filename)

  plot_training_results(history, n_epochs)

  # %tensorboard --logdir logs/fit