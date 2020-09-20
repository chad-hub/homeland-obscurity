# %%
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
# %%
class ResidualUnit(keras.layers.Layer):
  def __init__(self, filters, strides=1, activation='relu', **kwargs):
    super().__init__(**kwargs)
    self.activation = keras.activations.get(activation)
    self.main_layers = [
          keras.layers.Conv2D(filters, 3, strides=strides,
                      padding='valid', use_bias=False),
          keras.layers.BatchNormalization(),
          self.activation,
          keras.layers.Conv2D(filters, 3, strides=strides,
                      padding='valid', use_bias=False),
          keras.layers.BatchNormalization()],
    self.skip_layers = []
    if strides > 1:
          self.skip_layers = [
            keras.layers.Conv2D(filters, 1, strides=strides,
                                padding='valid', use_bias=False),
            keras.layers.BatchNormalization()]
  def call(self, inputs):
    Z = inputs
    for layer in self.main_layers:
      Z = layer(Z)
    skip_Z = inputs
    for layer in self.skip_layers:
      skip_Z = layer(skip_Z)
    return self.activation(Z + skip_Z)


# %%
model = keras.Sequential()
models.add(layers.Conv2D(64, 7,padding='valid', activation='relu', name='target_layer')