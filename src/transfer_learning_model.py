# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timeit
import matplotlib

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
keras = tf.keras
AUTOTUNE = tf.data.experimental.AUTOTUNE ## tf.data transformation parameters

matplotlib.style.use('ggplot')

import image_pipeline
import pickle

# %%
def add_new_last_layer(base_model, nb_classes=7):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    # Get the output shape of the models last layer
    x = base_model.output
    # Convert final MxNxC tensor output into a 1xC tensor where C is the # of channels.
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dropout(.2)(x)
    predictions = keras.layers.Dense(7, activation='softmax')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=predictions)
    return model

def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  return model

def fit_model(model, train_gen, train_samps, n_epochs,
              val_gen, val_samps, batch_size):

  model = model.fit_generator(
        train_gen,
        epochs=n_epochs,
        steps_per_epoch=train_samps // batch_size,
        validation_data=val_gen,
        validation_steps=val_samps // batch_size)

  return model

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
def main():
  train_generator, validation_generator = image_pipeline.main()
  base_model = keras.applications.Xception(include_top=False,
                                        weights='imagenet',
                                        input_shape=(299,299,3)
                                        )
  train_model = add_new_last_layer(base_model)

  train_model = setup_to_transfer_learn(train_model,base_model)

  n_epoch =15
  batch_size = 32
  n_train_samples = 574
  n_validation_samples = 139

  history = fit_model(train_model, train_generator, n_train_samples,
                        n_epoch, validation_generator, n_validation_samples,
                        batch_size)

  filename = '../models/transfer_learn/train_model'
  tf.saved_model.save(train_model, filename)

  plot_training_results(history, n_epoch)
  labels = train_generator.class_indices
  labels = dict((v,k) for k,v in labels.items())

# %%
if __name__ == '__main__':
  main()

# %%
base_model = keras.applications.Xception(include_top=False,
                                        weights='imagenet',
                                        input_shape=(299,299,3)
                                        )
base_model.summary()