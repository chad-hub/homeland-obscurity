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
def add_new_last_layer(base_model, n_classes=5):
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
    predictions = keras.layers.Dense(n_classes, activation='softmax')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=predictions)
    return model

def setup_to_finetune(model, layer_level):
    """Freeze the bottom layer_level layers and train the rest

    Args:
     model: keras model
    """
    for layer in model.layers[:layer_level]:
        layer.trainable = False
    for layer in model.layers[layer_level:]:
        layer.trainable = True


def setup_to_transfer_learn(model, base_model, lr = 1e-5):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer=keras.optimizers.Adam(lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  return model


def fit_model(model, train_gen, n_epochs, val_gen, init_epoch):

  model = model.fit(
        train_gen,
        epochs=n_epochs,
        steps_per_epoch=train_gen.n // train_gen.batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.n // val_gen.batch_size,
        initial_epoch=init_epoch)

  return model

def plot_training_results(history, n_epochs, fine_tune, *args):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss=history.history['loss']
  val_loss=history.history['val_loss']

  epochs_range = range(n_epochs)

  initial_epochs = n_epochs

  if fine_tune:
    acc+= args[0].history['accuracy']
    val_acc+= args[0].history['val_accuracy']

    loss += args[0].history['loss']
    val_loss += args[0].history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0, 1.5])
    plt.plot([initial_epochs-1,initial_epochs-1],
                plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 2.0])
    plt.plot([initial_epochs-1,initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

  else:
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
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
                                        input_shape=(150,150,3)
                                        )
  train_model = add_new_last_layer(base_model)

  train_model = setup_to_transfer_learn(train_model,base_model,lr = 1e-3)

  n_epoch = 20
  fine_tune = False


  history = fit_model(train_model, train_generator, n_epoch, validation_generator, 0)
  plot_training_results(history, n_epoch, fine_tune)

  fine_tune = True

  if fine_tune:
    fine_tune_at = 123
    fine_tune_epochs = 20
    total_epochs = n_epoch + fine_tune_epochs
    setup_to_finetune(train_model,fine_tune_at)


    tune_history = fit_model(train_model, train_generator, total_epochs, validation_generator,
                        history.epoch[-1])
    plot_training_results(history, n_epoch, fine_tune, tune_history)

  filename = '../models/transfer_learn/train_model'
  tf.saved_model.save(train_model, filename)


  labels = train_generator.class_indices
  labels = dict((v,k) for k,v in labels.items())

# %%
if __name__ == '__main__':
  main()

# %%
