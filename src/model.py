# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timeit
import matplotlib

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
keras = tf.keras
AUTOTUNE = tf.data.experimental.AUTOTUNE ## tf.data transformation parameters

matplotlib.style.use('ggplot')

import image_pipeline
import pickle
# %%
def image_process(data_dir, batch_size, img_height, img_width, num_classes):
  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                          data_dir,
                          color_mode = 'rgb',
                          validation_split=0.2,
                          subset="training",
                          seed=42,
                          image_size=(img_height, img_width),
                          batch_size=batch_size)
  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                          data_dir,
                          color_mode = 'rgb',
                          validation_split=0.2,
                          subset="validation",
                          seed=42,
                          image_size=(img_height, img_width),
                          batch_size=batch_size)

  class_names = train_ds.class_names

  plt.figure(figsize=(10, 10))
  for images, labels in train_ds.take(1):
    for i in range(num_classes):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[labels[i]])
      plt.axis("off")
    plt.tight_layout()
    plt.show()


  return train_ds, val_ds, class_names

# %%
def create_model(num_classes, data_augmentation, img_height, img_width):

  plt.figure(figsize=(10, 10))
  for images, _ in train_ds.take(1):
    for i in range(9):
      augmented_images = data_augmentation(images)
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(augmented_images[0].numpy().astype("uint8"))
      plt.axis("off")
    plt.tight_layout()
    plt.show()


  model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    layers.Conv2D(32, 5, strides=2, padding='same', activation='relu'),
    layers.MaxPooling2D(strides=2),
    layers.Dropout(0.1),

    layers.Conv2D(32, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(strides=2),
    layers.Dropout(0.1),

    layers.Conv2D(32, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(strides=2),
    layers.Dropout(0.1),

    layers.Conv2D(32, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(strides=2),
    layers.Dropout(0.1),

    layers.Flatten(),
    layers.Dense(num_classes, activation='softmax'),
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  print(model.summary())

  return model

def train_model(model, n_epochs, train_ds, val_ds):
  history = model.fit(train_ds,
                      validation_data = val_ds,
                      epochs=n_epochs)

  acc = history.history['accuracy']

  ##plot training results
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

  return model

# %%
def plot_home_weights(class_names, train_data):
  """Plot the weights from our fit fully connected network as an image."""
  train_data = tf.image.rgb_to_grayscale(train_data)
  model = keras.Sequential()
  ## unstacking rows of pixels in the image and lining them up
  model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)))
  model.add(keras.layers.Flatten())
  ## The second (and last) layer is a 10-node softmax layer that
  ##    returns an array of 10 probability scores that sum to 1
  model.add(keras.layers.Dense(7, activation='softmax'))
  model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  # print(model.summary())
  model.fit(train_data, epochs=10)

  plt.figure(figsize=(10, 10))
  for idx, c in enumerate(class_names):
    ax = plt.subplot(3, 3, idx + 1)
    house_weigths = np.reshape(model.weights[0][:,idx], (299,299))
    plt.imshow(house_weigths, cmap=plt.cm.winter, interpolation="nearest")
    plt.title(f'{c} weights')
    ax.grid(False)
  plt.tight_layout()
  plt.show()



# %%
if __name__ == '__main__':
  batch_size = 32
  img_height = 299
  img_width = 299
  num_classes = 7
  n_epochs = 10
  data_dir = '../data'

  data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(img_height,
                                                              img_width,
                                                              3)),
    # layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.2)
  ])

  train_ds, val_ds, class_names = image_process(data_dir, batch_size, img_height, img_width, num_classes)

  model = create_model(num_classes, data_augmentation, img_height, img_width)

  model = train_model(model, n_epochs, train_ds, val_ds)


# %%
# %%
plot_home_weights(class_names, train_ds)
# %%
class_names
