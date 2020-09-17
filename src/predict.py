# %%
import numpy as np
import pandas as pd
import timeit
import matplotlib
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
# from tensorflow import loadLayersModel
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
keras = tf.keras
AUTOTUNE = tf.data.experimental.AUTOTUNE ## tf.data transformation parameters

plt.style.use('ggplot')

import transfer_learning_model
import image_pipeline
from model import image_process
# %%
def predict_one(model, test_image_path, labels, batch_size):
  img = keras.preprocessing.image.load_img(test_image_path,target_size=(299,299,3))
  img = keras.preprocessing.image.img_to_array(img)
  img = img.reshape((1,) + img.shape)
  pred = model.predict(img)
  predicted_class_indices = np.argmax(pred, axis=1)
  predictions = [labels[k] for k in predicted_class_indices
  return predictions


# %%
def predict_all(model, val_gen, val_samps, labels):
  Y_preds = model.predict(val_gen,
                                          val_samps // batch_size )
  y_pred = np.argmax(Y_preds, axis=1)
  cm = confusion_matrix(val_gen.classes, y_pred)
  return cm


def display_img(prediction, test_image_path, labels):
  plt.imshow(keras.preprocessing.image.load_img(test_image_path))
  plt.title(f'Predicted: {prediction[0]}')
  plt.grid(None)


def plot_confusion_matrix(cm, labels):
  figure = plt.figure(figsize=(8, 8))
  ax = sns.heatmap(cm, annot=True,cmap=plt.cm.Blues)
  plt.title('Confusion Matrix')
  ax.set_xticklabels(list(labels.values()), rotation=45)
  ax.set_yticklabels(list(labels.values())[::-1], rotation=45)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()
  plt.show()


# %%
if __name__ == '__main__':
  nb_epoch =5
  batch_size = 32
  nb_train_samples = 574
  nb_validation_samples = 139
  img_height = 299
  img_width = 299
  num_classes = 7
  data_dir = '../data'

  train_generator, validation_generator = image_pipeline.main()

  labels = train_generator.class_indices
  labels = dict((v,k) for k,v in labels.items())

  cnn_filename = '../models/transfer_learn/train_model'
  transfer_filename = '../models/cnn_sequential/train_model'

  train_model = tf.keras.models.load_model(cnn_filename)

  test_image_path = '../data/cape-cod/download.3.jpg'


  prediction = predict_one(train_model, test_image_path , labels, batch_size)
  display_img(prediction, test_image_path, labels)
  cm = predict_all(train_model, validation_generator, nb_validation_samples, labels)
  plot_confusion_matrix(cm, labels)
