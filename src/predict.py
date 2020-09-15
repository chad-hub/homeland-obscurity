# %%
import numpy as np
import pandas as pd
import timeit
import matplotlib
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from tensorflow import loadLayersModel
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
keras = tf.keras
AUTOTUNE = tf.data.experimental.AUTOTUNE ## tf.data transformation parameters

plt.style.use('ggplot')

import transfer_learning_model
import image_pipeline
# %%
def predict_one(model, test_image_path, labels,
                val_gen, val_samps, batch_sizez):
  img = keras.preprocessing.image.load_img(test_image_path,target_size=(299,299))
  img = keras.preprocessing.image.img_to_array(img)
  img = img.reshape((1,) + img.shape)
  pred = model.predict(img)
  predicted_class_indices = np.argmax(pred, axis=1)
  predictions = [labels[k] for k in predicted_class_indices]
  plt.imshow(keras.preprocessing.image.load_img(test_image_path))
  return predictions


# %%
def predict_all(model, val_gen, val_samps, labels):
  Y_preds = train_model.predict_generator(val_gen,
                                          val_samps // batch_size )
  y_pred = np.argmax(Y_preds, axis=1)
  cm = confusion_matrix(validation_generator.classes, y_pred)
  return cm

# %%
if __name__ == '__main__':
  train_generator, validation_generator = image_pipeline.main()
  labels = train_generator.class_indices
  labels = dict((v,k) for k,v in labels.items())

  filename = '../models/transfer_learn/train_model'
  train_model = tf.keras.models.load_model(filename)

  nb_epoch =10
  batch_size = 32
  nb_train_samples = 574
  nb_validation_samples = 139

  predictions = predict_one(train_model, '../data/tudor/download.5.jpg', labels, validation_generator, nb_validation_samples, batch_size)
  print(predictions)