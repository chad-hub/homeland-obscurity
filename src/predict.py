# %%
import numpy as np
import pandas as pd
import timeit
import matplotlib
import matplotlib.pyplot as plt
import pickle

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
  Y_preds = model.predict(val_gen,
                                          val_samps // batch_size )
  y_pred = np.argmax(Y_preds, axis=1)
  # cm = confusion_matrix(val_gen.classes, y_pred)
  return y_pred

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

  # train_generator, validation_generator = image_pipeline.main()

  train_ds, val_ds, class_names = image_process(data_dir, batch_size, img_height, img_width, num_classes)

  # labels = train_generator.class_indices
  labels = class_names

  # labels = dict((v,k) for k,v in labels)

  filename = '../models/cnn_sequential/train_model'
  train_model = tf.keras.models.load_model(filename)



  prediction = predict_one(train_model, '../data/southwestern/download.3.jpg', labels, batch_size)
  print(prediction)

  pred_all = predict_all(train_model, val_ds, nb_validation_samples, labels)
  print(pred_all)
# %%
val_labels = []
for _, label in val_ds.take(1):
  for i in range(139):
    val_labels.append(class_names[label[i]])

val_labels[10:]
# %%
