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
plt.rcParams['axes.grid'] = False
# from model import image_process
# %%
def predict_one(model, test_image_path, labels):
  img = keras.preprocessing.image.load_img(test_image_path,target_size=(150,150,3))
  img = keras.preprocessing.image.img_to_array(img)
  img = img.reshape((1,) + img.shape)
  pred = model.predict(img)
  print(pred)
  predicted_class_indices = np.argmax(pred, axis=1)
  prob = pred[0][predicted_class_indices]
  # print(predicted_class_indices)
  predictions = [labels[k] for k in predicted_class_indices]
  return predictions, prob


# %%
def predict_all(model, val_gen, labels):
  Y_preds = model.predict(val_gen, val_gen.n // val_gen.batch_size + 1 )
  # print(Y_preds)
  y_pred = np.argmax(Y_preds, axis=1)
  # print(y_pred)
  cm = confusion_matrix(val_gen.classes, y_pred)
  cr = classification_report(val_gen.classes, y_pred, target_names=list(labels.values()), output_dict=True)
  # top_k = keras.applications.xception.decode_predictions(Y_preds, top=5)

  return cm, cr, Y_preds


def display_img(prediction, test_image_path, labels, prob):
  plt.imshow(keras.preprocessing.image.load_img(test_image_path))
  plt.grid(None)
  print(prediction)
  plt.title(f'Predicted: {prediction[0]} \n {np.round(float(prob)*100, 2)}% Probability')
  plt.grid(False)



def plot_confusion_matrix(cm, labels):
  figure = plt.figure(figsize=(10, 10))
  ax = sns.heatmap(cm, annot=True,cmap=plt.cm.Blues, fmt='.0f')
  plt.title('Confusion Matrix')
  ax.set_xticklabels(list(labels.values()), rotation=45)
  ax.set_yticklabels(list(labels.values()), rotation=45)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  # plt.tight_layout()
  plt.show()


# %%
if __name__ == '__main__':
  n_epoch = 5
  img_height = 150
  img_width = 150
  num_classes = 5
  # data_dir = '../data'

  train_generator, validation_generator, test_generator = image_pipeline.main()

  labels = train_generator.class_indices
  labels = dict((v,k) for k,v in labels.items())

  transfer_filename = '../models/transfer_learn/train_model'
  cnn_filename = '../models/cnn_sequential/train_model'

  train_model = tf.keras.models.load_model(cnn_filename)
  # print(train_model.summary())

  test_image_path = '../data/test/modern/7.jpg'


  prediction, prob = predict_one(train_model, test_image_path , labels)
  display_img(prediction, test_image_path, labels, prob)

  cm, cr, all_prob = predict_all(train_model, validation_generator, labels)
  plot_confusion_matrix(cm, labels)
  table = pd.DataFrame(cr).transpose()
  display(table)

# %%
train_model.summary()