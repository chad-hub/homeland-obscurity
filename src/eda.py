# %%
# Image processing
import skimage
from skimage import io, color
from skimage.filters import roberts, sobel, scharr, prewitt, gaussian
from skimage import feature
from skimage.color import rgb2gray
import matplotlib.image as mpimg
import scipy.ndimage as ndimage

#Plotting / arrays
import matplotlib.pyplot as plt
import numpy as np
from numpy import expand_dims
import os, random, cv2

# Keras / Tensorflow
from keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
keras = tf.keras
from keras.preprocessing.image import img_to_array
from tensorflow.keras import activations
from tensorflow import utils

#sytle
matplotlib.style.use('ggplot')
plt.rcParams['axes.grid'] = False

# %%
# Compute the Canny filter for two values of sigma
def canny_filter(img):
  test_img = img.img_to_array()
  edges1 = feature.canny(test_img, sigma=3)
  edges2 = feature.canny(test_img, sigma=5)

  # display results
  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                      sharex=True, sharey=True)

  ax1.imshow(test_img, cmap=plt.cm.gray)
  ax1.axis('off')
  ax1.set_title('Image', fontsize=20)

  ax2.imshow(edges1, cmap=plt.cm.gray)
  ax2.axis('off')
  ax2.set_title('Canny filter, $\sigma=2$', fontsize=20)

  ax3.imshow(edges2, cmap=plt.cm.gray)
  ax3.axis('off')
  ax3.set_title('Canny filter, $\sigma=5$', fontsize=20)

  fig.tight_layout()
  np.sum(edges1)

# %%
def see_edges(img):
  test_img = rgb2gray(img)

  edge_roberts = roberts(test_img)
  edge_sobel = sobel(test_img)
  edge_prewitt = prewitt(test_img)

  fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True,
                        figsize=(8, 4))

  ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
  ax[0].set_title('Roberts Edge Detection')

  ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
  ax[1].set_title('Sobel Edge Detection')

  ax[2].imshow(edge_prewitt, cmap=plt.cm.gray)
  ax[2].set_title('Prewitt Filter')

  for a in ax:
      a.axis('off')
  plt.tight_layout()


## function to display activations
def display_activation(activations, col_size, row_size, act_index):
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    plt.suptitle(f'Output of layer {act_index+1} - Baseline', fontsize=24)
    # plt.grid(None)
    for row in range(0,row_size):
        for col in range(0,col_size):
            plt.grid(None)
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1



# %%
def image_sizes():
    x = []
    y = []
    for dirpath, dirnames, filenames in os.walk("../data/train/"):
        for filename in [f for f in filenames if f.endswith('.jpg')]: # to loop over all images you have on the directory
            img_shape = cv2.imread(dirpath + '/' + filename).shape
            x.append(img_shape[0])
            y.append(img_shape[1])
    plt.scatter(x = x, y=y)
    plt.title('Size of Images Scraped')
    plt.show()

# %%
def show_augmented_img(dir_path):
    img_path = random.choice(os.listdir(dir_path))
    img = keras.preprocessing.image.load_img(dir_path+img_path,target_size=(224,224, 3))
    data = img_to_array(img)
    samps = expand_dims(data, 0)
    plt.figure(figsize=(8, 8))
    for i in range(9):
        augmented_images = data_augmentation(samps)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype('uint8'))
        plt.axis("off")
    plt.suptitle('Data Augmentation', fontsize=18)
    # plt.tight_layout()
    plt.show()

# %%
if __name__ == '__main__':
    image_sizes()

    data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(124,
                                                              124,
                                                              3)),
        layers.experimental.preprocessing.RandomRotation(0.005),
        layers.experimental.preprocessing.RandomZoom(0.01),
        layers.experimental.preprocessing.RandomHeight(0.20),
        layers.experimental.preprocessing.RandomWidth(0.20),
    ])

    paths = ['../data/train/tudor/', '../data/train/modern/',
                '../data/train/ranch/', '../data/train/victorian/','../data/train/cape-cod/' ]

    show_augmented_img(np.random.choice(paths))

    img1 = keras.preprocessing.image.load_img('../data/train/cape-cod/2.jpg',target_size=(150,150))
    img2 = keras.preprocessing.image.load_img('../data/train/ranch15.jpg',target_size=(150,150))
    img3 = keras.preprocessing.image.load_img('../data/train/modern/7.jpg',target_size=(150,150))

    see_edges(img1)

    canny_filter(img1)

    transfer_filename = '../models/transfer_learn/train_model'
    cnn_filename = '../models/cnn_sequential/train_model'

    ##input model type
    model_test = tf.keras.models.load_model(cnn_filename)
    layer_outputs = [layer.output for layer in model_test.layers]
    activation_model = Model(inputs=model_test.input, outputs=layer_outputs)
    activations = activation_model.predict(tf.keras.preprocessing.image.img_to_array(img3).reshape(1,150,150,3))
    display_activation(activations, 5, 5, 4)
