# %%
import skimage
import skimage.io
from skimage import io, color
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage import feature
import tensorflow as tf
keras = tf.keras
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
# from basic_image_eda import BasicImageEDA
# %%
img1 = skimage.io.imread('../data/tudor/download.1.jpg')
img2 = skimage.io.imread('../data/victorian/download.1.jpg')
img3 = skimage.io.imread('../data/modern/download.1.jpg')

# %%
img3.shape
# %%
# Compute the Canny filter for two values of sigma
# coin_gray = rgb2gray(coin)
# io.imshow(coin_gray);
def canny_filter(img):
  test_img = rgb2gray(img)
  edges1 = feature.canny(test_img, sigma=2)
  edges2 = feature.canny(test_img, sigma=5)

  # display results
  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                      sharex=True, sharey=True)

  ax1.imshow(test_img, cmap=plt.cm.gray)
  ax1.axis('off')
  ax1.set_title('VI image', fontsize=20)

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
# %%
see_edges(img3)