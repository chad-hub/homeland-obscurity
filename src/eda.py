# %%
import skimage
import skimage.io
from skimage import io, color
from skimage.filters import roberts, sobel, scharr, prewitt, gaussian
from skimage import feature
import tensorflow as tf
keras = tf.keras
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
# from basic_image_eda import BasicImageEDA
# %%
img1 = skimage.io.imread('../data/tudor/download.2.jpg')
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
  edges1 = feature.canny(test_img, sigma=1.5)
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
see_edges(img1)

# %%
canny_filter(img1)
# %%
%matplotlib inline
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.dpi'] = 100

# %%
sobel_img = sobel(rgb2gray(img1))
plt.imshow(sobel_img, cmap=plt.cm.gray)
plt.grid(None)

# %%
blurred = gaussian(sobel_img, sigma=0.5)
plt.imshow(blurred)
plt.grid(None)
# %%
import numpy as np

# %%
light_spots = np.array((img1 > 245).nonzero()).T
# %%
light_spots.shape
# %%

plt.plot(light_spots[:, 1], light_spots[:, 0], 'o')
plt.imshow(img1)
plt.title('light spots in image')
# plt.imshow(rgb2gray(img1))
# %%
dark_spots = np.array((img1 < 3).nonzero()).T
dark_spots.shape

# %%
plt.plot(dark_spots[:, 1], dark_spots[:, 0], 'o')
plt.imshow(img1)
plt.title('dark spots in image')


# %%
import boto3
# import awscli

def detect_labels(photo, bucket):

    client=boto3.client('rekognition', region_name='us-east-1')

    response = client.detect_labels(Image={'S3Object':{'Bucket': bucket, 'Name':photo}},
        MaxLabels=3)

    print('Detected labels for ' + photo)
    print()
    for label in response['Labels']:
        print ("Label: " + label['Name'])
        print ("Confidence: " + str(label['Confidence']))
        print ("Instances:")
        for instance in label['Instances']:
            print ("  Bounding box")
            print ("    Top: " + str(instance['BoundingBox']['Top']))
            print ("    Left: " + str(instance['BoundingBox']['Left']))
            print ("    Width: " +  str(instance['BoundingBox']['Width']))
            print ("    Height: " +  str(instance['BoundingBox']['Height']))
            print ("  Confidence: " + str(instance['Confidence']))
            print()

        print ("Parents:")
        for parent in label['Parents']:
            print ("   " + parent['Name'])
        print ("----------")
        print ()
    return len(response['Labels'])


def main():
    photo='jpg_imgs/cape-cod/download.2.jpg'
    bucket='cbh-capstone3'
    label_count=detect_labels(photo, bucket)
    print("Labels detected: " + str(label_count))


if __name__ == "__main__":
    main()


# %%
