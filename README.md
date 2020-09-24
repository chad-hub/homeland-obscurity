# Homeland-Obscurity
Using deep learning to classify home architecture


## Inspiration

<p float="center">
  <img src="/eda_plots/hagia_sophia.PNG" alt="drawing" width="400"/>
  <img src="eda_plots/duomo.png" alt="drawing" width="400"/>
</p>

Convolutional Neural Networks (CNNs) are especially good at processing images and featurizing their shapes, edges, curves, and depth. As I've learned more about CNNs and applied them throughout my work, I found myself wanting to look deeper at these networks while applying them to something I find personally interesting. Architecture is the perfect means with which to explore this deeper. I've been fortunate to have traveled many places all over the world and experienced a wide variety of cultures. While traveling, I was always drawn to the unique architectures associated with different cultures - and how they becuase the pride of each location. I chose to process images of different home architectural styles. My reasoning:
  1) There are many home architectures that have very distinguishing features
  2) There are many resources for home images
  
 ## Data
 | Architecture | Train Images | Test Images |
 |--------------|--------------|-------------|
 | Tudor | 508 | 40 |
 | Modern | 600 | 49 |
 | Victorian | 476 | 41 |
 | Ranch | 426 | 42 |
 | Modern | 500 | 40 |
 
 Sources: Zillow.com, google images, bing images, Pintrest
 
<p float="center">
  <img src="/eda_plots/tudor.png" alt="drawing" width="300" height="300"/>
  <img src="eda_plots/modern.PNG" alt="drawing" width="300" height="320"/>
  <img src="eda_plots/victorian.PNG" alt="drawing" width="300" height="300"/>
</p>

<p align="center">
  <img src="eda_plots/ranch.png" alt="drawing" width="300" height="340"/>
  <img src="eda_plots/cap_cod.PNG" alt="drawing" width="300" height="300"/>
</p>

I've highlighted what I believe to be the most prominent and consistent features exhibited by the five home styles I selected. I sought out images that were fairly consistent with these features, and opted for the photos that most centrally displayed the homes. 

## CNNs

CNNS are widely regarded for thir ability to efficiently process images. In fact, they were originally modeled after how neurons in the visual cortex interact. Research by David H. Hubel and Torsten Wiesel in the late 1950s proved that neurons in the visual cortex have a small receptive field, and react only to specific shapes. Different neurons react to different shapes, and together they form the visual field. Their research also showed that some neurons react to more complex patterns, and that these patterns were combinations of the simpler shapes perceived by other neurons. 

This study laid the ground work for CNNs. CNNs operate in a similar fashion
<p align="center">
  <img src="/eda_plots/home_cnn.png" alt="drawing" width="600" height="300"/>
</p>
 
In the above illustration, each individual filter will travel the entirety of the image, capturing the filters respective patterns and aggregating that information into what are called activation maps, or convolutions. A helpful analogy is to think about polarized sunglasses: only the light that is aligned with the orientation of the polarized lens reaches our eyes. These activation maps are then pooled, which is a downsizing operation. The pooling captures the strongest signals in the activation maps and then reduces the size of the convolution. At shallow depths, these activation maps capture simple shapes, lines, edges, etc. These are called 'spatial patterns'. In and of themselves, these spatial patterns don't provide a whole lot of insight. As we include more and more convolutional layers, however, these activation maps begin to capture more complex, or 'cross-channel' patterns. Keeping with the architecture analogy, shallow layers capture a sharp angle, line, or edge, and as we get deeper we begin to see a door, window, and house begin to form. After the pooling layer, we flatten out the information and connect it to our classifications - 5 in our case. 

## Transfer Learning
The true power of CNNs are most evident when we employ transfer-learning. With transfer learning, we utilize a pre-trained network and adjust it for our needs. Tehre are many readily available models to choose from, but I was most interested in, and had the best success with the Xception architecture (Francois Chollet - 2016). This model is trained on 350 million images and around 17,000 classes, and heavily features seperable convolutional layers. 

<p align="center">
  <img src="/eda_plots/xception.PNG" alt="drawing" width="300"/>
</p>

  
