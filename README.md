[![Language](https://img.shields.io/badge/language-python-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/framework-tensorflow-orange.svg)](https://github.com/enesdemirag)
[![License](http://img.shields.io/:license-mit-green.svg)](http://enesdemirag.mit-license.org)
# map-GANerator
2D Map Generator - Generative Adversarial Network

## Prerequisites
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [tensorflow](https://www.tensorflow.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- [termcolor](https://pypi.org/project/termcolor/)

## Dataset
I wrote [this script](map-scrapper.py) to get random 512x512 map images from [mapgen2](https://github.com/redblobgames/mapgen2) and collect the dataset with 6000 maps.

## Run
- Get the dataset from [here](https://www.kaggle.com/enesdemirag/mapgen2).
- Clone this repo and run [train](train.py) script to train from start with your desired hyperparameters for dataset and models.
> You can also run [this notebook](mapGAN.ipynb) on colab for faster results.

## Results
![epochs](results.gif)

## Model
I create a Deep Convolutional Generative Adversarial Network (DCGAN) using Tensorflow with the help of Keras.

### Configurations
```
Resolution:       64px
Epochs:             50
Batch Size:         32
Buffer Size:      6000
Seed Size:         100
```

### Generator
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 4096)              413696    
_________________________________________________________________
reshape (Reshape)            (None, 4, 4, 256)         0         
_________________________________________________________________
up_sampling2d (UpSampling2D) (None, 8, 8, 256)         0         
_________________________________________________________________
conv2d (Conv2D)              (None, 8, 8, 256)         590080    
_________________________________________________________________
batch_normalization (BatchNo (None, 8, 8, 256)         1024      
_________________________________________________________________
activation (Activation)      (None, 8, 8, 256)         0         
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 16, 16, 256)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 16, 16, 256)       590080    
_________________________________________________________________
batch_normalization_1 (Batch (None, 16, 16, 256)       1024      
_________________________________________________________________
activation_1 (Activation)    (None, 16, 16, 256)       0         
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 32, 32, 256)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 128)       295040    
_________________________________________________________________
batch_normalization_2 (Batch (None, 32, 32, 128)       512       
_________________________________________________________________
activation_2 (Activation)    (None, 32, 32, 128)       0         
_________________________________________________________________
up_sampling2d_3 (UpSampling2 (None, 64, 64, 128)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 64, 64, 128)       147584    
_________________________________________________________________
batch_normalization_3 (Batch (None, 64, 64, 128)       512       
_________________________________________________________________
activation_3 (Activation)    (None, 64, 64, 128)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 64, 64, 3)         3459      
_________________________________________________________________
activation_4 (Activation)    (None, 64, 64, 3)         0         
=================================================================
```

### Discriminator
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_5 (Conv2D)            (None, 32, 32, 32)        896       
_________________________________________________________________
leaky_re_lu (LeakyReLU)      (None, 32, 32, 32)        0         
_________________________________________________________________
dropout (Dropout)            (None, 32, 32, 32)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 16, 16, 64)        18496     
_________________________________________________________________
zero_padding2d (ZeroPadding2 (None, 17, 17, 64)        0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 17, 17, 64)        256       
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 17, 17, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 17, 17, 64)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 9, 9, 128)         73856     
_________________________________________________________________
batch_normalization_5 (Batch (None, 9, 9, 128)         512       
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 9, 9, 128)         0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 9, 9, 128)         0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 9, 9, 256)         295168    
_________________________________________________________________
batch_normalization_6 (Batch (None, 9, 9, 256)         1024      
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 9, 9, 256)         0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 9, 9, 256)         0         
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 9, 9, 512)         1180160   
_________________________________________________________________
batch_normalization_7 (Batch (None, 9, 9, 512)         2048      
_________________________________________________________________
leaky_re_lu_4 (LeakyReLU)    (None, 9, 9, 512)         0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 9, 9, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 41472)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 41473     
=================================================================
```