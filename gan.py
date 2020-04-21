# Import necassary packages
import os 
import time
import numpy as np                                          # for gaussian random 
from PIL import Image                                       # to save results
from tqdm import tqdm                                       # to visualize the training process
import tensorflow as tf                                     # framework
from termcolor import cprint                                # colored printing
import matplotlib.pyplot as plt                             # to show results
from tensorflow.keras.layers import LeakyReLU               # leaky version of a Rectified Linear Unit
from tensorflow.keras.optimizers import Adam                # the Adam optimizer
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import UpSampling2D, Conv2D

# Configure
resolution = 64                                             # should be multiple of 32
channels = 3                                                # rgb image
dataset_path = "dataset/"
epochs = 50
batch_size = 32
buffer_size = 60000

# Preview
preview_rows = 3
preview_cols = 3
preview_margin = 12

# Input noise vector size
seed_size = 100

# Print
cprint("--- Configurations ---", "blue", attrs=['bold'])
cprint("Resolution: %10d" %resolution, "green")
cprint("Epochs: %14d" %epochs, "green")
cprint("Batch Size: %10d" %batch_size, "green")
cprint("Buffer Size: %9d\n" %buffer_size, "green")

