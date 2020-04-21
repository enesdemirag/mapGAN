# Import necassary packages
import os 
import time
import numpy as np
from model import GAN                                       # DCGAN model
from tqdm import tqdm                                       # to visualize the training process
import tensorflow as tf                                     # framework
from termcolor import cprint                                # colored printing
import matplotlib.pyplot as plt                             # to show results

# Configure
resolution = 64                                             # 32 * k,  k > 1
channels = 3                                                # rgb image
binary = os.path.join("dataset.npy")                        # Import dataset
epochs = 50
batch_size = 32
buffer_size = 6000
image_shape = (resolution, resolution, channels)

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
cprint("Buffer Size: %9d" %buffer_size, "green")
cprint("Seed Size: %11d\n" %seed_size, "green")

# Load data
cprint("Loading dataset from '%s'" %binary, "blue", attrs=['bold'])
data = np.load(binary)
dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size).batch(batch_size)

# Create models
gan = GAN(resolution=resolution, channel=channels)
g = gan.generator(seed_size, resolution, channels)
d = gan.discriminator(image_shape)

# Test generator and discriminator before training
cprint("Testing generator output..", "yellow", attrs=['bold'])
noise = tf.random.normal([1, seed_size])
generated_image = g(noise, training=False)
prediction = d(generated_image)
cprint("Generated image is %d%% real." %(prediction * 100), "red")

# Show generated image
plt.imshow(generated_image[0, :, :, 0])
plt.show()
