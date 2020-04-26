# Import dependencies
import numpy as np
from PIL import Image
import tensorflow as tf
from random import randint
import matplotlib.pyplot as plt

# Load model
g = tf.keras.models.load_model('mapGAN.h5', compile=False)
g.compile(loss='binary_crossentropy', optimizer='adam')

# Generate
noise = tf.random.normal([1, randint(0, 100)])
generated_map = g(noise, training=False)

# Show generated image
plt.imshow(generated_map[0, :, :, 0])
plt.show()
im = Image.fromarray(generated_map)
im.save("map.png")