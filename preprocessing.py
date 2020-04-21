import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Get images names
dataset_path = "dataset/"
images = [img for img in os.listdir(dataset_path)]

# Resize resolution
resolution = 64
channels = 3

# Create array
dataset = np.ndarray(shape=(len(images), channels, resolution, resolution), dtype=np.float32)

i = 0
for image in images:
    img = load_img(dataset_path + "/" + image)              # PIL image
    img.thumbnail((resolution, resolution))                 # Resize

    x = img_to_array(img)                                   # Convert to numpy array
    x = x.reshape((channels, resolution, resolution))       # Reshape
    x = (x - 128.0) / 128.0                                 # Normalize (Between -1 to +1)
    dataset[i] = x
    i = i + 1

np.save("dataset.npy", dataset)
print("All images converted to array")