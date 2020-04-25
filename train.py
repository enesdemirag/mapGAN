# Import dependencies
import os 
import time
import numpy as np
from PIL import Image
from model import GAN                                       # DCGAN model
import tensorflow as tf                                     # framework
from termcolor import cprint                                # colored printing
import matplotlib.pyplot as plt                             # to show results

# Configure
resolution = 64                                             # 32 * k,  k > 1
channels = 3                                                # rgb image
binary = os.path.join("dataset.npy")                        # Import dataset
epochs = 50
batch_size = 32
buffer_size = 3000
image_shape = (resolution, resolution, channels)

# Preview
preview_rows = 3
preview_cols = 3
preview_margin = 10

# Input noise vector size
seed_size = 100

# Print
cprint("--- Configurations ---", "blue", attrs=['bold'])
cprint("Resolution: %8dpx" %resolution, "green")
cprint("Epochs: %14d" %epochs, "green")
cprint("Batch Size: %10d" %batch_size, "green")
cprint("Buffer Size: %9d" %buffer_size, "green")
cprint("Seed Size: %11d\n" %seed_size, "green")

# Load dataset
cprint("To load dataset please press Enter", "red", attrs=['bold'])
input()
cprint("Loading dataset from '%s'" %binary, "blue", attrs=['bold'])
data = np.load(binary)
dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size).batch(batch_size)

# Load models
cprint("To load models please press Enter", "red", attrs=['bold'])
input()
gan = GAN(resolution=resolution, channel=channels)
g = gan.generator(seed_size, resolution, channels)
d = gan.discriminator(image_shape)

cprint("--- Generator Model ---", "yellow", attrs=['bold'])
g.summary()
cprint("--- Discriminator Model ---", "yellow", attrs=['bold'])
d.summary()

# Test generator and discriminator before training
cprint("To test models please press Enter", "red", attrs=['bold'])
input()
cprint("Testing generator output..", "blue", attrs=['bold'])
noise = tf.random.normal([1, seed_size])
generated_image = g(noise, training=False)
prediction = d(generated_image)
cprint("Generated image is %d%% real." %(prediction * 100), "cyan")

# Show generated image
plt.imshow(generated_image[0, :, :, 0])
plt.show()

# To show clean looking epoch time
def nice_time(sec):
    h = int(sec / (60 * 60))
    m = int((sec % (60 * 60)) / 60)
    s = sec % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

@tf.function
def train_step(images):
    seed = tf.random.normal([batch_size, seed_size])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = g(seed, training=True)
        real_output = d(images, training=True)
        fake_output = d(generated_images, training=True)
        
        gen_loss = gan.generator_loss(fake_output)
        disc_loss = gan.discriminator_loss(real_output, fake_output)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, g.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, d.trainable_variables)

        gan.generator_optimizer.apply_gradients(zip(gradients_of_generator, g.trainable_variables))
        gan.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d.trainable_variables))
    
    return gen_loss, disc_loss

def train(dataset, epochs):
    fixed_seed = np.random.normal(0, 1, (preview_rows * preview_cols, seed_size))
    train_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        gen_loss_list = []
        disc_loss_list = []
        for image_batch in dataset:
            t = train_step(image_batch)
            gen_loss_list.append(t[0])
            disc_loss_list.append(t[1])

        g_loss = sum(gen_loss_list) / len(gen_loss_list)
        d_loss = sum(disc_loss_list) / len(disc_loss_list)

        epoch_elapsed = time.time() - epoch_start
        cprint("Epoch: %d, Generator Loss: %f, Discriminator Loss: %f, Elapsed Time: %s" %((epoch + 1), g_loss, d_loss, nice_time(epoch_elapsed)), "blue", attrs=['bold'])
        save_images(epoch, fixed_seed)
        g.save("mapGAN-" + str(epoch) + ".h5")

    train_elapsed = time.time() - train_start
    cprint("Training Time: %s" %nice_time(train_elapsed), "yellow", attrs=['bold'])

def save_images(index, noise):
    image_array = np.full((preview_margin * 2 + (preview_rows * (resolution + preview_margin)), preview_margin * 2 + (preview_cols * (resolution + preview_margin)), 3), 255, dtype=np.uint8)
    generated_images = g.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    image_count = 0
    for row in range(preview_rows):
        for col in range(preview_cols):
            r = row * (resolution + 16) + preview_margin
            c = col * (resolution + 16) + preview_margin
            image_array[r:r + resolution, c:c + resolution] = generated_images[image_count] * 255
            image_count += 1

    im = Image.fromarray(image_array)
    im.save("map-" + str(index + 1) + ".png")

# Train GAN
cprint("To train network please press Enter", "red", attrs=['bold'])
input()
train(dataset, epochs)

# Test generator and discriminator after training
cprint("To test models please press Enter", "red", attrs=['bold'])
input()
cprint("Testing generator output..", "blue", attrs=['bold'])
noise = tf.random.normal([1, seed_size])
generated_image = g(noise, training=False)
prediction = d(generated_image)
cprint("Generated image is %d%% real." %(prediction * 100), "cyan")

# Show generated image
plt.imshow(generated_image[0, :, :, 0])
plt.show()

# Save generator
cprint("To save models please press Enter", "red", attrs=['bold'])
input()
g.save("mapGAN.h5")
cprint("Generator model saved as 'mapGAN.h5'.", "blue", attrs=['bold'])
