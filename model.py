# Import necassary packages
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import UpSampling2D, Conv2D

class GAN(object):
    def __init__(self, resolution=64, channel=3):
        self.rows = resolution
        self.cols = resolution
        self.channel = channel
        self.D = None                                       # Discriminator
        self.G = None                                       # Generator
        self.AM = None                                      # Adversarial Model
        self.DM = None                                      # Discriminator Model
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = Adam(1.5e-4, 0.5)        # should be tuned
        self.discriminator_optimizer = Adam(1.5e-4, 0.5)    # should be tuned

    def discriminator(self, img_shape):
        self.D = Sequential()

        self.D.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        self.D.add(LeakyReLU(alpha=0.2))

        self.D.add(Dropout(0.25))
        self.D.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        self.D.add(ZeroPadding2D(padding=((0,1), (0,1))))
        self.D.add(BatchNormalization(momentum=0.8))
        self.D.add(LeakyReLU(alpha=0.2))

        self.D.add(Dropout(0.25))
        self.D.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        self.D.add(BatchNormalization(momentum=0.8))
        self.D.add(LeakyReLU(alpha=0.2))

        self.D.add(Dropout(0.25))
        self.D.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        self.D.add(BatchNormalization(momentum=0.8))
        self.D.add(LeakyReLU(alpha=0.2))

        self.D.add(Dropout(0.25))
        self.D.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        self.D.add(BatchNormalization(momentum=0.8))
        self.D.add(LeakyReLU(alpha=0.2))

        self.D.add(Dropout(0.25))
        self.D.add(Flatten())
        self.D.add(Dense(1, activation='sigmoid'))

        return self.D

    def generator(self, seed_size, resolution, channels):
        self.G = Sequential()

        self.G.add(Dense(4 * 4 * 256, activation="relu", input_dim=seed_size))
        self.G.add(Reshape((4, 4, 256)))

        self.G.add(UpSampling2D())
        self.G.add(Conv2D(256, kernel_size=3, padding="same"))
        self.G.add(BatchNormalization(momentum=0.8))
        self.G.add(Activation("relu"))

        self.G.add(UpSampling2D())
        self.G.add(Conv2D(256, kernel_size=3, padding="same"))
        self.G.add(BatchNormalization(momentum=0.8))
        self.G.add(Activation("relu"))
   
        self.G.add(UpSampling2D())
        self.G.add(Conv2D(128, kernel_size=3, padding="same"))
        self.G.add(BatchNormalization(momentum=0.8))
        self.G.add(Activation("relu"))

        self.G.add(UpSampling2D(size=(int(resolution / 32), int(resolution / 32))))
        self.G.add(Conv2D(128, kernel_size=3, padding="same"))
        self.G.add(BatchNormalization(momentum=0.8))
        self.G.add(Activation("relu"))

        self.G.add(Conv2D(channels, kernel_size=3, padding="same"))
        self.G.add(Activation("tanh"))

        return self.G

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)