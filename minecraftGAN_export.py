from schematic import SchematicFile

import tensorflow as tf

import glob
import imageio
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

tf.executing_eagerly()

BUFFER_SIZE = 60000
BATCH_SIZE = 256
BASE = 2
B2 = BASE * 2
B3 = BASE * 4
noise_dim = 100

EXPORTGRID = 16
EXPORTSPACING = 2

EXPORTPATH = "data/generatedExample.schematic"

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(BASE*BASE*BASE*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((BASE, BASE, BASE, 256)))
    assert model.output_shape == (None, BASE, BASE, BASE, 256) # Note: None is the batch size

    model.add(layers.Conv3DTranspose(128, (5, 5, 5), strides=(1, 1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, BASE, BASE, BASE, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv3DTranspose(64, (5, 5, 5), strides=(2, 2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, B2, B2, B2, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv3DTranspose(1, (5, 5, 5), strides=(2, 2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, B3, B3, B3, 1)

    return model

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv3D(64, (5, 5, 5), strides=(2, 2, 2), padding='same',
                                     input_shape=[B3, B3, B3, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv3D(128, (5, 5, 5), strides=(2, 2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints_%dx' % B3
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Load our pre-trained model
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

def generateStructures(model):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(tf.random.normal([EXPORTGRID**2, noise_dim]), training=False)

  #export a grid of generated structures.
  world = np.empty((B3, (B3 + EXPORTSPACING) * EXPORTGRID, 0), np.uint8)
  for i in range(EXPORTGRID):
      row = np.empty((B3, 0, B3), np.uint8)
      for j in range(EXPORTGRID):
          structure = predictions[i*EXPORTGRID + j, :, :, :, 0] * .5 + .5
          structure = np.concatenate((structure, 
                np.zeros((B3, EXPORTSPACING, B3), 
                dtype=np.uint8)), axis=1)
          row = np.concatenate((row, structure), axis=1)
      world = np.concatenate((world, row), axis=2)
      world = np.concatenate((world, 
                np.zeros((B3, (B3 + EXPORTSPACING) * EXPORTGRID, EXPORTSPACING), 
                dtype=np.uint8)), axis=2)
  return world

exportArea = generateStructures(generator)
print("generated %s" % str(exportArea.shape))
exportSchematic = SchematicFile(shape=exportArea.shape)
exportSchematic.blocks = exportArea
exportSchematic.save(EXPORTPATH)
print("exported to " + "data/generatedExample.schematic")