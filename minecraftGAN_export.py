from schematic import SchematicFile

import tensorflow as tf

import numpy as np
import os
from tensorflow.keras import layers
import modelGAN

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

generator = modelGAN.make_generator_model(BASE)

discriminator = modelGAN.make_discriminator_model(BASE)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

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