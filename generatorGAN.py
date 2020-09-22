from schematic import SchematicFile

import tensorflow as tf

import numpy as np
import os
from tensorflow.keras import layers
import modelGAN

## Using a pre-trained model, export a set of sample output data.

tf.executing_eagerly()

BUFFER_SIZE = 90000
BATCH_SIZE = 256
BASE = 2
B2 = BASE * 2
B3 = BASE * 4
noise_dim = 100

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

def makeStructures(structureCount):
    output = generator(tf.random.normal([structureCount, noise_dim]), training=False)
    output = output * .5 + .5
    output = output.numpy()
    output.shape = [structureCount, B3, B3, B3]
    return output