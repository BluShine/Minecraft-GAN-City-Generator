import tensorflow as tf

import numpy as np
import os
import PIL
import time

import modelGAN

## Train a GAN to create 8x8 chunks of Minecraft buildings.

tf.executing_eagerly()

BUFFER_SIZE = 90000
BATCH_SIZE = 256
BASE = 2
B2 = BASE * 2
B3 = BASE * 4

EPOCHS = 20
EPOCH_EXPORT_STEP = 10
EPOCH_IMAGE_STEP = 10
noise_dim = 100
example_grid = 10
num_examples_to_generate = example_grid**2

minecraftSlices = np.load('data/np_samples_8x.npy')
minecraftSlices = minecraftSlices.reshape(minecraftSlices.shape[0], B3, B3, B3, 1).astype('float32')
minecraftSlices = minecraftSlices * 2 - 1

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(minecraftSlices).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = modelGAN.make_generator_model(BASE)

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

discriminator = modelGAN.make_discriminator_model(BASE)
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

generator_optimizer = tf.keras.optimizers.Adam(1e-3)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-3)

checkpoint_dir = './training_checkpoints_%dx' % B3
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images
    if (epoch % EPOCH_IMAGE_STEP == 0) :
        generate_and_save_images(generator,
                                epoch + 1,
                                seed)
    if (epoch % EPOCH_EXPORT_STEP == 0) :
        checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  generate_and_save_images(generator,
                           epochs,
                           seed).show()

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)


  preview = np.empty((0, B3 * example_grid,3), np.uint8)
  for i in range(example_grid):
      row = np.empty((B3, 0, 3), np.uint8)
      for j in range(example_grid):
          imageR = predictions[i*example_grid + j, :, :, :, 0] * 127.5 + 127.5
          imageG = np.average(imageR, axis=0)
          image = np.stack((imageR[0], imageG, imageG), axis=2)
          row = np.hstack((row, image))
      preview = np.vstack((preview, row))

  outputimage = PIL.Image.fromarray(preview.astype(np.uint8))
  outputimage.save('image_at_epoch_{:04d}.png'.format(epoch))
  return outputimage

train(train_dataset, EPOCHS)