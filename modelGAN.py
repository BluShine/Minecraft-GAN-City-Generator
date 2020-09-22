import tensorflow as tf
from tensorflow.keras import layers

def make_generator_model(BASE):
    B2 = BASE * 2
    B3 = BASE * 4
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

def make_discriminator_model(BASE):
    B3 = BASE * 4
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