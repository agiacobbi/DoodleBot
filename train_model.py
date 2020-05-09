import os
import time
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras import layers
from PIL import Image


PATH_DATA = './processed_data/vehicle_data/'
PATH_CHECKPOINTS = './training_checkpoints/'
PATH_IMAGE_CHECKPOINTS = './progress_images/vehicle_images/'
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 150
NOISE_DIM = 100
NUM_TO_GENERATE = 25
SEED = tf.random.normal([NUM_TO_GENERATE, NOISE_DIM])


def load_data():
    print('Gathering data...')
    all_doodles = []

    for filename in os.listdir(PATH_DATA):
        if filename.endswith('.npy'):
            arr = np.load(PATH_DATA + filename)
            all_doodles.append(arr)
            del arr

    dataset = np.vstack(all_doodles)
    dataset = dataset.reshape(dataset.shape[0], 28, 28, 1)
    dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    print('done.')
    return dataset
    

def build_generator():
    print('Creating generator network...')
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    # model.summary()
    print('done.')
    return model

def build_discriminator():
    print('Creating discriminator network...')
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    # model.summary()
    print('done.')
    return model


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

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


def train(generator, discriminator, generator_optimizer, discriminator_optimizer, dataset):
    print("Training model...")
    checkpoint_dir = PATH_CHECKPOINTS
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    for epoch in range(EPOCHS):
        start = time.time()

        for image_batch in dataset:
            train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, image_batch)

        # Produce images for the GIF as we go
        generate_and_save_images(generator,
                                epoch + 1,
                                SEED)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    generate_and_save_images(generator,
                            EPOCHS,
                            SEED)

    print("done.")

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(5,5))

    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(PATH_IMAGE_CHECKPOINTS + 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()
    # plt.show()


def main():
    if not os.path.exists(PATH_IMAGE_CHECKPOINTS):
        os.makedirs(PATH_IMAGE_CHECKPOINTS)

    dataset = load_data()
    generator = build_generator()
    discriminator = build_discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    train(generator, discriminator, generator_optimizer, discriminator_optimizer, dataset)


if __name__ == "__main__":
    main()