import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path

from model.generators import Generator
from model.discriminators import Discriminator
from data.aligned_dataset import test_ds, train_ds

def generator_loss(disc_generated_output, gen_output, target):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (100. * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def generate_images(model, test_input, tar):
      prediction = model(test_input, training=True)
      plt.figure(figsize=(15,15))

      display_list = [test_input[0], tar[0], prediction[0]]
      title = ['Input Image', 'Ground Truth', 'Predicted Image']

      for i in range(3):
          plt.subplot(1, 3, i+1)
          plt.title(title[i])
          # getting the pixel values between [0, 1] to plot it.
          plt.imshow(display_list[i] * 0.5 + 0.5)
          plt.axis('off')
      plt.show()

def train():
    BASE_PATH = Path(__file__)

    train_dataset = train_ds( str(BASE_PATH.parents[1].joinpath('dataset/facades/train/*.jpg') ) )
    test_dataset  = test_ds( str( BASE_PATH.parents[1].joinpath('dataset/facades/test/*.jpg') ) )

    generator = Generator()
    discriminator = Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    import datetime
    log_dir = "logs/"

    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    epochs = 200

    @tf.function
    def train_step(input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)


    for epoch in range(epochs):
        start = time.time()

        # display.clear_output(wait=True)

        for example_input, example_target in test_dataset.take(1):
            generate_images(generator, example_input, example_target)
        print("Epoch: ", epoch)

        # Train
        for n, (input_image, target) in train_dataset.enumerate():
            print('.', end='')
            if (n + 1) % 100 == 0:
                print()

            train_step(input_image, target, epoch)

            # with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            #     gen_output = generator(input_image, training=True)
            #
            #     disc_real_output = discriminator([input_image, target], training=True)
            #     disc_generated_output = discriminator([input_image, gen_output], training=True)
            #
            #     gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
            #     disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
            #
            # generator_gradients = gen_tape.gradient(gen_total_loss,
            #                                         generator.trainable_variables)
            # discriminator_gradients = disc_tape.gradient(disc_loss,
            #                                              discriminator.trainable_variables)
            #
            # generator_optimizer.apply_gradients(zip(generator_gradients,
            #                                         generator.trainable_variables))
            # discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
            #                                             discriminator.trainable_variables))
            #
            # with summary_writer.as_default():
            #     tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            #     tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            #     tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            #     tf.summary.scalar('disc_loss', disc_loss, step=epoch)

        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))
    checkpoint.save(file_prefix=checkpoint_prefix)


if __name__=='__main__':
    train()
    # test_dataset = test_ds('../dataset/facades/test/*.jpg')
    # generator = Generator()
    # for example_input, example_target in test_dataset.take(1):
    #     generate_images(generator, example_input, example_target)