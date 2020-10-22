import tensorflow as tf
import matplotlib.pyplot as plt

"""
data augmentation: resize to 286 * 286
    and then random cropping into 256 *256
    
    with random flipping left-to-right, with prob 0.5
"""

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image, crop_height, crop_width ):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop( stacked_image, size=[2, crop_height, crop_width, 3])

    return cropped_image[0], cropped_image[1]

# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image, 256, 256 )

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def load_image_test(image_file):
      input_image, real_image = load(image_file)
      input_image, real_image = resize(input_image, real_image, 256, 256)
      input_image, real_image = normalize(input_image, real_image)

      return input_image, real_image

def train_ds( file_pattern, batch_size=1, buffer_size=400 ):
    train_dataset = tf.data.Dataset.list_files( file_pattern )
    train_dataset = train_dataset.map(load_image_train,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle( buffer_size )
    train_dataset = train_dataset.batch( batch_size )

    return train_dataset

def test_ds( file_pattern, batch_size=1 ):
    test_dataset = tf.data.Dataset.list_files( file_pattern )
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch( batch_size )

    return test_dataset

if __name__=='__main__':
    PATH = '../dataset/facades/train/100.jpg'
    inp, re = load(PATH)

    plt.figure(figsize=(6, 6))
    for i in range(4):
        rj_inp, rj_re = random_jitter(inp, re)
        plt.subplot(2, 2, i + 1)
        plt.imshow(rj_inp / 255.0)
        plt.axis('off')
    plt.show()