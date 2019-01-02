# Original Code
# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_run_loop.py
# modified by doublejtoh, 12.31.2018.

import tensorflow as tf
import os
import re
import json
from config import IMG_EXT

def exclude_batch_norm(name):
    """
    exclude batch_norm variables from l2 loss update.
    """

    return 'batch_normalization' not in name

def learning_rate_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates,
    base_lr=0.1, warmup=False):
    """
    :param batch_size: number of images in each training batch.
    :param batch_denom: scaler scalar.
    :param num_images: total number of training images.
    :param boundary_epochs: list of ints representing the epochs at which decaying learning rate. e.g) [20, 40, 60]
    :param decay_rates: list of floats representing decay rates. e.g) [1, 0.1, 0.01, 0.001]
    :param base_lr: initial learning rate scaled with batch_denom
    :param warmup: 5 epoch warmup for initial learning rate.
    :return: function that returns learning rate from global step(arg)
    """

    # scaled learning_rate
    initial_learning_rate = base_lr * batch_size / batch_denom # 0.008
    # total number of batches in each epoch.
    batches_per_epoch = num_images / batch_size

    # boundary of batches.
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    # decayed learning rate boundaries.
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        lr = tf.train.piecewise_constant(global_step, boundaries, vals)
        if warmup:
            warmup_steps = int(batches_per_epoch * 5)
            warmup_lr = (
                initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
                    warmup_steps, tf.float32))

            # if global step is smaller than warmup_steps, returns warmup_learning rate.
            return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
        return lr

    return learning_rate_fn

def process_record(filename, is_training, resized_image_width, resized_image_height, label=None):
    """
    read JPEG file and resize images.
    :return:
    """
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)

    if is_training:
        # add four extra pixels on each side. (Zero padding)
        image = tf.image.resize_image_with_crop_or_pad(
            image, resized_image_height + 1, resized_image_width + 1
        )

        # Randomly crop [resized_image_height, resized_image_width] from image.
        image = tf.random_crop(image, [resized_image_height, resized_image_width, 3])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Subtract the per-pixel mean.
        image = tf.image.per_image_standardization(image)

    else:
        # Resize image
        image = tf.image.resize_images(image, [resized_image_height, resized_image_width])

        # Subtract the per-pixel mean.
        return tf.image.per_image_standardization(image)
    return image, label

def input_fn_train(inputs, labels, image_width, image_height, batch_size, num_epochs, process_record_fn, shuffle_buffer=100000, num_parallel_batches=1):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.repeat(num_epochs)

    # better performance than dataset.map and then dataset.batch
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda filename, label: process_record_fn(filename, True, image_width, image_height, label),
            batch_size=batch_size,
            num_parallel_batches=num_parallel_batches,
            drop_remainder=False)
    )

    # To keep above processing out of the critical training path.
    # reference: https://github.com/tensorflow/models/blob/master/official/resnet/resnet_run_loop.py#L97~#L102
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    iter = dataset.make_one_shot_iterator()

    return iter

def input_fn_eval(inputs, image_width, image_height, batch_size, num_epochs, process_record_fn, num_parallel_batches=1):
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.repeat(num_epochs)
    # better performance than dataset.map and then dataset.batch
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda filename: process_record_fn(filename, False, image_width, image_height),
            batch_size=batch_size,
            num_parallel_batches=num_parallel_batches,
            drop_remainder=False)
    )
    # To keep above processing out of the critical training path.
    # reference: https://github.com/tensorflow/models/blob/master/official/resnet/resnet_run_loop.py#L97~#L102
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    iter = dataset.make_one_shot_iterator()

    return iter

def get_filenames_labels(img_dir, label_mapped_json_path):
    """

    :param img_dir: img directory path.
    :param label_mapped_json_path: label mapped json file path.
    :return: filenames, labels casted in tf.Constant
    """
    re_model = re.compile("^(\d+)_")

    with open(label_mapped_json_path) as f:
        data = json.load(f)

    filenames = os.listdir(img_dir)
    filenames = [filename for filename in filenames if filename.endswith(IMG_EXT)]
    labels_mapped = [data[re_model.match(filename).group(1)] for filename in filenames]
    filenames = [os.path.join(img_dir, filename) for filename in filenames]
    filenames = tf.constant(filenames)
    labels = tf.constant(labels_mapped, dtype=tf.int32)
    return filenames, labels

def get_filenames(img_dir, sorted=False):
    """

    :param img_dir:  img directory path.
    :return: filenames casted in tf.Constant
    """
    filenames = os.listdir(img_dir)
    if sorted:
        filenames.sort()
    filenames = [filename for filename in filenames if filename.endswith(IMG_EXT)]
    filenames = [os.path.join(img_dir, filename) for filename in filenames]
    return tf.constant(filenames)

def get_block_sizes(resnet_size):
    """
    :return: block_sizes determined by resnet_size.
    """
    return {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8 ,36, 3]
    }.get(resnet_size)


