# Original Code
# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
# modified by doublejtoh, 12.31.2018.

import tensorflow as tf
from helper import *
from config import _BATCH_NORM_DECAY, _BATCH_NORM_EPSILON, _WEIGHT_DECAY, _MOMENTUM, LABELS_PRED
from dotmap import DotMap
import time
import re
import json

def batch_norm(inputs, training, data_format):
    return tf.layers.batch_normalization(
        inputs=inputs,
        axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=training,
        fused=True
    )

def fixed_padding(inputs, kernel_size, data_format):
    # kernel size에 따라 pad. (input size에 독립적)
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [
            [0, 0], [0, 0],
            [pad_beg, pad_end], [pad_beg, pad_end]
        ])
    else:
        padded_inputs = tf.pad(inputs, [
            [0, 0],
            [pad_beg, pad_end], [pad_beg, pad_end],
            [0, 0]
        ])
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):

    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format
    )

def residual_layer(inputs, filters, strides, num_blocks, data_format, training):

    # filter num of each resiudal block's output.
    out_filters = filters * 4

    # first block should use projection shortcut.
    inputs = residual_block(
        inputs=inputs,
        filters=filters,
        strides=strides,
        out_filters=out_filters,
        projection_shortcut=True,
        data_format=data_format,
        training=training
    )

    # other blocks don't use projection shortcut and its conv stride is 1.
    for _ in range(1, num_blocks):
        inputs = residual_block(
            inputs=inputs,
            filters=filters,
            strides=1,
            out_filters=out_filters,
            projection_shortcut=False,
            data_format=data_format,
            training=training
        )

    return inputs



def residual_block(inputs, filters, strides, out_filters, projection_shortcut, data_format, training):
    shortcut = inputs
    inputs = batch_norm(
        inputs=inputs,
        training=training,
        data_format=data_format
    )
    inputs = tf.nn.relu(inputs)

    # 1x1 convolution shortcut.
    # Shape of shorcut output should match with residual block's output
    if projection_shortcut is True:
        shortcut = conv2d_fixed_padding(
            inputs=inputs,
            filters=out_filters,
            kernel_size=1,
            strides=strides,
            data_format=data_format
        )

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=1,
        data_format=data_format
    )
    inputs = batch_norm(
        inputs=inputs,
        training=training,
        data_format=data_format
    )
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        data_format=data_format
    )
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=out_filters,
        kernel_size=1,
        strides=1,
        data_format=data_format
    )

    return inputs + shortcut





class Model:
    def __init__(self, resnet_size, initial_kernel_size, initial_kernel_stride, kernel_strides, initial_pool_size, initial_pool_stride, initial_filters,input_width, input_height, input_channels, num_classes, data_format):
        # network params
        self.resnet_size = resnet_size
        self.initial_kernel_size = initial_kernel_size
        self.initial_kernel_stride = initial_kernel_stride
        self.block_sizes = get_block_sizes(self.resnet_size)
        self.kernel_strides = kernel_strides
        self.initial_pool_size = initial_pool_size
        self.initial_pool_stride = initial_pool_stride
        self.initial_filters = initial_filters
        self.training = tf.placeholder(tf.bool)

        # input image params
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.data_format = data_format

        # build resnet
        self.build_resnet()

    def get_session(self, checkpoint_path, max_to_keep):

        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())


    def build_resnet(self):
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width, self.input_channels], name='inputs')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')

        inputs = self.inputs

        with tf.variable_scope('resnet_model'):
            if self.data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            # initial conv layer
            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=self.initial_filters, kernel_size=self.initial_kernel_size,
                strides=self.initial_kernel_stride, data_format=self.data_format)

            # initial max pool layer. NEED TO BE EXPERIMENTAL.
            inputs = tf.layers.max_pooling2d(
                inputs=inputs, pool_size=self.initial_pool_size,
                strides=self.initial_pool_stride, padding='SAME',
                data_format=self.data_format)

            # residual layers.
            for i, num_blocks in enumerate(self.block_sizes):
                filters = self.initial_filters * (2**i)
                inputs = residual_layer(
                    inputs=inputs,
                    filters=filters,
                    strides=self.kernel_strides[i],
                    num_blocks=num_blocks,
                    data_format=self.data_format,
                    training=self.training
                )

            inputs = batch_norm(inputs, self.training, self.data_format)
            inputs = tf.nn.relu(inputs)

            # average pooling is same as reduce_mean.
            # reduce_mean performs better than AveragePooling2D
            axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
            inputs = tf.reduce_mean(inputs, axes, keepdims=True) # output shape: [1, 1, channels]
            inputs = tf.squeeze(inputs, axes) # output shape: (channels, )

            # final dense layer.
            inputs = tf.layers.dense(inputs=inputs, units=self.num_classes) # output shape: (self.num_classes, )
            inputs = tf.identity(inputs, 'outputs')

            self.outputs = inputs

    def optimize(self, config):
        # learning rate
        self.learning_rate_fn = learning_rate_decay(
            batch_size=config.batch_size, batch_denom=256,
            num_images=config.num_images, boundary_epochs=[20, 40, 60, 80],
            decay_rates=[1, 0.1, 0.01, 0.001, 1e-4], warmup=True, base_lr=.128)

        self.lr = self.learning_rate_fn(self.global_step)
        tf.summary.scalar('learning_rate', self.lr)

        # for numerical stability.
        self.outputs = tf.cast(self.outputs, tf.float32)

        # for evaluation.
        self.predictions = {
            'classes': tf.argmax(self.outputs, axis=1),
            'probabilities': tf.nn.softmax(self.outputs, name='softmax_tensor')
        }

        # define accuracy over one batch.
        self.accuracy = tf.metrics.accuracy(labels=self.labels, predictions=self.predictions['classes'])
        tf.identity(self.accuracy[1], name='accuracy')
        tf.summary.scalar('accuracy', self.accuracy[1])

        # cross entropy
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            logits=self.outputs, labels=self.labels)
        tf.identity(cross_entropy, name='cross_entropy')
        tf.summary.scalar('cross_entropy', cross_entropy)

        # l2 loss
        l2_loss = config.weight_decay * tf.add_n(
            # loss is computed using fp32 for numerical stability.
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
             if exclude_batch_norm(v.name)])
        tf.summary.scalar('l2_loss', l2_loss)

        self.loss = cross_entropy + l2_loss
        tf.identity(self.loss, 'loss')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.lr,
                momentum=config.momentum).minimize(loss=self.loss, global_step=self.global_step)

    def train(self, flags):

        # training config casted as DotMap(dict that can be accessed as dot)
        config = DotMap(flags.flag_values_dict())
        config.num_classes = self.num_classes
        # total batches per epoch.
        config.num_batches = int(config.num_images / config.batch_size)
        config.weight_decay = _WEIGHT_DECAY
        config.momentum = _MOMENTUM

        # get filenames, labels for input_fn
        filenames, labels = get_filenames_labels(config.training_data_dir, config.training_json_path)

        # input function
        self.iter = input_fn_train(
            inputs=filenames,
            labels=labels,
            image_width=self.input_width,
            image_height=self.input_height,
            batch_size=config.batch_size,
            num_epochs=config.max_training_epochs,
            process_record_fn=process_record,
        )
        self.next_batch = self.iter.get_next()

        # global epoch (training set epoch)
        self.global_epoch = tf.Variable(0, name='global_epoch', trainable=False)

        # global step (mini-batch training step)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # optimize
        self.optimize(config)

        # get Session & restore checkpoint.
        self.get_session(config.checkpoint_path, config.checkpoint_max_to_keep)

        with self.sess as sess:
            last_epoch = sess.run(self.global_epoch)
            # # init input iterator.
            # sess.run(self.iter.initializer, feed_dict={ self.inputs: filenames, self.labels: labels })
            print("Batch size : %s" % (config.batch_size))
            print("Number of images for training : %s " % (config.num_images))
            print("Number of batches for each epoch: %s " % (config.num_batches))
            print("Max training epochs: %s " % (config.max_training_epochs))
            print("Starting from epoch %s " % (last_epoch))
            for _ in range(last_epoch, config.max_training_epochs):
                total_loss = 0
                total_acc = 0
                epoch = sess.run(self.global_epoch)
                for i in range(config.num_batches):
                    start_vect = time.time()
                    try:
                        inputs, labels = sess.run([self.next_batch[0], self.next_batch[1]])

                    except Exception as ex:
                        print(ex)
                        continue
                    _, loss_value, acc_value = sess.run([self.optimizer, self.loss, self.accuracy[1]], feed_dict={ self.inputs: inputs, self.labels: labels, self.training: True })
                    total_loss += loss_value
                    total_acc += acc_value
                    print("Batch : %s / %s, Batch loss: %.4f, Batch accuracy: %.4f, Training Runtime : %.2f seconds..." % (i+1, config.num_batches, loss_value, acc_value, time.time() - start_vect))
                print("Epoch: {}, Average Loss: {:.4f}, Training average accuracy: {:.4f}".format(epoch, total_loss / config.num_batches, total_acc / config.num_batches))

                # global step increase
                sess.run(tf.assign_add(self.global_epoch, 1, name='global_epoch_increment'))

                if epoch % config.checkpoint_save_epoch == 0:
                    self.saver.save(sess, config.checkpoint_path, global_step=epoch)
                    print("Saved checkpoint for epoch %s " % (epoch))

    def predict(self, flags, data_dir, pred_out_path):

        # get filenames for input_fn
        filenames = get_filenames(data_dir, True)

        # training config casted as DotMap(dict that can be accessed as dot)
        config = DotMap(flags.flag_values_dict())
        config.num_images = int(filenames.get_shape()[0])
        config.num_batches = int(config.num_images / config.batch_size)
        config.weight_decay = _WEIGHT_DECAY
        config.momentum = _MOMENTUM
        config.batch_steps = config.num_batches if config.num_images % config.batch_size == 0 and config.num_batches != 0 else config.num_batches + 1
        # input function
        self.iter = input_fn_eval(
            inputs=filenames,
            image_width=self.input_width,
            image_height=self.input_height,
            batch_size=config.batch_size,
            num_epochs=1,
            process_record_fn=process_record,
        )
        self.next_batch = self.iter.get_next()

        # global epoch
        self.global_epoch = tf.Variable(0, name='global_epoch', trainable=False)

        # for numerical stability.
        self.outputs = tf.cast(self.outputs, tf.float32)

        # for evaluation.
        self.predictions = {
            'classes': tf.argmax(self.outputs, axis=1),
            'probabilities': tf.nn.softmax(self.outputs, name='softmax_tensor')
        }

        # get Session & restore checkpoint.
        self.get_session(config.checkpoint_path, config.checkpoint_max_to_keep)

        with self.sess as sess:
            last_epoch = sess.run(self.global_epoch) - 1
            print("Batch size : %s" % (config.batch_size))
            print("Number of images for evaluation : %s " % (config.num_images))
            print("Number of batches for evaluation : %s " % (config.batch_steps))
            print("Evaluating from epoch %s " % (last_epoch))

            for _ in range(1):
                for i in range(config.batch_steps):
                    start_vect = time.time()
                    inputs = sess.run(self.next_batch)
                    predictions = sess.run(self.predictions['classes'], feed_dict={ self.inputs: inputs, self.training: False })
                    with open(pred_out_path + "_" + str(int(time.time())) + ".txt", 'a') as pred_file:
                        pred_file.writelines([str(line) + "\n" for line in predictions])
                    print("Batch : %s / %s, Evaluate Runtime : %.2f seconds..." % (i + 1, config.num_batches, time.time() - start_vect))
                print("Evaluated..")


















