import tensorflow as tf
from os import path as ospath
from model import Model
from config import LABELS_PRED, TRAINING_IMG_DIR, TRAINING_DATA_DIR, TRAINING_JSON_PATH, TEST_IMG_DIR, TEST_DATA_DIR, CHECKPOINT_PATH, CHECKPOINT_SAVE_EPOCH, CHECKPOINT_MAX_TO_KEEP, _IMAGE_WIDTH, _IMAGE_HEIGHT, _IMAGE_CHANNELS, _NUM_CLASSES, _NUM_IMAGES

def define_flags():
    tf.app.flags.DEFINE_integer('max_training_epochs', 100000,
                               'Maximum training epoch. \n'
                               'If larger, training ends.')
    tf.app.flags.DEFINE_integer('batch_size', 16,
                                'Batch size')
    tf.app.flags.DEFINE_string('training_data_dir', TRAINING_IMG_DIR,
                               'Training data directory')
    tf.app.flags.DEFINE_string('training_json_path', TRAINING_JSON_PATH,
                               'Training data labels mapping file path')
    tf.app.flags.DEFINE_string('test_data_dir', TEST_IMG_DIR,
                               'Test data directory')
    tf.app.flags.DEFINE_string('checkpoint_path', CHECKPOINT_PATH,
                               'Save/Saved checkpoint path')
    tf.app.flags.DEFINE_integer('num_images', _NUM_IMAGES,
                                'Total number of training data images.')
    tf.app.flags.DEFINE_integer('checkpoint_save_epoch', CHECKPOINT_SAVE_EPOCH,
                                'Checkpoint save for every "checkpoint_save_epoch" epoch.')
    tf.app.flags.DEFINE_integer('checkpoint_max_to_keep', CHECKPOINT_MAX_TO_KEEP,
                                'Checkpoint files max to keep')
    tf.app.flags.DEFINE_integer('resnet_size', 50,
                                'resnet size selection.'
                                'must be one of [50, 101, 152]')
    tf.app.flags.DEFINE_boolean('training_predict', False,
                                'On training dataset, \n'
                                'make labels_pred.txt (predictions) \n')
    tf.app.flags.DEFINE_string('training_predict_output_path', TRAINING_DATA_DIR,
                               'Output path where labels_pred.txt and \n')
    tf.app.flags.DEFINE_boolean('test_predict', False,
                                'On test dataset, \n'
                                'make labels_pred.txt (predictions) \n')
    tf.app.flags.DEFINE_string('test_predict_output_path', TEST_DATA_DIR,
                               'Output path where labels_pred.txt and \n')

def main():
    FLAGS = tf.app.flags.FLAGS
    resnet_model = Model(
        resnet_size=FLAGS.resnet_size,
        initial_kernel_size=7,
        initial_kernel_stride=2,
        kernel_strides=[1, 2, 2, 2],
        initial_pool_size=3,
        initial_pool_stride=2,
        initial_filters=64,
        input_width=_IMAGE_WIDTH,
        input_height=_IMAGE_HEIGHT,
        input_channels=_IMAGE_CHANNELS,
        num_classes=_NUM_CLASSES,
        data_format='channels_last'
    )
    if FLAGS.training_predict:
        resnet_model.predict(
            flags=FLAGS,
            data_dir=FLAGS.training_data_dir,
            pred_out_path=ospath.join(FLAGS.training_predict_output_path, LABELS_PRED)
        )
    elif FLAGS.test_predict:
        resnet_model.predict(
            flags=FLAGS,
            data_dir=FLAGS.test_data_dir,
            pred_out_path=ospath.join(FLAGS.test_predict_output_path, LABELS_PRED)
        )
    else:
        resnet_model.train(FLAGS)


if __name__ == '__main__':
    define_flags()
    main()
