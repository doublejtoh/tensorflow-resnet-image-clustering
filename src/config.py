"""
폴더와 임시 파일을 지정하는 Constants 들입니다.
"""

# dataset
DATASET = "WomenProducts"

# image format
IMG_EXT = "jpg"

# paths
TRAINING_IMG_DIR = "../img/" + DATASET + "/training"                                # training images
TRAINING_JSON_PATH = "../data/" + DATASET + "/training/label_mapped.json"           # training Label mapping json file. (Key: original label e.g. 15136734052, Value: mapped label(0~(num_classes-1)))
TRAINING_DATA_DIR = "../data/" + DATASET + "/training"
TEST_IMG_DIR = "../img/" + DATASET + "/test"                                        # test/evaluation images
TEST_DATA_DIR = "../data/" + DATASET + "/test"

# checkpoint
CHECKPOINT_PATH = "../models/" + DATASET + "/"
CHECKPOINT_SAVE_EPOCH = 1                                                           # checkpoint saving frequency. 1 means save checkpoint every epoch.
CHECKPOINT_MAX_TO_KEEP = 5                                                          # how many checkpoint files to maintain.

# files generated
IMG_PATHS = "img_paths.txt"                                                         # filename of image file list
LABELS_TRUE = "labels_true"                                                         # filename of true labels
LABELS_PRED = "labels_pred"                                                         # filename of predicted labels

# for clustering
NUM_IMGS_PER_MODEL = 70                                                             # images per cluster

# training hyper parameters.
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

# training data parameters.
_IMAGE_WIDTH = 300
_IMAGE_HEIGHT = 300
_IMAGE_CHANNELS = 3
_NUM_CLASSES = 153
_NUM_IMAGES = 10618

