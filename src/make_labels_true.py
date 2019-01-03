import os
import re
import json
import argparse
import numpy as np
from config import *


def make_labels_true(training):
    """
    이미지 폴더의 파일명을 분석하여 정답 레이블(labels_true)을 생성하는 함수입니다.
    정답 레이블에 대응하는 이미지 파일 경로는 IMG_PATHS 에 저장된다.
    """
    # regex rule to extract true labels
    re_model = re.compile("^(\d+)_")

    # get list of files
    if os.path.exists(IMG_DIR) is False:
        print("Folder %s not found, copy image files to %s" % (IMG_DIR, IMG_DIR))
        return
    img_paths = os.listdir(IMG_DIR)
    img_paths.sort()
    img_paths = [filename for filename in img_paths if filename.endswith(IMG_EXT)]

    labels_true = [re_model.match(img_path).group(1) for img_path in img_paths]

    ''' json file write. for training convenience. '''
    unique_labels_true = list(set([label for label in labels_true]))
    unique_labels_json = dict((label, idx) for idx, label in enumerate(unique_labels_true))

    if training:
        with open(JSON_PATH, 'w') as f:
            json.dump(unique_labels_json, f)
    ''' '''

    print("Total number of images: %d, total number of models: %d" % (len(labels_true), len(unique_labels_true)))

    if os.path.exists(DATA_DIR) is False:
        os.makedirs(DATA_DIR)

    with open(os.path.join(DATA_DIR, IMG_PATHS), 'w') as f:
        f.writelines([line + "\n" for line in img_paths])

    if training:
        with open(os.path.join(DATA_DIR, LABELS_TRUE + ".txt"), 'w') as f:
            f.writelines([str(unique_labels_json[str(line)]) + "\n" for line in labels_true])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', action='store_true', default=False)
    args, _ = parser.parse_known_args()

    if args.training == True:
        IMG_DIR = TRAINING_IMG_DIR
        DATA_DIR = TRAINING_DATA_DIR
        JSON_PATH = TRAINING_JSON_PATH
    else:
        IMG_DIR = TEST_IMG_DIR
        DATA_DIR = TEST_DATA_DIR
        JSON_PATH = None

    make_labels_true(training=args.training)
