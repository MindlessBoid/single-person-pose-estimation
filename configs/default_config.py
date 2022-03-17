import os

# Dataset configs
#ppl with kp > 0 
NUM_TRAIN_EXAMPLES_DEFAULT = 149813 
NUM_VALID_EXAMPLES_DEFAULT = 6352 
#ppl with kp > 8
NUM_TRAIN_EXAMPLES_HALF_KPTS = 108444
NUM_VALID_EXAMPLES_HALF_KPTS = 4491

NUM_KEYPOINTS = 17 # COCO number of keypoints per person
BBOX_SCALE = 1.25
NUM_EXAMPLER_PER_TFRECORD = 2048

# Directories
DATASET_DIR = 'dataset'

IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
TRAIN_IMAGES_DIR = os.path.join(IMAGES_DIR, 'train2017')
VALID_IMAGES_DIR = os.path.join(IMAGES_DIR, 'val2017')

ANNOT_DIR = os.path.join(DATASET_DIR, 'annotations')
TRAIN_ANNOT_FILE = os.path.join(ANNOT_DIR, 'person_keypoints_train2017.json')
VALID_ANNOT_FILE = os.path.join(ANNOT_DIR, 'person_keypoints_val2017.json')

TFRECORDS_DIR = os.path.join(DATASET_DIR, 'tfrecords')
TRAIN_TFRECORDS_DIR = os.path.join(TFRECORDS_DIR, 'train')
VALID_TFRECORDS_DIR = os.path.join(TFRECORDS_DIR, 'valid')


# Data frame configs
MIN_NUM_KEYPOINTS = NUM_KEYPOINTS // 2 # to filter out exmaples have less than half num keypoints