import os

# Dataset configs

NUM_KEYPOINTS = 17 # COCO number of keypoints per person
BBOX_SCALE = 1.25
NUM_EXAMPLER_PER_TFRECORD = 2048

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
# heatmaps
LABEL_WIDTH = 64 
LABEL_HEIGHT = 64 
LABEL_SHAPE  = (LABEL_HEIGHT, LABEL_WIDTH, NUM_KEYPOINTS)
GAUSSIAN_KERNEL = 7

# Training seting
BATCH_SIZE = 16
SHUFFLE_BUFFER = 1000 # for better randomness increase if ram is avail

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