import os

# Dataset configs
NUM_KEYPOINTS = 17 # COCO number of keypoints per person
MIN_NUM_KEYPOINTS = 10 # min number of keypoitsn one persone MUST have
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
HM_ACTIVATION = 'sigmoid'
# hourglass settings
HG_NUM_FILTERS = 256 # only used inside hg module
HG_NUM_STACKS = 4


# Training seting
BATCH_SIZE = 16
SHUFFLE_BUFFER = 1000 # for better randomness increase if ram is avail
LEARNING_RATE = 0.01 # default

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


# Temporary dirs
TEMPORARY_DIR = 'temp'
CHECKPOINTS_PATH = os.path.join(TEMPORARY_DIR, 'checkpoints')
LOGS_PATH =  os.path.join(TEMPORARY_DIR, 'logs')

