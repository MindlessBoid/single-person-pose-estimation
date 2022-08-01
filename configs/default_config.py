import os
import numpy as np
# Dataset configs
NUM_KEYPOINTS = 17 # COCO number of keypoints per person
MIN_NUM_KEYPOINTS = 5 # min number of keypoitsn one person MUST have
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
HM_SIGMA = 1

# hourglass settings
HG_NUM_CHANNELS = 256 # only used inside hg module
HG_NUM_STACKS = 2


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

# Skeleton and stuff
COCO_INDEX_FLIP_PAIRS = [[1, 2],
                         [3, 4],
                         [5, 6],
                         [7, 8],
                         [9, 10],
                         [11, 12],
                         [13, 14],
                         [15, 16]]
COCO_KEYPOINT_LABELS = ["nose", 
                        "left_eye", "right_eye", 
                        "left_ear", "right_ear", 
                        "left_shoulder", "right_shoulder", 
                        "left_elbow", "right_elbow", 
                        "left_wrist", "right_wrist", 
                        "left_hip", "right_hip", 
                        "left_knee", "right_knee", 
                        "left_ankle", "right_ankle"]
#https://matplotlib.org/stable/gallery/color/named_colors.html
COCO_KEYPOINT_COLORS = ['red',
                        'brown', 'chocolate',
                        'orange', 'tan',
                        'lime', 'teal',
                        'navy', 'violet',
                        'black', 'coral',
                        'yellow', 'gold',
                        'cyan', 'green',
                        'orchid', 'indigo']
COCO_SKELETON = np.array([
                    [16, 14],
                    [14, 12],
                    [17, 15],
                    [15, 13],
                    [12, 13],
                    [6, 12],
                    [7, 13],
                    [6, 7],
                    [6, 8],
                    [7, 9],
                    [8, 10],
                    [9, 11],
                    [2, 3],
                    [1, 2],
                    [1, 3],
                    [2, 4],
                    [3, 5],
                    [4, 6],
                    [5, 7]]) - 1

