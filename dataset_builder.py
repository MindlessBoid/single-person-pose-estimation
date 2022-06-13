import tensorflow as tf
import numpy as np
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from utils import gaussian

class DatasetBuilder:
  def __init__(self, config, ratio = 1):
    '''
    Construct DatasetBuilder object
    :param cofig:
    :param ratio: dataset ratio (0, 1] if 1 take the whole dataset otherwise take ratio * num_tfrecords
    '''
    assert 0 < ratio <= 1
    self.image_shape = config.IMAGE_SHAPE # input
    self.label_shape = config.LABEL_SHAPE # output
    self.num_keypoints = config.NUM_KEYPOINTS
    self.gaussian_kernel = config.GAUSSIAN_KERNEL
    self.sigma = config.HM_SIGMA
    self.batch_size = config.BATCH_SIZE
    self.shuffle_buffer = config.SHUFFLE_BUFFER
    self.train_filenames = sorted(tf.io.gfile.glob(f"{config.TRAIN_TFRECORDS_DIR}/*.tfrec"))
    self.valid_filenames = sorted(tf.io.gfile.glob(f"{config.VALID_TFRECORDS_DIR}/*.tfrec"))
    if ratio < 1:
      self.train_filenames = self.train_filenames[:int(np.ceil(ratio * len(self.train_filenames)))]
      self.valid_filenames = self.valid_filenames[:int(np.ceil(ratio * len(self.valid_filenames)))]

    self.num_train_examples = self.get_ds_length(self.train_filenames)
    self.num_valid_examples = self.get_ds_length(self.valid_filenames)

    print(f'Train dataset with {len(self.train_filenames)} tfrecords and {self.num_train_examples} examples.')
    print(f'Valid dataset with {len(self.valid_filenames)} tfrecords and {self.num_valid_examples} examples.')

  def build_datasets(self) -> tf.data.Dataset:
    ds_train = tf.data.TFRecordDataset(self.train_filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE) #create dataset
    ds_train = ds_train.shuffle(self.shuffle_buffer) 
    ds_train = ds_train.map(self.parse_tfrecord_fn, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.map(self.prepare_example, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.map(self.make_train_label, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.batch(self.batch_size) #batch after shuffling to get unique batches at each epoch
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)#good practice to end the pipeline by prefetching

    ds_valid = tf.data.TFRecordDataset(self.valid_filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE) #create dataset
    ds_valid = ds_valid.map(self.parse_tfrecord_fn, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds_valid = ds_valid.map(self.prepare_example, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds_valid = ds_valid.map(self.make_valid_label, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds_valid = ds_valid.batch(self.batch_size) #batch after shuffling to get unique batches at each epoch
    ds_valid = ds_valid.repeat()
    ds_valid = ds_valid.prefetch(tf.data.experimental.AUTOTUNE)#good practice to end the pipeline by prefetching

    return ds_train, ds_valid

  def get_ds_prediction(self) -> tf.data.Dataset:
    ''' For prediction, using valid dataset
        Giving larger batch_size for fast prediction -> may cause OOM
    '''
    ds = tf.data.TFRecordDataset(self.valid_filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE) 
    ds = ds.map(self.parse_tfrecord_fn, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds = ds.map(self.prepare_prediction_example, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds = ds.batch(self.batch_size) 
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
  

  def make_train_label(self, image, kps_x, kps_y, kps_v):

    # Perform augmentation 1
    aug_image, aug_kps_x, aug_kps_y = self.tf_augment_1(image, kps_x, kps_y, kps_v)
    # Perform augmentation 2
    aug_image = self.augment_2(aug_image)
    # Gen heatmaps
    heatmaps = self.tf_gen_heatmaps(aug_kps_x, aug_kps_y)

    return aug_image, heatmaps
  

  def make_valid_label(self, image, kps_x, kps_y, kps_v):
    # Just gen heatmaps
    heatmaps = self.tf_gen_heatmaps(kps_x, kps_y)

    return image, heatmaps


  def prepare_example(self, example):
    '''
      Prepare one exmaple
      1. Extract meta data and image
      2. Resize the image to (256, 256)
      3. Recalculate the keypoints into heatmap's dimension (64, 64) and throw away invalid keypoints
      4. Return the image and x, y coordinates and visible flag (to mask later on)
    '''
    image = example['image']
    image_width = example['width']
    image_height = example['height']
    kps_x = example['keypoints/x']
    kps_y = example['keypoints/y']
    kps_v = example['keypoints/vis']

    # Resize the image
    image = tf.image.resize(image, size = (self.image_shape[1], self.image_shape[0]))

    # Recalculate x, y
    kps_x /= tf.cast(image_width, dtype = tf.float32) # normalize
    kps_y /= tf.cast(image_height, dtype = tf.float32) # normalize
    kps_x *= self.label_shape[1]
    kps_y *= self.label_shape[0]

    return image, kps_x, kps_y, kps_v

  def prepare_prediction_example(self, example):
    '''
    Basically the same with prepare_example but we dont generate heatmaps nor recalculate keypoints and get all meta data
    '''
    image = example['image']

    meta = {}
    meta['original_height'] = example['height']
    meta['original_width'] = example['width']
    meta['ann_id'] = example['ann_id']
    meta['image_id'] = example['image_id']
    meta['coco_url'] = example['coco_url']
    meta['keypoints/x'] = example['keypoints/x']
    meta['keypoints/y'] = example['keypoints/y']
    meta['keypoints/vis'] = example['keypoints/vis']
    meta['bbox_x'] = example['bbox_x'] # top left bbox
    meta['bbox_y'] = example['bbox_y'] # top left bbox
    meta['offset_width'] = example['offset_width'] # amount to add if the bbox out of the image 
    meta['offset_height'] = example['offset_height'] # amount to add if the bbox out of the image 

    # to 256x256
    resized_image = tf.image.resize(image, (self.image_shape[0], self.image_shape[1]))

    return resized_image, meta

  '''
    Augmentation pipeline:
    1. augment_1: Affect both keypoints and image
      - Random rotation
      - Random flip left-right
      - Random scale
      * This function uses imgaug library so need to wrap around tf_numpy_function -> slow
    2. augment_2: only affect image
      - Random brightness
      - Random contrast
      - Random hue
      - Random saturation
      * This one use tf.image -> faster
  '''
  def np_augment_1(self, image, kps_x, kps_y, kps_v):
    '''
      Augment the image and keypoints
      Only augment the valid keypoints (not (0, 0)) and all keypoints has to be augemted in image shape (hasnt been resized)
      kps_x, kps_y: x and y coordisnate in shape of (number of kps), in heatmap's space (64, 64)
      This function is robust it already eleminates all points outside of heatmap (64, 64)
    '''
    imgaug_kps = [] # to store imgaug keypoint format
    idxs = []
    # Extract x, y and store the keypoint's index
    for i, vis in enumerate(kps_v):
      if vis: 
        imgaug_kps.append(Keypoint(x = kps_x[i], y = kps_y[i]))
        idxs.append(i)
    kpsoi = KeypointsOnImage(imgaug_kps, shape = (self.label_shape[1], self.label_shape[0]))

    #augment
    seed = np.random.randint(2**32-1)
    ia.seed(seed)
    seq = iaa.Sequential([
      iaa.Affine(scale = (0.8, 1.2), rotate = (-30, 30)),
      iaa.Fliplr(0.5),
      ], random_order = True) #cause rotate/scale then flip can be problematic
    aug_img, aug_kps = seq(image = image, keypoints = kpsoi)

    arr = aug_kps.to_xy_array()
    output_kps_x = np.zeros(shape = self.num_keypoints , dtype = np.float32)
    output_kps_y = np.zeros(shape = self.num_keypoints , dtype = np.float32)

    for i, xy in zip(idxs, arr):
      output_kps_x[i] = xy[0]
      output_kps_y[i] = xy[1]
    
    return aug_img, output_kps_x, output_kps_y
  
  def tf_augment_1(self, image, kps_x, kps_y, kps_v):
    return tf.numpy_function(self.np_augment_1, [image, kps_x, kps_y, kps_v], (tf.float32, tf.float32, tf.float32)) # 4 inputs, 3 outputs

  def augment_2(self, image):
    '''
      image should be a tf.float32 tensor
    '''
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)

    # make sure image is in range [0.0, 1.0]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


  def np_gen_heatmaps(self, kps_x, kps_y):
    '''
    Creates 2D heatmaps from keypoints coordinates for one single example/image
    :param kps_x, kps_y: x and y coordisnate in shape of (number of kps)
    :output: array of heatmaps (heatmap_width, heatmap_height, number of kps)
    '''
    assert len(kps_x) == len(kps_y) == self.num_keypoints
    heatmaps = np.zeros(self.label_shape, dtype = np.float32)
    for i in range(self.num_keypoints):
      x = int(kps_x[i])
      y = int(kps_y[i])
      if 0 < x < self.label_shape[1] and 0 < y < self.label_shape[0]:
        hm = np.zeros((self.label_shape[1], self.label_shape[0]), dtype = np.float32) 
        heatmaps[: ,: , i] = gaussian(hm, (x, y), self.sigma)
        heatmaps[: ,: , i] = heatmaps[:,:,i] / heatmaps[:,:,i].max()#normalize
    return heatmaps

  def tf_gen_heatmaps(self, kps_x, kps_y):
    return tf.numpy_function(self.np_gen_heatmaps, [kps_x, kps_y], tf.float32)

  # Some utils
  @staticmethod
  def parse_tfrecord_fn(example):
    '''
    Used for extract examples for tfrecords
    '''
    feature_description = {
        "ann_id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_path": tf.io.FixedLenFeature([], tf.string),
        "coco_url": tf.io.FixedLenFeature([], tf.string),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "keypoints/x": tf.io.VarLenFeature(tf.float32),
        "keypoints/y": tf.io.VarLenFeature(tf.float32),
        "keypoints/vis": tf.io.VarLenFeature(tf.int64),
        "keypoints/num": tf.io.FixedLenFeature([], tf.int64),
        "bbox_x": tf.io.FixedLenFeature([], tf.float32),
        "bbox_y": tf.io.FixedLenFeature([], tf.float32),
        "offset_width": tf.io.FixedLenFeature([], tf.float32),
        "offset_height": tf.io.FixedLenFeature([], tf.float32)
    }
    
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.image.decode_image(example["image"], channels = 3, dtype = tf.float32, expand_animations =False)
    example["keypoints/x"] = tf.sparse.to_dense(example["keypoints/x"])
    example["keypoints/y"] = tf.sparse.to_dense(example["keypoints/y"])
    example["keypoints/vis"] = tf.sparse.to_dense(example["keypoints/vis"])
    return example


  @staticmethod
  def get_ds_length(filenames):
    length = 0
    for name in filenames:
      temp = name.split('-')[-1]
      temp = temp.split('.')[0]
      length += int(temp)
    return length