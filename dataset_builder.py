import tensorflow as tf
import numpy as np
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

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
    ds_train = ds_train.map(self.prepare_train_example, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.batch(self.batch_size) #batch after shuffling to get unique batches at each epoch
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)#good practice to end the pipeline by prefetching

    ds_valid = tf.data.TFRecordDataset(self.valid_filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE) #create dataset
    ds_valid = ds_valid.map(self.parse_tfrecord_fn, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds_valid = ds_valid.map(self.prepare_valid_example, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds_valid = ds_valid.batch(self.batch_size) #batch after shuffling to get unique batches at each epoch
    ds_valid = ds_valid.repeat()
    ds_valid = ds_valid.prefetch(tf.data.experimental.AUTOTUNE)#good practice to end the pipeline by prefetching

    return ds_train, ds_valid
  def get_ds_prediction(self, batch_size = 16) -> tf.data.Dataset:
    ''' For prediction, using valid dataset
        Giving larger batch_size for fast prediction -> may cause OOM
    '''
    ds_valid = tf.data.TFRecordDataset(self.valid_filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE) 
    ds_valid = ds_valid.map(self.parse_tfrecord_fn, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds_valid = ds_valid.map(self.prepare_prediction_example, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds_valid = ds_valid.batch(batch_size) 
    ds_valid = ds_valid.prefetch(tf.data.experimental.AUTOTUNE)
    return ds_valid


  def prepare_train_example(self, example):
    # Getting all the needed data first
    raw_image = example['image']
    raw_image_height = example['height']
    raw_image_width = example['width']
    kpts_x = example['keypoints/x']
    kpts_y = example['keypoints/y']

    # Augmentation in raw image's space (shape)
    aug_image, aug_kpts_x, aug_kpts_y = self.tf_augment(raw_image, kpts_x, kpts_y)
    aug_image.set_shape(raw_image.get_shape()) # set the shape same so it wont appear None

    # Generate heatmaps
    # Recal x, y in heatmap's space
    h_ratio = self.label_shape[0] / raw_image_height
    w_ratio = self.label_shape[1] / raw_image_width
    aug_kpts_x = aug_kpts_x * tf.cast(w_ratio, tf.float32) # in label_shape (64, 64)
    aug_kpts_y = aug_kpts_y * tf.cast(h_ratio, tf.float32) # in label_shape (64, 64)

    heatmaps = self.tf_gen_heatmaps(aug_kpts_x, aug_kpts_y)
    
    # resize imgage to desire size
    aug_image = tf.image.resize(aug_image, (self.image_shape[0], self.image_shape[1]))
    

    return aug_image, heatmaps
  
  def prepare_valid_example(self, example):
    # Getting all the needed data first
    raw_image = example['image']
    raw_image_height = example['height']
    raw_image_width = example['width']
    kpts_x = example['keypoints/x']
    kpts_y = example['keypoints/y']

    # Generate heatmaps
    # Recal x, y in heatmap's space
    h_ratio = self.label_shape[0] / raw_image_height
    w_ratio = self.label_shape[1] / raw_image_width
    kpts_x = kpts_x * tf.cast(w_ratio, tf.float32) # in label_shape (64, 64)
    kpts_y = kpts_y * tf.cast(h_ratio, tf.float32) # in label_shape (64, 64)

    heatmaps = self.tf_gen_heatmaps(kpts_x, kpts_y)
    
    # resize imgage to desire size
    raw_image = tf.image.resize(raw_image, (self.image_shape[0], self.image_shape[1]))
    
    return raw_image, heatmaps
  
  def prepare_prediction_example(self, example):
    '''
    Basically the same with ds_valid but we dont generate heatmaps nor recalculate keypoints and get all meta data
    '''
    raw_image = example['image']

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
    resized_image = tf.image.resize(raw_image, (self.image_shape[0], self.image_shape[1]))

    return resized_image, meta

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
  
  def np_augment(self, image, kpts_x, kpts_y):
    '''
    Augment the image and keypoints
    Only augment the valid keypoints (not (0, 0)) and all keypoints has to be augemted in image shape (hasnt been resized)
    :param kpts_x, kpts_y: x and y coordisnate in shape of (number of kpts):
    '''
    imgaug_kpts = [] # to store imgaug keypoint format
    idxs = []
    for i, kpt in enumerate(zip(kpts_x, kpts_y)):
      if 0 < kpt[0] < image.shape[1] and 0 < kpt[1] < image.shape[0]:
        imgaug_kpts.append(Keypoint(x = kpt[0], y = kpt[1]))
        idxs.append(i)
    kptsoi = KeypointsOnImage(imgaug_kpts, shape = image.shape)

    #augment
    seed = np.random.randint(2**32-1)
    ia.seed(seed)
    seq = iaa.Sequential([
      iaa.Affine(scale = (0.75, 1.25), rotate = (-30, 30)),
      iaa.Fliplr(0.5),
      ], random_order = False) #cause rotate/scale then flip can be problematic
    aug_img, aug_kps = seq(image = image, keypoints = kptsoi)

    arr = aug_kps.to_xy_array()
    output_kpts_x = np.zeros(shape = self.num_keypoints , dtype = np.float32)
    output_kpts_y = np.zeros(shape = self.num_keypoints , dtype = np.float32)

    for i, xy in zip(idxs, arr):
      output_kpts_x[i] = xy[0]
      output_kpts_y[i] = xy[1]
    
    return aug_img, output_kpts_x, output_kpts_y
  
  def tf_augment(self, image, kpts_x, kpts_y):
    return tf.numpy_function(self.np_augment, [image, kpts_x, kpts_y], (tf.float32, tf.float32, tf.float32))
 

  def np_gen_heatmaps(self, kpts_x, kpts_y):
    '''
    Creates 2D heatmaps from keypoints coordinates for one single example/image
    :param kpts_x, kpts_y: x and y coordisnate in shape of (number of kpts)
    :output: array of heatmaps (heatmap_width, heatmap_height, number of kpts)
    '''
    assert len(kpts_x) == len(kpts_y) == self.num_keypoints
    heatmaps = np.zeros(self.label_shape, dtype = np.float32)
    for i in range(self.num_keypoints):
      x = int(kpts_x[i])
      y = int(kpts_y[i])
      if 0 < x < self.label_shape[1] and 0 < y < self.label_shape[0]: 
        heatmaps[y][x][i] = 1.0
        heatmaps[:,:,i] = cv2.GaussianBlur(heatmaps[:,:,i], (self.gaussian_kernel, self.gaussian_kernel), 0)#blur
        heatmaps[:,:,i] = heatmaps[:,:,i] / heatmaps[:,:,i].max()#normalize
    return heatmaps

  def tf_gen_heatmaps(self, kpts_x, kpts_y):
    return tf.numpy_function(self.np_gen_heatmaps, [kpts_x, kpts_y], tf.float32)

  # Some utils
  @staticmethod
  def get_ds_length(filenames):
    length = 0
    for name in filenames:
      temp = name.split('-')[-1]
      temp = temp.split('.')[0]
      length += int(temp)
    return length