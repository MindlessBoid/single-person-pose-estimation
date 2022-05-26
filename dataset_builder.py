from IPython.core.display import set_matplotlib_formats
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
    ds_train = ds_train.map(self.tf_create_train_labels, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)#good practice to end the pipeline by prefetching

    ds_valid = tf.data.TFRecordDataset(self.valid_filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE) #create dataset
    ds_valid = ds_valid.map(self.parse_tfrecord_fn, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds_valid = ds_valid.map(self.prepare_valid_example, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds_valid = ds_valid.batch(self.batch_size) #batch after shuffling to get unique batches at each epoch
    ds_valid = ds_valid.map(self.tf_create_valid_labels, num_parallel_calls = tf.data.experimental.AUTOTUNE)
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

  
  def np_create_train_labels(self, images_batch, kps_x_batch, kps_y_batch):
    ''' Perform augmentation on images and keypoints
        Then using keypoints to yeil heatmaps
        All in batches
    Params:
      images_batch: batch of images, expected to be resized to the same shape
      kp_x_batch: batch of x coordinates, scaled to the current shape of image they belong to (256, 256)
      kp_y_batch: batch of y coordinates, scaled to the current shape of image they belong to (256, 256)
    Returns:
      Augumented images
      Heatmaps of augumented keypoints
    '''
    # Augmentation
    aug_images_batch, aug_kps_x_batch, aug_kps_y_batch = self.np_augment(images_batch, kps_x_batch, kps_y_batch)
    
    # Heatmaps 
    # recal x, y from (256, 256) -> (64, 64)
    w_ratio = self.label_shape[1] / self.image_shape[1]
    h_ratio = self.label_shape[0] / self.image_shape[0]
    heatmaps_batch = self.np_gen_heatmaps_batch(aug_kps_x_batch*w_ratio, aug_kps_y_batch*h_ratio)

    return aug_images_batch, heatmaps_batch

  def tf_create_train_labels(self, images_batch, kps_x_batch, kps_y_batch):
      return tf.numpy_function(self.np_create_train_labels, [images_batch, kps_x_batch, kps_y_batch], (tf.float32, tf.float32))


  def np_create_valid_labels(self, images_batch, kps_x_batch, kps_y_batch):
    ''' For valid just convert keypoints into heatmaps
    Params:
      images_batch: batch of images, expected to be resized to the same shape
      kp_x_batch: batch of x coordinates, scaled to the current label shape (64, 64)
      kp_y_batch: batch of y coordinates, scaled to the current label shape (64, 64)
    Returns:
      Images
      Heatmaps ofkeypoints
    '''

    # Heatmaps 
    heatmaps_batch = self.np_gen_heatmaps_batch(kps_x_batch, kps_y_batch)

    return images_batch, heatmaps_batch

  def tf_create_valid_labels(self, images_batch, kps_x_batch, kps_y_batch):
      return tf.numpy_function(self.np_create_valid_labels, [images_batch, kps_x_batch, kps_y_batch], (tf.float32, tf.float32))


  def prepare_train_example(self, example):
    # Getting all the needed data first
    raw_image = example['image']
    raw_image_height = example['height']
    raw_image_width = example['width']
    kps_x = example['keypoints/x']
    kps_y = example['keypoints/y']

    # Recalculate x and y from original shape to (256, 256)
    h_ratio = self.image_shape[0] / raw_image_height
    w_ratio = self.image_shape[1] / raw_image_width
    kps_x = kps_x * tf.cast(w_ratio, tf.float32) 
    kps_y = kps_y * tf.cast(h_ratio, tf.float32) 

    # Resize image
    image = tf.image.resize(raw_image, (self.image_shape[0], self.image_shape[1]))
  
    return image, kps_x, kps_y
  
  def prepare_valid_example(self, example):
    # Getting all the needed data first
    raw_image = example['image']
    raw_image_height = example['height']
    raw_image_width = example['width']
    kps_x = example['keypoints/x']
    kps_y = example['keypoints/y']

    # Generate heatmaps
    # Recal x, y into heatmaps' shape (64, 64)
    h_ratio = self.label_shape[0] / raw_image_height
    w_ratio = self.label_shape[1] / raw_image_width
    kps_x = kps_x * tf.cast(w_ratio, tf.float32) 
    kps_y = kps_y * tf.cast(h_ratio, tf.float32) 
    
    # Resize imgage to desire size
    image = tf.image.resize(raw_image, (self.image_shape[0], self.image_shape[1]))
    
    return image, kps_x, kps_y
  
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
  
  def np_augment(self, images, kps_x, kps_y):
      '''
      Only work with batch, and with numpy arrays

      Params:
        images: batch of images (batch_size, height, witdth, channels)
        kps_x: batch of x coordinates (batch_size, number of keypoints)
        kps_y: batch of y coordinates (batch_size, number of keypoints)
      Returns:
        augmented images
        augmented keypoints x coordinates
        augmented keypoints y coordinates

      Throws:
        TODO make sure 1st shapes of params are the same
      '''
      
      number_of_examples_in_a_batch = images.shape[0] # not always batch_size, cuase of remainder 

      batch_of_kpsoi = [] 
      batch_of_indices = []
      for i in range(number_of_examples_in_a_batch):
        # one instance of a batch
        xs = kps_x[i]
        ys = kps_y[i]
        image = images[i]

        # to store
        imgaug_kps = []
        indices = []
        for i, kp in enumerate(zip(xs, ys)):
          if 0 < kp[0] < image.shape[1] and 0 < kp[1] < image.shape[0]:
            imgaug_kps.append(Keypoint(x = kp[0], y = kp[1]))
            indices.append(i)
        
        kpsoi = KeypointsOnImage(imgaug_kps, shape = image.shape)
        batch_of_kpsoi.append(kpsoi)
        batch_of_indices.append(indices)

      #augment
      seed = np.random.randint(2**32-1)
      ia.seed(seed)
      seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(scale = (0.75, 1.25), rotate = (-30, 30)),
        ], random_order = True) #cause rotate/scale then flip can be problematic
      
      aug_imgs_batch, aug_kps_batch = seq(images = images, keypoints = batch_of_kpsoi)

      # pad the invisible keypoints
      output_kpts_x_batch = np.zeros(shape = (number_of_examples_in_a_batch, self.num_keypoints) , dtype = np.float32)
      output_kpts_y_batch = np.zeros(shape = (number_of_examples_in_a_batch, self.num_keypoints) , dtype = np.float32)

      for i in range(number_of_examples_in_a_batch):
        arr = aug_kps_batch[i].to_xy_array()
        indices = batch_of_indices[i]
        for j, xy in zip(indices, arr):
            output_kpts_x_batch[i, j] = xy[0]
            output_kpts_y_batch[i, j] = xy[1]

      return aug_imgs_batch, output_kpts_x_batch, output_kpts_y_batch
    
  #def tf_augment(image, kpts_x, kpts_y):
  #   return tf.numpy_function(np_augment, [image, kpts_x, kpts_y], (tf.float32, tf.float32, tf.float32))
 
  def np_gen_heatmaps_batch(self, kps_x_batch, kps_y_batch):
    '''
    Params:
      kps_x_batch: x coordinates batch
      kps_y_batch: y coordinates batch
    Returns:
      A numpy array of a batch of heatmaps
    '''
    heatmaps_batch = []
    for kps_x, kps_y in zip(kps_x_batch, kps_y_batch):
      heatmaps_batch.append(self.np_gen_heatmaps(kps_x, kps_y))

    return np.array(heatmaps_batch, dtype = np.float32)

  def tf_gen_heatmaps_batch(self, kps_x_batch, kps_y_batch):
      return tf.numpy_function(self.np_gen_heatmaps_batch, [kps_x_batch, kps_y_batch], tf.float32)

  def np_gen_heatmaps(self, kpts_x, kpts_y):
    '''
    Creates 2D heatmaps from keypoints coordinates for one single example/image
    Only work with single instnace
    Params:
      kpts_x, kpts_y: x and y coordisnate in shape of (number of kpts)

    Returns: 
      array of heatmaps (heatmap_width, heatmap_height, number of kpts)
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

  # Some utils
  @staticmethod
  def get_ds_length(filenames):
    length = 0
    for name in filenames:
      temp = name.split('-')[-1]
      temp = temp.split('.')[0]
      length += int(temp)
    return length

