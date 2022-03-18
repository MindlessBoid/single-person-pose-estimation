import tensorflow as tf
import matplotlib.pyplot as plt
import os

def transform_bbox_square(bbox, scale = 1):
  '''
  Make bbox into square with the side is the longer side of old bbox
  :param bbox: bbox (x, y, width, height)
  :param scale: scale the bbox
  '''
  x, y, w, h = bbox
  center_x = x + w/2
  center_y = y + h/2

  if w >= h:
    new_w = w
    new_h = w
  else:
    new_w = h
    new_h = h

  new_w *= scale
  new_h *= scale
  new_x = center_x - new_w/2
  new_y = center_y - new_h/2

  return new_x, new_y, new_w, new_h


'''Helpers'''
def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]) #only for jpeg/jpg
    )

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
def int64_feature_list(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_example(image, image_path, example, index, bbox_scale):
  ''' For writing tfrecord example
  :param image: a tensor (height, width, 3)
  :param image_path: path
  :param example: a row in the dataframe
  :param index: index of dataframe, also is a image_id
  :param bbox_scale: to scale bbox, default = 1.25
  TODO: deal with invisible (v = 1) kps that are out of bbox
  '''

  img_height = int(tf.shape(image)[0])
  img_width = int(tf.shape(image)[1])
  
  ## Bbox
  #0: x left top, 1: y left top, 2: width, 3: height
  bbox_x, bbox_y, bbox_w, bbox_h = transform_bbox_square(example['bbox'], scale = bbox_scale) # make squre bbo
  # offset_width, offset_height, target_width, target_hegiht is for padding
  offset_width = 0 
  offset_height = 0
  target_width = bbox_w
  target_height = bbox_h

  #print("bbox:", int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h))
  # check bound for x
  if bbox_x < 0:
    offset_width = -bbox_x
    bbox_w += bbox_x # since crop uses bbox_w, we should account for when bbox < 0
    bbox_x = 0
  if bbox_x + bbox_w > img_width:
    bbox_w = img_width - bbox_x
  #check bound for y
  if bbox_y < 0:
    offset_height = -bbox_y
    bbox_h += bbox_y
    bbox_y = 0
  if bbox_y + bbox_h > img_height:
    bbox_h = img_height - bbox_y
  #print("bbox:", int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h))

  # Crop 
  # since we use bbox to crop the bbox width and height will become out new width and height
  image = tf.image.crop_to_bounding_box(image, int(bbox_y), int(bbox_x), int(bbox_h), int(bbox_w))
  # Pad
  #print("offset:",int(offset_width), int(offset_height), int(target_height), int(target_width))
  image = tf.image.pad_to_bounding_box(image, int(offset_height), int(offset_width), int(target_height), int(target_width))

  ## Parse x and y coords
  kps = example['keypoints']
  #recalculate for cropping 
  # coco format (...xn, yn, vn,...)
  xcoords = [kps[i] for i in range(len(kps)) if i%3 == 0]
  ycoords = [kps[i] for i in range(len(kps)) if i%3 == 1]
  xcoords = [x - bbox_x + offset_width if x > 0  else 0 for x in xcoords]
  ycoords = [y - bbox_y + offset_height if y > 0  else 0 for y in ycoords]

  #visibility flag
  vis = [int(kps[i]) for i in range(len(kps)) if i%3 == 2]
  num = example['num_keypoints']
  
  ## Annotation id, unique for each example
  ann_id = example['ann_id']

  ## Image id
  image_id = index # since we use image id as index for coco_df

  ## Features
  feature = {
        "ann_id": int64_feature(ann_id),
        "image_id": int64_feature(image_id),
        "image": image_feature(image),
        "image_path": bytes_feature(image_path),
        "width": int64_feature(int(target_width)), # since we crop and pad
        "height": int64_feature(int(target_height)), # since we crop and pad
        "keypoints/x": float_feature_list(xcoords),
        "keypoints/y": float_feature_list(ycoords),
        "keypoints/vis": int64_feature_list(vis),
        "keypoints/num": int64_feature(num)
    }
  return tf.train.Example(features=tf.train.Features(feature=feature))

def gen_TFRecords(df, config, is_train: bool):

  num_examples_per_record = config.NUM_EXAMPLER_PER_TFRECORD
  num_tfrecords = len(df) // num_examples_per_record 
  if len(df) % num_examples_per_record:
    num_tfrecords += 1 # add one record if there are any remaining samples

  if is_train:
    output_folder = config.TRAIN_TFRECORDS_DIR
    images_dir = config.TRAIN_IMAGES_DIR
  else:
    output_folder = config.VALID_TFRECORDS_DIR
    images_dir = config.VALID_IMAGES_DIR
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)  # creating TFRecords output folder
  
  for tfrecord in range(num_tfrecords):
    examples = df[(tfrecord * num_examples_per_record) : ((tfrecord + 1) * num_examples_per_record)]
    
    with tf.io.TFRecordWriter(
        output_folder + "/file_" + output_folder.split('/')[-1] + "_%.2i-%i.tfrec" % (tfrecord, len(examples))
    ) as writer:
        for index, row in examples.iterrows():
            image_path = os.path.join(images_dir, row['image_path'])
            image = tf.io.decode_jpeg(tf.io.read_file(image_path))
            tfrecord_example = create_example(image, image_path, row, index, config.BBOX_SCALE)
            writer.write(tfrecord_example.SerializeToString())
  print('TFRecords generated at', output_folder)

if __name__ == '__main__':
  from configs import default_config as cfg
  from coco_df import gen_trainval_df

  train_df, valid_df = gen_trainval_df(cfg)
  gen_TFRecords(train_df, cfg, is_train = True)
  gen_TFRecords(valid_df, cfg, is_train = False)