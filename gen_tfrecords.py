import tensorflow as tf
import matplotlib.pyplot as plt
import os
import argparse
import time
from datetime import datetime, timedelta
from utilities.data_utils import transform_bbox_square, crop_and_pad
from configs import default_config as cfg
from coco_df import gen_trainval_df


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
  bbox = transform_bbox_square(example['bbox'], scale = bbox_scale) # make square bbo
  # offset_width, offset_height, target_width, target_hegiht is for padding
  adjusted_image = crop_and_pad(image, bbox)

  ## Parse x and y coords
  kps = example['keypoints']
  #recalculate for cropping 
  # coco format (...xn, yn, vn,...)
  xs = [kps[i] for i in range(len(kps)) if i%3 == 0]
  ys = [kps[i] for i in range(len(kps)) if i%3 == 1]
  vs = [int(kps[i]) for i in range(len(kps)) if i%3 == 2]

  # filter xs, ys, vis: only take ones inside the box and vis > 0
  # some keypoints still can be outside of the adjusted box -> set vis flag = 0
  filtered_xs = []
  filtered_ys = []
  filtered_vs = []
  for x, y, v in zip(xs, ys, vs):
    x = x - bbox[0]
    y = y - bbox[1]
    if 0 < x < bbox[2] and 0 < y < bbox[3] and v > 0:
      filtered_xs.append(x)
      filtered_ys.append(y)
      filtered_vs.append(v)
    else:
      filtered_xs.append(0)
      filtered_ys.append(0)
      filtered_vs.append(0)


  # Number of keypoints
  kps_vis = list(map(lambda v: v>0, filtered_vs))
  num_kps = sum(kps_vis)

  ## Annotation id, unique for each example
  ann_id = example['ann_id']

  ## Image id
  image_id = index # since we use image id as index for coco_df

  ## COCO url
  coco_url = example['coco_url']

  ## Features
  feature = {
        "ann_id": int64_feature(ann_id),
        "image_id": int64_feature(image_id),
        "image": image_feature(adjusted_image),
        "image_path": bytes_feature(image_path),
        "coco_url": bytes_feature(coco_url),
        "width": int64_feature(int(tf.shape(adjusted_image)[1])), # since we crop and pad
        "height": int64_feature(int(tf.shape(adjusted_image)[0])), # since we crop and pad
        "keypoints/x": float_feature_list(filtered_xs),
        "keypoints/y": float_feature_list(filtered_ys),
        "keypoints/vis": int64_feature_list(filtered_vs),
        "keypoints/num": int64_feature(num_kps),
        "bbox_x": float_feature(bbox[0]),
        "bbox_y": float_feature(bbox[1]),
        "original_bbox": float_feature_list(example["bbox"])
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

################################## HELPERS ##############################################################
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

if __name__ == '__main__':
  train_df, valid_df = gen_trainval_df(cfg, drop_min_num_kps=True)

  print('Generating train TFRecord files:')
  start = time.time()
  gen_TFRecords(train_df, cfg, is_train = True)
  total_time = time.time() - start
  print("Total time: {}".format(str(timedelta(seconds=total_time))))

  print('Generating valid TFRecord files:')
  start = time.time()
  gen_TFRecords(train_df, cfg, is_train = False)
  total_time = time.time() - start
  print("Total time: {}".format(str(timedelta(seconds=total_time))))


  
  
  

  









