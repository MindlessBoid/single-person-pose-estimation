import tensorflow as tf
from pycocotools.coco import COCO
import pandas as pd
import numpy as np

def get_meta(coco):
  # Get all images identifier, the length should be the number of images
  ids = list(coco.imgs.keys())
  for i, img_id in enumerate(ids):
    img_meta = coco.imgs[img_id]
    ann_ids = coco.getAnnIds(imgIds = img_id)
    # retrieve meta data for all persons in the current image
    anns = coco.loadAnns(ann_ids)
    # basic para of an image
    img_file_name = img_meta['file_name']
    w = img_meta['width']
    h = img_meta['height']
    url = img_meta['coco_url']
    

    yield [img_id, img_file_name, w, h, url, anns]

def convert_to_df(coco):
  images_data = []
  persons_data = []

  # iterate over all images
  for img_id, img_file_name, w, h, url, meta in get_meta(coco):
    images_data.append({
        'image_id': int(img_id),
        'coco_url': url,
        'image_path': img_file_name,
        'width': int(w),
        'height': int(h)
        })
    # iterate over all metadata
    for m in meta:
      persons_data.append({
          'ann_id': m['id'], #each example will have a unique id
          'image_id': m['image_id'],
          'is_crowd': m['iscrowd'],
          'bbox': m['bbox'],
          'num_keypoints': m['num_keypoints'],
          'keypoints': m['keypoints'],
      })
  # create dataframes
  images_df = pd.DataFrame(images_data)
  images_df.set_index('image_id', inplace = True) # set imgae_id as 1st row

  persons_df = pd.DataFrame(persons_data)
  persons_df.set_index('image_id', inplace = True)

  return images_df, persons_df

def gen_trainval_df(config, drop_min_num_kps: bool = False):

  min_num_kps = 0
  if drop_min_num_kps:
    min_num_kps = config.MIN_NUM_KEYPOINTS
  #train
  train_coco = COCO(config.TRAIN_ANNOT_FILE) 
  train_images_df, train_persons_df = convert_to_df(train_coco)
  train_coco_df = pd.merge(train_images_df, train_persons_df, right_index=True, left_index=True)
  train_coco_df = train_coco_df[(train_coco_df['is_crowd'] == 0) & (train_coco_df['num_keypoints'] > min_num_kps)]

  #valid
  valid_coco = COCO(config.VALID_ANNOT_FILE)
  valid_images_df, valid_persons_df = convert_to_df(valid_coco)
  valid_coco_df = pd.merge(valid_images_df, valid_persons_df, right_index=True, left_index=True)
  valid_coco_df = valid_coco_df[(valid_coco_df['is_crowd'] == 0) & (valid_coco_df['num_keypoints'] >  min_num_kps)]

  print(f"Only examples that are not crowd and num_keypoints > {min_num_kps} are chosen !")
  print(f"Length of train df: {len(train_coco_df)}")
  print(f"Length of valid df: {len(valid_coco_df)}")
  return train_coco_df, valid_coco_df

