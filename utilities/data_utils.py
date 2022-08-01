import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_cv_image(image_file, show=False):
  '''return an RGB cv image'''
  image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if show:
    plt.figure(figsize=(12,12))
    plt.imshow(image)
  return image


def transform_bbox_square(bbox, scale = 1):
  '''
  Usages:
    Transform a bounding box into a square bounding with side = old longer side
  
  Params:
    bbox: (x, y, widht, height)
    scale: scale of bbox, 1 doesnt affect

  Returns:
    new_bbox

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


def crop_and_pad(image, square_bbox):
  '''
  Usages:
    Crop and pad to bbox
  
  Params:
    image: image tensor
    square_bbox: (x, y, width, height)
  
  Returns:
    adjusted image
  '''

  ## Image width and height
  image_height = int(tf.shape(image)[0])
  image_width = int(tf.shape(image)[1])

  x, y, w, h = square_bbox
  xmin, ymin = x, y
  xmax, ymax = x + w, y + h

  ## Pad first
  offset_width = 0 # number of rows to add to the right
  offset_height = 0 # number of cols to add to the left
  target_width = image_width
  target_height = image_height

  # Boder cases
  if xmin < 0:
    offset_width = int(abs(x))
    target_width += offset_width

  if ymin < 0:
    offset_height = int(abs(y))
    target_height += offset_height

  if xmax > image_width:
    target_width += int(xmax - image_width) + 1  # bit offset for cropping

  if ymax > image_height:
    target_height += int(ymax - image_height) + 1 # bit offset for cropping

  
  adjusted_image = tf.image.pad_to_bounding_box(image, offset_height, offset_width, 
                                              target_height, target_width)
  
  ## Then crop to bbox
  adjusted_image = tf.image.crop_to_bounding_box(adjusted_image, int(max(ymin, 0)), int(max(xmin, 0)),
                                                                 int(h), int(w))
  
  return adjusted_image

def heatmaps_to_keypoints_v1(heatmaps, conf_threshold = 1e-6):
  '''
  Usages:
    Turn heatmaps to coordinates
  
  Params:
    heatamps: np.array, shape (w, h, num_kps)
    conf_threhold: confidence threshold
  
  Returns:
    keypoints: xs, ys, confidence score

  '''
  height = heatmaps.shape[0]
  width = heatmaps.shape[1]
  num_kps = heatmaps.shape[2]

  keypoints = np.zeros(shape=(num_kps, 3), dtype=np.float32)
  for i in range(num_kps):
    heatmap = heatmaps[:,:,i]
    index = np.argmax(heatmap) # return index when flattened
    x = index % width
    y = index // height
    conf = heatmap[y, x]
    if conf > conf_threshold:
      keypoints[i, 0] = x
      keypoints[i, 1] = y
      keypoints[i, 2] = conf
    else:
      keypoints[i, 0] = 0
      keypoints[i, 1] = 0
      keypoints[i, 2] = 0
  return keypoints


def heatmaps_to_keypoints_v2(heatmaps, conf_threshold = 1e-6):
  '''
  Usages:
    Turn heatmaps to coordinates + 0.25 distance from the highest and 2nd highest in a 3x3 patch
  
  Params:
    heatamps: np.array, shape (w, h, num_kps)
    conf_threhold: confidence threshold
  
  Returns:
    keypoints: np.array([x, y, confidence score])

  '''
  height = heatmaps.shape[0]
  width = heatmaps.shape[1]
  num_kps = heatmaps.shape[2]

  keypoints = np.zeros(shape=(num_kps, 3), dtype=np.float32)
  for i in range(num_kps):
    heatmap = heatmaps[:,:,i]
    index = np.argmax(heatmap) # return index when flattened
    x = index % width
    y = index // height
    conf = heatmap[y, x]

    # 3x3 patch
    x1 = max(x-1, 0)
    x2 = min(x+2, width)
    y1 = max(y-1, 0)
    y2 = min(y+2, height)
    patch = heatmap[y1:y2, x1:x2]
    patch[1][1] = 0
    patch_index = np.argmax(patch)
    patch_x = patch_index%3
    patch_y = patch_index//3

    delta_x = patch_x/4
    delta_y = patch_y/4


    if conf > conf_threshold:
      keypoints[i, 0] = x + delta_x
      keypoints[i, 1] = y + delta_y
      keypoints[i, 2] = conf
    else:
      keypoints[i, 0] = 0
      keypoints[i, 1] = 0
      keypoints[i, 2] = 0
  return keypoints


# This func is unmodified and ripped from: https://github.com/princeton-vl/pose-hg-train/blob/master/src/pypose/draw.py
def gaussian(img, pt, sigma = 1):
   # Draw a 2D gaussian
   # Check that any part of the gaussian is in-bounds
   ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
   br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
   if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
           br[0] < 0 or br[1] < 0):
       # If not, just return the image as is
       return img
   # Generate gaussian
   size = 6 * sigma + 1
   x = np.arange(0, size, 1, float)
   y = x[:, np.newaxis]
   x0 = y0 = size // 2
   # The gaussian is not normalized, we want the center value to equal 1
   g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
   # Usable gaussian range
   g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
   g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
   # Image range
   img_x = max(0, ul[0]), min(br[0], img.shape[1])
   img_y = max(0, ul[1]), min(br[1], img.shape[0])
   img[img_y[0]:img_y[1], img_x[0]:img_x[1]
       ] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
   return img