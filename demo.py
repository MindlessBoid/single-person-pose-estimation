import tensorflow as tf
from utilities.visualization_utils import *
from utilities.data_utils import *
from configs import default_config as cfg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import io


class Demo:
  def __init__(self, person_detector, keypoints_detetor, cfg, max_num_ppl=6, person_conf_thres=1e-6, keypoints_conf_thres=1e-6):
    '''
      Detect multiple people in one image
    '''

    self.person_detector = person_detector
    self.keypoints_detetor = keypoints_detetor
    self.person_conf_thres = person_conf_thres
    self.keypoints_conf_thres = keypoints_conf_thres
    self.COCO_SKELETON = cfg.COCO_SKELETON
    self.max_num_ppl = max_num_ppl

  def detect(self, image):
    ## Read the image, yolov5 does not accept tf Tensor, image must be RGB cv image
    ## Stage 1: person_detector
    # Predict
    stage1_result = self.person_detector(image)
    # Convert to pandas DataFrame, only take person category
    df = stage1_result.pandas().xyxy[0]
    human_df = person_df = df[(df['name']=='person') & (df['confidence'] >self.person_conf_thres)]
    # Get bounding boxes
    bboxes = []
    xmins = human_df['xmin'].values
    ymins = human_df['ymin'].values
    xmaxs = human_df['xmax'].values
    ymaxs = human_df['ymax'].values
    for xmin, ymin, xmax, ymax in zip(xmins, ymins, xmaxs, ymaxs):
      bboxes.append((xmin, ymin, xmax-xmin, ymax-ymin))
    # Restrict number of ppl in one picture
    bboxes = bboxes[:self.max_num_ppl]

    ## Stage 2: keypoints_detector
    # Crop to bboxes
    tf_image = tf.convert_to_tensor(image)
    tf_image = tf.image.convert_image_dtype(tf_image, dtype=tf.float32) # safe conversion
    cropped_images = []
    transformed_bboxes = []
    for bbox in bboxes:
      new_bbox = transform_bbox_square(bbox, cfg.BBOX_SCALE)
      new_image = crop_and_pad(tf_image, new_bbox)
      new_image = tf.image.resize(new_image, (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH))
      transformed_bboxes.append(new_bbox)
      cropped_images.append(new_image)
    # Predict
    keypoints_list = []
    if cropped_images:
      pred = self.keypoints_detetor.predict(tf.convert_to_tensor(cropped_images)) # return a list
      for hms in pred[-1]:
        kps = heatmaps_to_keypoints_v2(hms, self.keypoints_conf_thres)
        # Noremalize
        kps[:, 0] /= cfg.LABEL_WIDTH
        kps[:, 1] /= cfg.LABEL_HEIGHT
        keypoints_list.append(kps)
      
    self.image = image
    self.cropped_images = cropped_images
    self.original_bboxes = bboxes
    self.square_bboxes = transformed_bboxes
    self.keypoints_list = keypoints_list

  def show(self, figsize = (12, 12), show_bboxes=False, save=False):
    '''
      Show the orginal image with all predictions
      show_bboxes only show the bbox of person_dectector not transform ones
    '''
    fig = plt.figure(figsize=(figsize))
    plt.imshow(self.image)
    ax = plt.gca()
    for keypoints, bbox, old_bbox in zip(self.keypoints_list, self.square_bboxes, self.original_bboxes):
      xs = keypoints[:, 0]
      ys = keypoints[:, 1]
      # extracting pairs
      xs1, ys1 = xs[self.COCO_SKELETON[:,0]], ys[self.COCO_SKELETON[:,0]]
      xs2, ys2 = xs[self.COCO_SKELETON[:,1]], ys[self.COCO_SKELETON[:,1]]
      for x1, y1, x2, y2 in zip(xs1, ys1, xs2, ys2):
        if x1 and y1 and x2 and y2:
          plt.plot((x1*bbox[2] + bbox[0], x2*bbox[2] + bbox[0]), (y1*bbox[3] + bbox[1], y2*bbox[3] + bbox[1]), 
                    marker='o', linewidth=5, markersize=7)

      # Draw bboxes   
      if show_bboxes:
        rect = patches.Rectangle((old_bbox[0], old_bbox[1]), old_bbox[2], old_bbox[3], linewidth=3, 
                                  edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()
    if save:
      plt.savefig('saved_file.png', bbox_inches='tight', pad_inches=0)

  def show_bboxes(self, figsize = (12, 12), show_square_bboxes=False):
    plt.figure(figsize=(figsize))
    plt.imshow(self.image)
    ax = plt.gca()
    bboxes = self.square_bboxes if show_square_bboxes else self.original_bboxes
    for bbox in bboxes:
      rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=3, 
                                  edgecolor='r', facecolor='none')
      ax.add_patch(rect)
    plt.axis("off")
    plt.show()

  def show_separate(self, figsize = (12, 12), max_per_row=3):
    '''show predictions on cropped images'''
    num_rows = int(np.ceil(len(self.cropped_images)/max_per_row))
    num_cols = max_per_row
    fig = plt.figure(figsize=figsize)
    for i, item in enumerate(zip(self.cropped_images, self.keypoints_list)):
      img, keypoints = item
      fig.add_subplot(num_rows, num_cols, i+1)
      plt.imshow(img)
      xs = keypoints[:, 0]*img.shape[1]
      ys = keypoints[:, 1]*img.shape[0]
      # extracting pairs
      xs1, ys1 = xs[self.COCO_SKELETON[:,0]], ys[self.COCO_SKELETON[:,0]]
      xs2, ys2 = xs[self.COCO_SKELETON[:,1]], ys[self.COCO_SKELETON[:,1]]
      for x1, y1, x2, y2 in zip(xs1, ys1, xs2, ys2):
        if x1 and y1 and x2 and y2:
          plt.plot((x1, x2), (y1, y2), 
                    marker='o', linewidth=5, markersize=7)
      plt.tight_layout()
      plt.title(f'Image: {i+1}')
    plt.show()
  
  def create_overlay(self, figsize = (12, 12), show_bboxes=False):
    fig = plt.figure(figsize=(figsize))
    blank = np.zeros((self.image.shape), dtype=float)
    plt.imshow(blank)
    ax = plt.gca()
    for keypoints, bbox, old_bbox in zip(self.keypoints_list, self.square_bboxes, self.original_bboxes):
      xs = keypoints[:, 0]
      ys = keypoints[:, 1]
      # extracting pairs
      xs1, ys1 = xs[self.COCO_SKELETON[:,0]], ys[self.COCO_SKELETON[:,0]]
      xs2, ys2 = xs[self.COCO_SKELETON[:,1]], ys[self.COCO_SKELETON[:,1]]
      for x1, y1, x2, y2 in zip(xs1, ys1, xs2, ys2):
        if x1 and y1 and x2 and y2:
          plt.plot((x1*bbox[2] + bbox[0], x2*bbox[2] + bbox[0]), (y1*bbox[3] + bbox[1], y2*bbox[3] + bbox[1]), 
                    marker='o', linewidth=5, markersize=7)

      # Draw bboxes   
      if show_bboxes:
        rect = patches.Rectangle((old_bbox[0], old_bbox[1]), old_bbox[2], old_bbox[3], linewidth=3, 
                                  edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    plt.close()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1) # cv2 BGR image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (self.image.shape[1], self.image.shape[0])) # cv2 image is (w, h, channels)
    alpha = np.sum(img, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    overlay = np.dstack((img, alpha))
    return overlay