import matplotlib.pyplot as plt
from configs import default_config as cfg


def draw_keypoints(image, keypoints):
  '''
  Params:
    image: (h, w, 3)
    keypoints: (17, 3), [x, y, conf]
  '''
  xs = (keypoints[:, 0]/cfg.LABEL_WIDTH)*image.shape[1]
  ys = (keypoints[:, 1]/cfg.LABEL_HEIGHT)*image.shape[0]
  plt.figure(figsize=(12, 12))
  plt.imshow(image)
  plt.scatter(xs, ys, marker='o', color='r')
  plt.axis("off")

def draw_skeleton(image, keypoints):
  '''
  Params:
    image: (h, w, 3)
    keypoints: (17, 3), [x, y, conf]
  '''
  xs = (keypoints[:, 0]/cfg.LABEL_WIDTH)*image.shape[1]
  ys = (keypoints[:, 1]/cfg.LABEL_HEIGHT)*image.shape[0]

  plt.figure(figsize=(12, 12))
  plt.imshow(image)
  # extracting pairs
  xs1, ys1 = xs[cfg.COCO_SKELETON[:,0]], ys[cfg.COCO_SKELETON[:,0]]
  xs2, ys2 = xs[cfg.COCO_SKELETON[:,1]], ys[cfg.COCO_SKELETON[:,1]]

  # get rid of zeros
  for x1, y1, x2, y2 in zip(xs1, ys1, xs2, ys2):
    if x1 and y1 and x2 and y2:
      plt.plot((x1, x2), (y1, y2), marker='o', linewidth=5, markersize=12)
  plt.axis('off')