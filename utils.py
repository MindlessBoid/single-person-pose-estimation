import tensorflow as tf

def generate_2d_guassian(height, width, y0, x0, v0, sigma=1, scale=1):
  '''
  "The same technique as Tompson et al. is used for supervision. A MeanSquared Error (MSE) loss is
  applied comparing the predicted heatmap to a ground-truth heatmap consisting of a 2D gaussian
  (with standard deviation of 1 px) centered on the keypoint location."
  https://github.com/princeton-vl/pose-hg-train/blob/master/src/util/img.lua#L204

  Credit to: https://github.com/ethanyanjiali/deep-vision/blob/master/Hourglass/tensorflow/preprocess.py
  Work with Tensor

  '''
  heatmap = tf.zeros((height, width))

  # this gaussian patch is 7x7, let's get four corners of it first
  xmin = x0 - 3 * sigma
  ymin = y0 - 3 * sigma
  xmax = x0 + 3 * sigma
  ymax = y0 + 3 * sigma
  # if the patch is out of image boundary we simply return nothing according to the source code
  # [1]"In these cases the joint is either truncated or severely occluded, so for
  # supervision a ground truth heatmap of all zeros is provided."
  # flag 0 is not included (x=0, y=0)
  if xmin >= width or ymin >= height or xmax < 0 or ymax <0 or v0 == 0:
      return heatmap

  size = 6 * sigma + 1
  x, y = tf.meshgrid(tf.range(0, 6*sigma+1, 1), tf.range(0, 6*sigma+1, 1), indexing='xy')

  # the center of the gaussian patch should be 1
  center_x = size // 2
  center_y = size // 2

  # generate this 7x7 gaussian patch
  gaussian_patch = tf.cast(tf.math.exp(-(tf.square(x - center_x) + tf.math.square(y - center_y)) / (tf.math.square(sigma) * 2)) * scale, dtype=tf.float32)

  # part of the patch could be out of the boundary, so we need to determine the valid range
  # if xmin = -2, it means the 2 left-most columns are invalid, which is max(0, -(-2)) = 2
  patch_xmin = tf.math.maximum(0, -xmin)
  patch_ymin = tf.math.maximum(0, -ymin)
  # if xmin = 59, xmax = 66, but our output is 64x64, then we should discard 2 right-most columns
  # which is min(64, 66) - 59 = 5, and column 6 and 7 are discarded
  patch_xmax = tf.math.minimum(xmax, width) - xmin
  patch_ymax = tf.math.minimum(ymax, height) - ymin

  # also, we need to determine where to put this patch in the whole heatmap
  heatmap_xmin = tf.math.maximum(0, xmin)
  heatmap_ymin = tf.math.maximum(0, ymin)
  heatmap_xmax = tf.math.minimum(xmax, width)
  heatmap_ymax = tf.math.minimum(ymax, height)

  # finally, insert this patch into the heatmap
  indices = tf.TensorArray(tf.int32, 1, dynamic_size=True)
  updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)

  count = 0

  for j in tf.range(patch_ymin, patch_ymax):
      for i in tf.range(patch_xmin, patch_xmax):
          indices = indices.write(count, [heatmap_ymin+j, heatmap_xmin+i])
          updates = updates.write(count, gaussian_patch[j][i])
          count += 1
          
  heatmap = tf.tensor_scatter_nd_update(heatmap, indices.stack(), updates.stack())

  return heatmap