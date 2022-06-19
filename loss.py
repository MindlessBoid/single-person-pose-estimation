import tensorflow as tf
def weighted_mse(y_true, y_pred):
  ''' Both inputs should has the same shape (batch_size, d0, d1, .. dN)
  Args:
    y_true: labels
    y_pred: predictions
  Returns:
    A tensor with shape (batch_size, d0, d1, .. dN-1) -> reduced last axis
    It should NOT return a scalar
  Raises:

  '''
  # weights has the same shape y_true and y_pred
  weights = tf.cast(y_true > 0, dtype = tf.float32)*81 + 1
  
  # check if the keypoint is valid, heatmap_sum has the shape of (batch_size, 1, 1, number of keypoint)
  #heatmap_sum = tf.math.reduce_sum(y_true, axis= [1, 2], keepdims=True)
  # valid keypoint = 1.0, invalid = 0.0
  #keypoint_weights = 1.0 - tf.cast(tf.equal(heatmap_sum, 0.0), tf.float32)

  return tf.reduce_mean(tf.math.square(y_true - y_pred) * weights, axis = -1)

def IOU(y_true, y_pred):
  epsilon = 1e-6
  inter = tf.reduce_sum(y_true*y_pred, axis = [1, 2])
  union = tf.reduce_sum(y_true*y_true, axis = [1, 2]) + tf.reduce_sum(y_pred*y_pred, axis = [1, 2]) - inter
  IoU = (inter + epsilon) / (union + epsilon)
  return 1 - tf.reduce_mean(IoU, axis = -1)

def weighed_keypoint_mse(y_true, y_pred):
  # check if the keypoint is valid, heatmap_sum has the shape of (batch_size, 1, 1, number of keypoint)
  heatmap_sum = tf.math.reduce_sum(y_true, axis= [1, 2], keepdims=True)
  # valid keypoint = 1.0, invalid = 0.0
  keypoint_weights = 1.0 - tf.cast(tf.equal(heatmap_sum, 0.0), tf.float32)

  return tf.reduce_mean(tf.math.square(y_true - y_pred) * keypoint_weights, axis = -1)
  