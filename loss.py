import tensorflow as tf
def weighted_mean_squared_error(y_true, y_pred):
  ''' Both inputs should has the same shape (batch_size, d0, d1, .. dN)
  Args:
    y_true: labels
    y_pred: predictions
  Returns:
    A tensor with shape (batch_size, d0, d1, .. dN-1) -> reduced last axis
    It should NOT return a scalar
  Raises:

  '''
  weights = tf.cast(y_true > 0, dtype = tf.float32)*81 + 1
  
  return tf.reduce_mean(tf.math.square(y_true - y_pred) * weights, axis = -1)