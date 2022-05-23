import tensorflow as tf
def weighted_mean_squared_error(truth, pred):
  weights = tf.cast(truth > 0, dtype = tf.float32)*81 + 1
  return tf.reduce_mean(tf.math.square(truth - pred) * weights)