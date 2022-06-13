import tensorflow as tf
import numpy as np

def compile_model_from_checkpoint(model, ckpt_path, optimizer, loss):
  ''' 
    Usages:
      This function to load model only so optimizer and loss dont really matter

    Params:
      model: tf model
      ckpt_path: should be anything before and '.ckpt'
      optimizer: 
      loss: applied for all outputs

    Returns:
      A compiled tensorflow model
  '''
  model.load_weights(ckpt_path)
  model.compile(optimizer = optimizer, loss = loss)
  return model

# This func is unmodified and ripped from: https://github.com/princeton-vl/pose-hg-train/blob/master/src/pypose/draw.py
def gaussian(img, pt, sigma):
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