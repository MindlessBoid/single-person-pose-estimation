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