import tensorflow as tf
import numpy as np
import glob

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

def get_epochs_from_ckpt_path(path):
  '''
    format of checkpoint file path/before/that/E{self.epochs}_{today}_cont.ckpt.index
  '''
  names = glob.glob(path + '/*_cont.ckpt.index')
  names.sort()

  ckpt_names = []
  epochs = []
  for name in names:
    
    ckpt_name = name[:-6] # eliminate '.index', to load using keras
    ckpt_names.append(ckpt_name)

    n = name.split('/')[-1] # get rid of slashes
    e = n.split('_')[0] # get E{epochs}
    e = int(e[1:]) # get rid of 'E'
    epochs.append(e)

  ckpt_names.append(path + '/best_val_loss_weights.ckpt')
  epochs.append(-1)
  return ckpt_names, epochs
