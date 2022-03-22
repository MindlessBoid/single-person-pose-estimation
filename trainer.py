from datetime import date
import tensorflow as tf
import math
import glob
import os
from callbacks import *

class Trainer:
  ''' TODO:
  Add a test function
  
  '''
  def __init__(self, model, ds_builder, epochs, learning_rate, config):
    self.model = model
    self.ds_train, self.ds_valid = ds_builder.build_datasets()

    self.steps_per_epoch = math.ceil(ds_builder.num_train_examples // config.BATCH_SIZE)
    self.valid_steps = math.ceil(ds_builder.num_valid_examples // config.BATCH_SIZE)
    self.epochs = epochs 
    self.checkpoints_path = config.CHECKPOINTS_PATH
    
    self.learning_rate = learning_rate
    self.batch_size = config.BATCH_SIZE
    self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
    self.loss = tf.keras.losses.MeanSquaredError()
  
  def train(self):
    self.model.compile(optimizer = self.optimizer,
            loss = self.loss)

    today = date.today().strftime("%d-%m-%Y")
    path = self.checkpoints_path + f'/{today}-E{self.epochs}' + '.cpkt'
    ckpt_callback = make_checkpoint_callback(path)
    callbacks = [ckpt_callback, PrintLR()]

    print(f'''Start traing with:
    1. Current date {today}.
    2. Number of epochs {self.epochs}.
    3. Learning rate {self.learning_rate}.
    4. Batch size {self.batch_size}.
    ''')
   
    self.model.fit(
      self.ds_train, 
      epochs = self.epochs,
      callbacks = callbacks,
      steps_per_epoch = self.steps_per_epoch,
      validation_data = self.ds_valid,
      validation_steps = self.valid_steps,
    )

    print(f'''Finished training!!
    Temporary checkpoint is saved at {self.checkpoints_path}
    To save model call save_model() method
    ''')
  
  def resume_training(self):
    '''
    This shoudl be called on a newly created instance
    '''
    assert os.path.exists(self.checkpoints_path)

    cpkt_name, previous_epochs = self.get_epochs_from_name(self.checkpoints_path)
    self.epochs += previous_epochs

    self.model.compile(optimizer = self.optimizer,
            loss = self.loss)
    
    print(f'Loading weights from {self.checkpoints_path}')
    self.model.load_weights(self.checkpoints_path + '/' + cpkt_name)

    today = date.today().strftime("%d-%m-%Y")
    new_path = self.checkpoints_path + f'/{today}-E{self.epochs}' + '.cpkt'
    ckpt_callback = make_checkpoint_callback(new_path)
    callbacks = [ckpt_callback, PrintLR()]

    print(f'''Start traing with:
    1. Current date {today}.
    2. Resume training for {self.epochs - previous_epochs}, from epoch {previous_epochs} to epoch {self.epochs}.
    3. Learning rate {self.learning_rate}.
    4. batch size {self.batch_size}.
    ''')
   
    self.model.fit(
      self.ds_train, 
      epochs = self.epochs,
      callbacks = callbacks,
      steps_per_epoch = self.steps_per_epoch,
      validation_data = self.ds_valid,
      validation_steps = self.valid_steps,
      initial_epoch = previous_epochs
    )

    print(f'''Finished training!!
    Temporary checkpoint is saved at {new_path}
    To save model call save_model() method
    ''')
  
  def save_model(self, path):
    self.model.save(path)


      
  @staticmethod
  def get_epochs_from_name(path):
    ''' Extract the checkpoint name and the epochs number which were saved in previous trains
        ONLY get the latest checkpoint 
    '''
    name = glob.glob(path + '/*.cpkt.index')
    name.sort()

    last = name[-1] #last in the list 
    last = last.split('/')[-1] # get rid of slashes
    ckpt_name = last[:-6] # eliminate '.index'
    
    epochs = ckpt_name[:-5] # eliminate '.cpkt'
    epochs = epochs.split('-')[-1] # get E{epcoch}
    epochs = int(epochs[1:]) # get rid of 'E'
    
    return ckpt_name, epochs