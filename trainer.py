from datetime import date
import tensorflow as tf
import math
import glob
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import pandas as pd
from callbacks import *
from keras import backend as K
from loss import weighted_mean_squared_error

class Trainer:
  ''' TODO:
  Add a test function
  
  '''
  def __init__(self, model, ds_builder, epochs, learning_rate, loss_str, config): 
    self.model = model
    self.ds_train, self.ds_valid = ds_builder.build_datasets()

    self.steps_per_epoch = math.ceil(ds_builder.num_train_examples // config.BATCH_SIZE)
    self.valid_steps = math.ceil(ds_builder.num_valid_examples // config.BATCH_SIZE)
    self.epochs = epochs 
    self.checkpoints_path = config.CHECKPOINTS_PATH
    self.logs_path = config.LOGS_PATH

    self.learning_rate = learning_rate
    self.batch_size = config.BATCH_SIZE
    self.optimizer =  tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
    self.loss = self.get_loss_from_string(loss_str)
  
  def train(self):
    self.model.compile(optimizer = self.optimizer, loss = self.loss)

    today = date.today().strftime("%d-%m-%Y")

    ckpt_callback = make_checkpoint_callback(self.checkpoints_path + '/best_val_loss_weights.ckpt')
    callbacks = [ckpt_callback, PrintLR()]

    print(f'''First training with:
    1. Current date {today}.
    2. Number of epochs {self.epochs}.
    3. Batch size {self.batch_size}.
    4. Optimizer configs: {self.model.optimizer.get_config()}
    ''')
    start = time.time()
    H = self.model.fit(
      self.ds_train, 
      epochs = self.epochs,
      callbacks = callbacks,
      steps_per_epoch = self.steps_per_epoch,
      validation_data = self.ds_valid,
      validation_steps = self.valid_steps,
    )
    end = time.time()
    if not os.path.exists(self.logs_path):
      os.makedirs(self.logs_path)
    pd.DataFrame(H.history).to_csv(self.logs_path + f"/log_{today}_E{self.epochs}_lr{self.learning_rate}.csv")

    # temporary save
    path = self.checkpoints_path + f'/{today}_E{self.epochs}_cont' + '.ckpt'
    self.model.save_weights(path)
    
    print('---------------------------------------------------------')
    print(f'''Finished training!!
    - Total training time {str(timedelta(seconds= end - start))}
    - Temporary checkpoints are saved at {self.checkpoints_path}
    - Log is save at {self.logs_path}
    ''')
  
  def resume_training(self):
    '''
    This should be called on a newly created instance
    '''
    assert os.path.exists(self.checkpoints_path) and os.path.exists(self.logs_path)

    # already check empty
    ckpt_name, previous_epochs, full_name = self.get_epochs_from_name(self.checkpoints_path)
    self.epochs += previous_epochs

    # Load and compile model from last training
    print(f'Loading weights from epoch {previous_epochs}')
    self.model.load_weights(self.checkpoints_path + '/' + ckpt_name)
    print(f'Loaded: {full_name}')
    self.model.compile(optimizer = self.optimizer, loss = self.loss)

    # Set learning rate
    K.set_value(self.model.optimizer.learning_rate, self.learning_rate) # seems like only way that works

    # Checkpoint to save best weights in current training session
    today = date.today().strftime("%d-%m-%Y")
    ckpt_callback = make_checkpoint_callback(self.checkpoints_path + '/temp.ckpt')
    callbacks = [ckpt_callback, PrintLR()]

    # Get data from logs 
    log_filenames = sorted(glob.glob(self.logs_path + '/*'))
    df = pd.concat(map(pd.read_csv, log_filenames), ignore_index=True)
    # last train
    print('---------------------------------------------------------')
    last_train = df.iloc[-1:] # get last row as a dataframe object
    print(f'- Result from last train session number {len(log_filenames)} at epoch {previous_epochs}:')
    for col_name_1, col_value_1 in last_train.iteritems():
      if col_name_1 != 'Unnamed: 0':
        print(f'{col_name_1}: {col_value_1.values[0]}')
    # best epoch so far
    print('---------------------------------------------------------')
    min_val_loss = df[df['val_loss'] == df['val_loss'].min()]
    print(f'- Best current val_loss at epoch {min_val_loss.index.values[0]+ 1}:')
    for col_name_2, col_value_2 in min_val_loss.iteritems():
      if col_name_2 != 'Unnamed: 0':
        print(f'{col_name_2}: {col_value_2.values[0]}')
    print('---------------------------------------------------------')
    
    print(f'''Resume training with:
    1. Train session number {len(log_filenames) + 1}.
    2. Current date {today}.
    3. Resume training for {self.epochs - previous_epochs} epochs, from epoch {previous_epochs} to epoch {self.epochs}.
    4. Batch size {self.batch_size}.
    5. Optimizer configs: {self.model.optimizer.get_config()}
    ''')
    # Train and save after training
    start = time.time()
    H = self.model.fit(
      self.ds_train, 
      epochs = self.epochs,
      callbacks = callbacks,
      steps_per_epoch = self.steps_per_epoch,
      validation_data = self.ds_valid,
      validation_steps = self.valid_steps,
      initial_epoch = previous_epochs
    )
    end = time.time()
    if not os.path.exists(self.logs_path):
      os.makedirs(self.logs_path)
    pd.DataFrame(H.history).to_csv(self.logs_path + f"/log_{today}_E{self.epochs}_lr{self.learning_rate}.csv")
    
    # Save last
    path = self.checkpoints_path + f'/{today}_E{self.epochs}_cont' + '.ckpt'
    self.model.save_weights(path)

    # Compare the best weights and save the better one
    print()
    print('---------------------------------------------------------')
    print('Comparing current best val_loss with previous best val_loss checkpoints')
    prev_min_val_loss = min_val_loss['val_loss'].values[0]
    curr_min_val_loss = min(H.history['val_loss'])

    best_data = self.checkpoints_path + '/best_val_loss_weights.ckpt.data-00000-of-00001'
    best_index = self.checkpoints_path + '/best_val_loss_weights.ckpt.index'
    temp_data = self.checkpoints_path + '/temp.ckpt.data-00000-of-00001'
    temp_index = self.checkpoints_path + '/temp.ckpt.index'

    if curr_min_val_loss < prev_min_val_loss:
      print('Current best val_loss is lower/better than previous best val_loss')
      if os.path.exists(best_data) and os.path.exists(best_index) and os.path.exists(temp_data) and os.path.exists(temp_index):
        os.remove(best_data)
        os.remove(best_index)
        os.rename(temp_data, best_data)
        os.rename(temp_index, best_index)
        print('Replaced old val_loss with new val_loss checkpoints')
      else:
        print('Paths do not exist!!')
    else:
      os.remove(temp_data)
      os.remove(temp_index)
      print('No improvement')


    print('---------------------------------------------------------')
    print(f'''Finished training!!
    Total training time {str(timedelta(seconds= end - start))}
    Temporary checkpoints are saved at {path}.
    Log is saved at {self.logs_path}
    ''')
  

  def get_best_weights_model(self):
    ''' Load best weight and compile model
        return the model
        Should run on a new instance
    '''
    # load before compiling
    print(f'Loading best weights from {self.checkpoints_path}')
    self.model.load_weights(self.checkpoints_path + '/best_val_loss_weights.ckpt')
    self.model.compile(optimizer = self.optimizer, loss = self.loss)

    return self.model

  def get_lattest_weights_model(self):
    ckpt_name, previous_epochs, full_name = self.get_epochs_from_name(self.checkpoints_path)

    # Load and compile model from last training
    print(f'Loading lattest trained weights from epoch {previous_epochs}')
    self.model.load_weights(self.checkpoints_path + '/' + ckpt_name)
    print(f'Loaded: {full_name}')
    self.model.compile(optimizer = self.optimizer, loss = self.loss)
    return self.model
      
  @staticmethod
  def get_epochs_from_name(path):
    ''' Extract the checkpoint name and the epochs number which were saved in previous trains
        ONLY get the latest checkpoint 
        format of checkpoint file E{num_of_epochs}-{date}.ckpt.index
    '''
    name = glob.glob(path + '/*_cont.ckpt.index')
    assert(name) # check empty
    name.sort()

    last = name[-1] #last in the list 
    last = last.split('/')[-1] # get rid of slashes
    ckpt_name = last[:-6] # eliminate '.index', to load using keras
    
    epochs = last.split('_')[1] # get E{epcoch}
    epochs = int(epochs[1:]) # get rid of 'E'
    
    return ckpt_name, epochs, last

  @staticmethod
  def get_loss_from_string(loss_str):
    loss_str = loss_str.lower()
    if loss_str == 'weighted_mse' or loss_str == 'weight_mean_squared_error':
      print('Weighed Mean Squared Error')
      return weighted_mean_squared_error
    elif loss_str == 'mse' or loss_str == 'mean_squared_error':
      print('Mean Squared Error')
      return tf.keras.losses.mean_squared_error
    else:
      print('None')
      return None







