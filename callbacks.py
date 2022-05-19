import tensorflow as tf
def make_checkpoint_callback(checkpoints_path):
  return tf.keras.callbacks.ModelCheckpoint(filepath = checkpoints_path,
                                            save_weights_only = True,
                                            monitor = 'val_loss',
                                            mode = 'min', # since monitor val_loss, overwrite when its min
                                            save_best_only=True,
                                            verbose = True)

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_begin(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      self.model.optimizer.lr.numpy()))