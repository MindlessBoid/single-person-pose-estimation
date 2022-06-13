from keras.layers import Conv2D, SeparableConv2D, BatchNormalization, MaxPool2D, Add, UpSampling2D, Input
import keras.backend as K
from keras import Model

def create_hourglass_model(num_classes, num_stacks, num_channels, input_shape, predict_activation, mobile = False):
  # Clear
  K.clear_session()

  bottleneck = bottleneck_block
  if mobile:
    bottleneck = bottleneck_block_mobile

  _input = Input(shape = input_shape)

  # Front module, reduce 1/4 resolution
  front_features = create_front_module(_input, num_channels, bottleneck)

  # Stack
  head_next_stage = front_features
  outputs = []
  for i in range(num_stacks):
    head_next_stage, head_to_loss = hourglass_module(head_next_stage, num_classes, num_channels, bottleneck, i, predict_activation)
    outputs.append(head_to_loss)

  model = Model(inputs=_input, outputs=outputs)

  print(f'''Created Hourglass model:
    1. {num_stacks} stacks.
    2. {model.count_params()} parameters. Call model.get_summary() for more detail.
    ''')

  return model  
  

def hourglass_module(x, num_classes, num_channels, bottleneck, hg_id, predict_activation):
  '''
    A single hourglass module
    Input of this module should have the resolution of (64, 64)
    1. Downsampling to 1/8
    2. Upsampling back to original
    3. Generate 2 heads: one for loss, one for next hourglass module
  '''
  # Down sample features f1, f2, f4, f8
  downsample_features = create_downsample_blocks(x, bottleneck, hg_id, num_channels)

  # Upsample features and merge with down sample features
  upsample_feature = create_upsample_blocks(downsample_features, bottleneck, hg_id, num_channels)

  # Heads
  head_next_stage, head_predict = create_heads(x, upsample_feature, num_classes, hg_id, num_channels, predict_activation)

  return head_next_stage, head_predict

def create_front_module(_input, num_channels, bottleneck):
  '''
    One done once, at the very beginning
    From (256, 256) -> (64, 64)
  '''
  _x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', name='front_conv_1x1_1')(_input)
  _x = BatchNormalization()(_x)

  _x = bottleneck(_x, num_channels//2, 'front_bottleneck_1')
  _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)

  _x = bottleneck(_x, num_channels//2, 'front_bottleneck_2')
  _x = bottleneck(_x, num_channels, 'front_bottleneck_3')

  return _x


def create_heads(x, upsample_feature_f1, num_classes, hg_id, num_channels, predict_activation: str):
  '''
    Create two heads: one for loss, one for the next hourglass
    x should be the input of the hourglass module
  '''

  name = 'hg' + str(hg_id)

  head = Conv2D(num_channels, kernel_size=(1, 1), activation='relu', padding='same', name=name+'_conv_1x1_1')(upsample_feature_f1)
  head = BatchNormalization()(head)

  # For prediction and loss calculation
  head_predict = Conv2D(num_classes, kernel_size=(1, 1), activation=predict_activation, padding='same', name=name+'_conv_1x1_predict')(head)

  # Heads for adding
  # use linear activation
  head = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same', name=name+'_conv_1x1_2')(head)
  head_m = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same', name=name+'_conv_1x1_3')(head_predict)

  # add all
  head_next_stage = Add()([head, head_m, x])
  
  return head_next_stage, head_predict


def create_upsample_blocks(downsample_features, bottleneck, hg_id, num_channels):
  '''
    Upsampling and merging accordingly to the scale:

    - downsample_f1 feature: 64 x 64 x num_channels
    - downsample_f2 feature: 32 x 32 x num_channels
    - downsample_f4 feature: 16 x 16 x num_channels
    - downsample_f8 feature: 8 x 8 x num_channels

    - bottom: 4 x 4 x num_channels

    - upsample_f8 feature: 8 x 8 x num_channels
    - upsample_f4 feature: 16 x 16 x num_channels
    - upsample_f2 feature: 32 x 32 x num_channels
    - upsample_f1 feature: 64 x 64 x num_channels

  '''
  name = 'hg' + str(hg_id)

  downsample_f1, downsample_f2, downsample_f4, downsample_f8 = downsample_features

  bottom = bottom_block(downsample_f8, bottleneck, hg_id, num_channels)

  upsample_f8 = connect_downsample_upsample(downsample_f8, bottom, bottleneck, num_channels, name + '_upsample_f8')
  upsample_f4 = connect_downsample_upsample(downsample_f4, upsample_f8, bottleneck, num_channels, name +'_upsample_f4')
  upsample_f2 = connect_downsample_upsample(downsample_f2, upsample_f4, bottleneck, num_channels, name +'_upsample_f2')
  upsample_f1 = connect_downsample_upsample(downsample_f1, upsample_f2, bottleneck, num_channels, name +'_upsample_f1')

  return upsample_f1


def bottom_block(downsample_f8, bottleneck, hg_id, num_channels):
  '''
    In the lowest resolution (4, 4)
    1. MaxPool to (4,4)
    2. 3 bottlenecks
  '''
  name = 'hg' + str(hg_id)

  _x = MaxPool2D()(downsample_f8)
  _x = bottleneck(_x, num_channels, name + '_downsample_f8_1')
  _x = bottleneck(_x, num_channels, name + '_downsample_f8_2')
  _x = bottleneck(_x, num_channels, name + '_downsample_f8_3')

  return _x


def connect_downsample_upsample(downsample_feature, upsample_feature, bottleneck, num_channels, name):
  '''
    Add the same scale downsample and up sample
    1. Apply 1 bottleneck for the downsample (if u see the figure its the connection up the air)
    2. Upscaling the upsample, which is lower then current down feature
    3. Add
    4. Apply one more bottleneck
  '''
  downsample_x = bottleneck(downsample_feature, num_channels, name + '_short') # shortcut
  upsample_x = UpSampling2D()(upsample_feature)
  # Add both
  _x = Add()([downsample_x, upsample_x])
  _x = bottleneck(_x, num_channels, name + '_merged')
  
  return _x
  

def create_downsample_blocks(x, bottleneck, hg_id, num_channels):
  '''
    Create 4 downsample blocks for 4 levels:
      dowsample_f1 feature: 64 x 64 x num_channels
      dowsample_f2 feature: 32 x 32 x num_channels
      dowsample_f4 feature: 16 x 16 x num_channels
      dowsample_f8 feature: 8 x 8 x num_channels
  '''
  name = 'hg' + str(hg_id)
  
  downsample_f1 = bottleneck(x, num_channels, name + '_downsample_f1')
  _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(downsample_f1)
  
  downsample_f2 = bottleneck(_x, num_channels, name + '_downsample_f2')
  _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(downsample_f2)

  downsample_f4 = bottleneck(_x, num_channels, name + '_downsample_f4')
  _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(downsample_f4)

  downsample_f8 = bottleneck(_x, num_channels, name + '_downsample_f8')

  return (downsample_f1, downsample_f2, downsample_f4, downsample_f8)


def bottleneck_block(x, num_out_channels, name):
  ''''
    Standard bottle neck block, using standard conv
  '''

  # Skip layer, to map if number of input channels is diff than output
  if K.int_shape(x)[-1] == num_out_channels:
    _skip = x
  else:
    _skip = Conv2D(filters=num_out_channels, kernel_size=(1, 1), activation='relu', padding='same', name=name + '_skip')(x)

  # 3 convs: num_out_channels/2, num_out_channels/2, num_out_channels/2
  _x = Conv2D(filters=num_out_channels//2, kernel_size=(1, 1), activation='relu', padding='same', name=name + '_conv_1x1_1')(x)
  _x = BatchNormalization()(_x)
  _x = Conv2D(filters=num_out_channels//2, kernel_size=(3, 3), activation='relu', padding='same', name=name + '_conv_3x3_2')(_x)
  _x = BatchNormalization()(_x)
  _x = Conv2D(filters=num_out_channels, kernel_size=(1, 1), activation='relu', padding='same', name=name + '_conv_1x1_3')(_x)
  _x = BatchNormalization()(_x)

  # Merge
  _x = Add(name=name + '_add')([_skip, _x])

  return _x


def bottleneck_block_mobile(x, num_out_channels, name):
  ''''
    Mobile bottle neck block, using separable conv, lightweight
  '''

  # Skip layer, to map if number of input channels is diff than output
  if K.int_shape(x)[-1] == num_out_channels:
    _skip = x
  else:
    _skip = SeparableConv2D(filters=num_out_channels, kernel_size=(1, 1), activation='relu', padding='same', name=name + '_skip')(x)

  # 3 convs: num_out_channels/2, num_out_channels/2, num_out_channels/2
  _x = SeparableConv2D(filters=num_out_channels//2, kernel_size=(1, 1), activation='relu', padding='same', name=name + '_conv_1x1_1')(x)
  _x = BatchNormalization()(_x)
  _x = SeparableConv2D(filters=num_out_channels//2, kernel_size=(3, 3), activation='relu', padding='same', name=name + '_conv_3x3_2')(_x)
  _x = BatchNormalization()(_x)
  _x = SeparableConv2D(filters=num_out_channels, kernel_size=(1, 1), activation='relu', padding='same', name=name + '_conv_1x1_3')(_x)
  _x = BatchNormalization()(_x)

  # Merge
  _x = Add(name=name + '_add')([_skip, _x])

  return _x