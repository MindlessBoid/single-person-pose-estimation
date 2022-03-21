from keras.layers import *
import keras.backend as K
from keras import Model

def create_hourglass_network(input_shape, num_classes, num_stacks, num_filters, bottleneck, hm_activation) -> Model:
  #clear last session
  K.clear_session() 
  _input = Input(shape = input_shape)
  front_features = create_front_module(_input, num_filters, bottleneck)

  head_next_stage = front_features

  outputs = []
  for i in range(num_stacks):
    hg_output = create_hourglass_module(head_next_stage, num_filters, bottleneck, i + 1)
    head_next_stage, head_loss = create_heads(head_next_stage, hg_output, num_classes, num_filters, hm_activation, i + 1)
    outputs.append(head_loss)
  
  model = Model(inputs = _input, outputs = outputs)
  return model

def ConvBlock(_input, num_filters, kernel_size, name, is_mobile = False):
  x = BatchNormalization(name = name + '_bn')(_input)
  x = Activation('relu', name = name + '_act')(x)
  if is_mobile:
     x = SeparableConv2D(filters = num_filters, kernel_size = kernel_size, padding = 'same', name = name + '_conv')(x)
  else:
    x = Conv2D(filters =num_filters, kernel_size = kernel_size, padding = 'same', name = name + '_conv')(x)
  return x

def create_bottleneck(_input, num_filters, name: str):
  '''
  :param _input: input
  :param num_filters: number of filters at output of the bottleneck
  :param name:
  '''

  # Skip layer, if the number of output filters of input is not match with the current bottleneck
  # Map it with a 1x1 ConvD, otherwise keep
  if K.int_shape(_input)[-1] == num_filters:
    skip = _input
  else:
    skip = Conv2D(filters = num_filters, kernel_size = (1, 1), padding = 'same', name = name + '_skip_conv')(_input)

  x = ConvBlock(_input = skip, num_filters = num_filters//2, kernel_size = (1, 1), name = name + '_convblock1')
  x = ConvBlock(_input = x, num_filters = num_filters//2, kernel_size = (3, 3), name = name + '_convblock2')
  x = ConvBlock(_input = x, num_filters = num_filters, kernel_size = (1, 1), name = name + '_convblock3')
  x = Add(name = name + '_output')([skip, x])

  return x

def create_bottleneck_mobile(_input, num_filters, name: str):
  '''
  :param _input: input
  :param num_filters: number of filters at output of the bottleneck
  :param name:
  '''

  # Skip layer, if the number of output filters of input is not match with the current bottleneck
  # Map it with a 1x1 ConvD, otherwise keep
  if K.int_shape(_input)[-1] == num_filters:
    skip = _input
  else:
    skip = SeparableConv2D(filters = num_filters, kernel_size = (1, 1), padding = 'same', name = name + '_skip_conv')(_input)

  x = ConvBlock(_input = skip, num_filters = num_filters//2, kernel_size = (1, 1), name = name + '_convblock1', is_mobile = True)
  x = ConvBlock(_input = x, num_filters = num_filters//2, kernel_size = (3, 3), name = name + '_convblock2', is_mobile = True)
  x = ConvBlock(_input = x, num_filters = num_filters, kernel_size = (1, 1), name = name + '_convblock3', is_mobile = True)
  x = Add(name = name + '_output_mobile')([skip, x])

  return x

def create_front_module(_input, num_filters, bottleneck):
  ''' Frond module
  7x7 conv2D -> 1/2 res
  1 bottleneck
  1 maxpool -> 1/2 res
  2 bottleneck
  output resolution should match with label resolution (.e.i 64x64)
  '''
  x = Conv2D(filters = num_filters//4,  kernel_size = (7, 7), strides = (2, 2), padding = 'same', name = 'front_conv')(_input)
  x = BatchNormalization(name = 'front_bn')(x)
  x = Activation('relu', name = 'front_act')(x)

  x = bottleneck(x, num_filters//2, 'front_bottleneck1')
  x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = 'front_maxpool')(x)
  x = bottleneck(x, num_filters//2, 'front_bottleneck2')
  x = bottleneck(x, num_filters, 'front_bottleneck3')

  return x

def create_hourglass_module(_input, num_filters, bottleneck, hgid: int):
  ''' Hourglass module
  4 downsampling + 4 upscaling
  lf1, lf2, lf3, lf4: 1, 1/2, 1/4, 1/8
  '''
  name = 'hg' + str(hgid)
  # Left features
  lf1 = Lambda(lambda x: x, name = name + '_lf1')(_input)# rename

  lf2 = MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = name + '_lf2_maxpool')(lf1)
  lf2 = bottleneck(lf2, num_filters, name = name + '_lf2_bottleneck')

  lf3 = MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = name + '_lf3_maxpool')(lf2)
  lf3 = bottleneck(lf3, num_filters, name = name + '_lf3_bottleneck')

  lf4 = MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = name + '_lf4_maxpool')(lf3)
  lf4 = bottleneck(lf4, num_filters, name = name + '_lf4_bottleneck')

  # Bottom layer
  bottom = MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = name + '_bottomlayer_maxpool')(lf4)
  bottom = bottleneck(bottom, num_filters, name = name + '_bottomlayer_bottleneck1')
  bottom = bottleneck(bottom, num_filters, name = name + '_bottomlayer_bottleneck2')
  bottom = bottleneck(bottom, num_filters, name = name + '_bottomlayer_bottleneck3')
  bottom = UpSampling2D(name = name + '_bottomlayer_upsampling')(bottom)

  # Connect left to right
  connect1 = bottleneck(lf1, num_filters, name = name + '_connect1_bottleneck')
  connect2 = bottleneck(lf2, num_filters, name = name + '_connect2_bottleneck')
  connect3 = bottleneck(lf3, num_filters, name = name + '_connect3_bottleneck')
  connect4 = bottleneck(lf4, num_filters, name = name + '_connect4_bottleneck')

  # Right features from 4 -> 1
  rf4 = Add(name = name + '_rf4')([bottom, connect4])

  rf3 = bottleneck(rf4, num_filters, name = name + '_rf3_bottlebeck')
  rf3 = UpSampling2D(name = name + '_rf3_upsampling')(rf3)
  rf3 = Add(name = name + '_rf3')([rf3, connect3])

  rf2 = bottleneck(rf3, num_filters, name = name + '_rf2_bottlebeck')
  rf2 = UpSampling2D(name = name + '_rf2_upsampling')(rf2)
  rf2 = Add(name = name + '_rf2')([rf2, connect2])

  rf1 = bottleneck(rf2, num_filters, name = name + '_rf1_bottlebeck')
  rf1 = UpSampling2D(name = name + '_rf1_upsampling')(rf1)
  rf1 = Add(name = name + '_rf1')([rf1, connect1])

  return rf1

def create_heads(hg_input, hg_output, num_classes, num_filters, hm_activation, hgid: int):
  ''''
  2 heads: 1 for heatmaps/label -> loss, 1 for next hourglass
  :param hg_input: input of the hourglass
  :param hg_output: right feature 1
  :param num_classes: num of kpts
  :param num_filters:
  :param hm_activation: either linear or sigmoid
  :param hgid: for naming
  '''
  name = 'head' + str(hgid)
  head = Conv2D(filters = num_filters, kernel_size = (1, 1), padding = 'same', name = name + '_conv1')(hg_output)
  head = BatchNormalization(name = name + '_bn')(head)
  head = Activation('relu', name = name + '_act')(head)

  # for intermediate supervision
  head_loss = Conv2D(filters = num_classes, kernel_size = (1, 1), activation = hm_activation, padding = 'same',
                     name = name + '_loss')(head)

  # map head_loss for next stage, default linar activation
  head_m = Conv2D(filters = num_filters, kernel_size = (1, 1), padding = 'same', name = name + '_conv2')(head_loss)
  # head for next stage
  head =  Conv2D(filters = num_filters, kernel_size = (1, 1), padding = 'same', name = name + '_conv3')(head)

  head_next_stage = Add(name = name + '_next_stage')([hg_input, head_m, head])

  return head_next_stage, head_loss