from keras.models import Model
from keras.layers import Input, Add
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose


def conv_downsample_block(x, block_id, filter, block, end_stride):

    for i in range(1, block + 1):
        name = 'conv_block_{}_{}'.format(block_id, i)
        stride = end_stride if i == block else 1

        kwargs = {'padding': 'same',
                  'activation': 'relu',
                  'name': name}

        x = Conv2D(filter, 3, strides=stride, **kwargs)(x)
    x = BatchNormalization(axis=1)(x)

    return x


def conv_upsample_block(x, block_id, filter, block, end_stride):
    for i in range(1, block + 1):
        name = 'conv_block_{}_{}'.format(block_id, i)
        stride = end_stride if i == block else 1

        kwargs = {'padding': 'same',
                  'activation': 'relu',
                  'name': name}

        x = Conv2DTranspose(filter, 3, strides=stride, **kwargs)(x)
    x = BatchNormalization(axis=1)(x)

    return x


def rgb_tuner(img_dim, model_name='tuner_cnn'):
    h, w, c = img_dim
    curr_h, curr_w = img_dim[:2]
    block_id = 1

    filter_sizes = [64, 128]
    block_sizes = [2, 2]
    end_strides = [2, 1]

    inputs = Input(shape=(h, w, c), name="input")
    x = inputs
    for filter, block, stride in zip(filter_sizes, block_sizes, end_strides):
        x = conv_downsample_block(x, block_id, filter, block, stride)

        curr_h, curr_w = int(curr_h / stride), int(curr_w / stride)
        block_id += 1

    for filter, block, stride in zip(filter_sizes[1::-1],
                                     block_sizes[1::-1],
                                     end_strides[1::-1]):
        x = conv_upsample_block(x, block_id, filter, block, stride)

        curr_h, curr_w = int(curr_h / stride), int(curr_w / stride)
        block_id += 1

    x = Conv2D(3, 2, strides=1, padding='same')(x)

    return Model(inputs=[inputs], outputs=[x], name=model_name)


def residual_block(x, block_id, filter):
    name = 'block_{}_conv_a'.format(block_id)
    r = Conv2D(filter, 3, padding="same", activation='relu', name=name)(x)
    r = BatchNormalization(axis=1,
                           name='block_{}_bn_a'.format(block_id))(r)

    name = 'block_{}_conv_b'.format(block_id)
    r = Conv2D(filter, 3, padding="same", activation='relu', name=name)(r)
    r = BatchNormalization(axis=1,
                           name='block_{}_bn_b'.format(block_id))(r)

    x = Add(name='block_{}_merge'.format(block_id))([x, r])

    return x


def rgb_tuner_resnet(img_dim, model_name='tuner_resnet'):
    h, w, c = img_dim
    block_id = 1
    n_blocks = 3

    inputs = Input(shape=(h, w, c), name="input")
    name = 'block_{}_initial'.format(block_id)
    x = Conv2D(64, 3, padding='same',
               activation='relu', name=name)(inputs)

    for i, filter in enumerate([64] * n_blocks):
        x = residual_block(x, block_id, filter)
        block_id += 1

    x = Conv2D(3, 2, strides=1, padding='same',
               name='block_{}_final'.format(block_id))(x)

    return Model(inputs=[inputs], outputs=[x], name=model_name)
