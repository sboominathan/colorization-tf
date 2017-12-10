from keras.models import Model
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose


def conv_downsample_block(x, block_id, filter, block, end_stride):

    for i in range(1, block + 1):
        name = 'conv_block_{0}_{1}'.format(block_id, i)
        stride = end_stride if i == block else 1

        kwargs = {'padding': 'same',
                  'activation': 'relu',
                  'name': name}

        x = Conv2D(filter, 3, strides=stride, **kwargs)(x)
    x = BatchNormalization(axis=1)(x)

    return x


def conv_upsample_block(x, block_id, filter, block, end_stride):
    for i in range(1, block + 1):
        name = 'conv_block_{0}_{1}'.format(block_id, i)
        stride = end_stride if i == block else 1

        kwargs = {'padding': 'same',
                  'activation': 'relu',
                  'name': name}

        x = Conv2DTranspose(filter, 3, strides=stride, **kwargs)(x)
    x = BatchNormalization(axis=1)(x)

    return x


def rgb_tuner(img_dim, batch_size, model_name='colorize'):
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
