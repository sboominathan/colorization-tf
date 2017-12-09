import os
import glob
import numpy as np
from skimage.io import imread
from rgb_tuner import rgb_tuner


def train():
    img_dim = [128, 128, 3]
    batch_size = 50
    n_epochs = 50
    data_dir = 'data'
    train_dir = os.path(data_dir, 'train')
    val_dir = os.path(data_dir, 'train')

    model = rgb_tuner(img_dim, batch_size)
    model.compile(loss='mean_squared_error', optimizer='adam')

    train_x, train_y = get_data(train_dir)
    val = get_data(val_dir)

    model.fit(x=train_x, y=train_y, validation_data=val,
              batch_size=batch_size, epochs=n_epochs)


def get_data(dir):
    input_dir = 'input'
    target_dir = 'target'

    input_paths = glob.glob(os.path.join(dir, input_dir, '*.jpg'))
    target_paths = glob.glob(os.path.join(dir, target_dir, '*.jpg'))

    input_images = []
    for path in input_paths:
        input_images.append(imread(path))

    target_images = []
    for path in target_paths:
        target_images.append(imread(path))

    input = np.stack(input_images, axis=0)
    target = np.stack(target_images, axis=0)

    return input, target
