import os
import glob
import sys
import numpy as np
from skimage.io import imread
import keras.backend as K
from rgb_tuner import rgb_tuner, rgb_tuner_resnet
from keras.callbacks import ModelCheckpoint
import argparse


parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--input', type=str)
parser.add_argument('--target', type=str)
parser.add_argument('--model', type=str, default='cnn')

args = parser.parse_args()


def main():
    global args
    assert args.model in ['cnn', 'resnet']
    img_dim = [128, 128, 3]
    batch_size = 50
    n_epochs = 50

    input, target = get_data(args.input, args.target)

    if args.model == 'cnn':
        model = rgb_tuner(img_dim)
    else:
        model = rgb_tuner_resnet(img_dim)
    print model.summary()

    model.compile(loss=rgb_loss, optimizer='adam')
    print 'Model created and compiled.'

    model_save = 'rgb_tuner.{}.best.hdf5'.format(args.model)
    checkpoint = ModelCheckpoint(model_save,
                                 monitor='val_loss',
                                 save_best_only=True)

    model.fit(x=input, y=target, validation_split=0.1,
              batch_size=batch_size, epochs=n_epochs, callbacks=[checkpoint])


def rgb_loss(img_true, img_pred):
    total_loss = K.sum(K.square(img_true - img_pred))
    mean_loss = total_loss / (128 * 128 * 3)
    return mean_loss


def get_data(x, y):
    input_paths = glob.glob(os.path.join(x, '*.jpg'))
    target_paths = glob.glob(os.path.join(y, '*.jpg'))
    print 'Paths obtained.'

    input_images = []
    for path in input_paths:
        input_images.append(imread(path) / 255.)
    print 'Input images read.'

    target_images = []
    for path in target_paths:
        target_images.append(imread(path) / 255.)
    print 'Target images read.'

    input = np.stack(input_images, axis=0)
    target = np.stack(target_images, axis=0)

    return input, target


if __name__ == '__main__':
    main()
