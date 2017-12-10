import os
import glob
import sys
import numpy as np
from skimage.io import imread, imsave
from rgb_tuner import rgb_tuner
from keras.callbacks import ModelCheckpoint
import argparse


parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--train_x', type=str)
parser.add_argument('--train_y', type=str)

args = parser.parse_args()


def main():
    global args
    img_dim = [128, 128, 3]
    batch_size = 50
    n_epochs = 50

    # data_dir = 'data'
    # train_dir = os.path.join(data_dir, 'train')
    # val_dir = os.path.join(data_dir, 'val')
    # test_dir = os.path.join(data_dir, 'test')

    train_x, train_y = get_data(args.train_x, args.train_y)
    # val = get_data(val_dir)
    # test = get_data(test_dir)

    model = rgb_tuner(img_dim, batch_size)
    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(x=train_x, y=train_y, validation_data=val,
    #           batch_size=batch_size, epochs=n_epochs)
    print 'Model created and compiled.'

    model_save = 'rgb_tuner.best.hdf5'
    checkpoint = ModelCheckpoint(model_save,
                                 monitor='val_loss',
                                 save_best_only=True)

    model.fit(x=train_x, y=train_y, validation_split=0.1,
              batch_size=batch_size, epochs=n_epochs, callbacks=[checkpoint])

    # preds = model.predict(test[0], verbose=1)

    # pred_path = os.path.join(test_dir, 'pred')
    # if not os.path.exists(pred_path):
    #     os.makedirs(pred_path)

    # assert preds[0].shape == tuple(img_dim)

    # for i, pred in preds:
    #     img_path = os.path.join(pred_path,
    #                             '{}_pred.jpg'.format(str(i + 1).zfill(8)))
    #     imsave(img_path, pred)


def get_data(x, y):
    input_paths = glob.glob(os.path.join(x, '*.jpg'))
    target_paths = glob.glob(os.path.join(y, '*.jpg'))
    print 'Paths obtained.'

    input_images = []
    for path in input_paths:
        input_images.append(imread(path))
    print 'Input images read.'

    target_images = []
    for path in target_paths:
        target_images.append(imread(path))
    print 'Target images read.'

    input = np.stack(input_images, axis=0)
    target = np.stack(target_images, axis=0)

    return input, target


if __name__ == '__main__':
    main()
