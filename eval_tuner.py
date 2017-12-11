import os
import glob
import sys
import numpy as np
from skimage.io import imread, imsave
from keras.models import load_model
import keras.backend as K
import argparse
import keras.losses


parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('model', type=str)
parser.add_argument('input', type=str)

args = parser.parse_args()


def main():
    global args
    img_dim = [128, 128, 3]

    eval = get_data(args.input)

    model = load_model(args.model)
    print 'Model created.'

    preds = model.predict(eval, verbose=1).astype(np.uint8)
    print preds

    pred_path = 'pred'
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    assert preds[0].shape == tuple(img_dim)

    for i in xrange(preds.shape[0]):
        img_path = os.path.join(pred_path,
                                '{}_pred.jpg'.format(str(i + 1).zfill(8)))
        imsave(img_path, preds[i])


def rgb_loss(img_true, img_pred):
    total_loss = K.sum(K.square(img_true - img_pred))
    mean_loss = total_loss / (128 * 128 * 3)
    return mean_loss


def get_data(data_path):
    if data_path.endswith('.jpg'):
        img_paths = [data_path]
    else:
        img_paths = glob.glob(os.path.join(data_path, '*.jpg'))
    print 'Paths obtained.'

    images = []
    for path in img_paths:
        images.append(imread(path))
        print imread(path)
    print 'Images read.'

    return np.stack(images, axis=0)


if __name__ == '__main__':
    keras.losses.rgb_loss = rgb_loss
    main()
