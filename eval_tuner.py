import os
import glob
import sys
import numpy as np
from skimage.io import imread, imsave
from keras.models import load_model
import argparse


parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('model', type=str)
parser.add_argument('input', type=str)

args = parser.parse_args()


def main():
    global args
    img_dim = [128, 128, 3]

    eval = get_data(input)

    model = load_model(args.model)
    print 'Model created.'

    preds = model.predict(eval[0], verbose=1)

    pred_path = 'pred'
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    assert preds[0].shape == tuple(img_dim)

    for i, pred in preds:
        img_path = os.path.join(pred_path,
                                '{}_pred.jpg'.format(str(i + 1).zfill(8)))
        imsave(img_path, pred)


def get_data(path):
    if path.endswith('.jpg'):
        img_paths = [path]
    else:
        img_paths = glob.glob(os.path.join(path, '*.jpg'))
    print 'Paths obtained.'

    images = []
    for path in img_paths:
        images.append(imread(path))
    print 'Images read.'

    return np.stack(images, axis=0)


if __name__ == '__main__':
    main()
