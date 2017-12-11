import tensorflow as tf 
import numpy as np
from skimage.io import imread
from skimage import color
import sys
import random
from skimage.transform import resize
from tqdm import tqdm


lists_f = open('data/train.txt')
points = np.load('resources/pts_in_hull.npy')
filename_list = []

for img_file in lists_f:
  filename_list.append(img_file.strip())
random.shuffle(filename_list)

points = points.astype(np.float64)
points = points[None, :, :]

probs = np.zeros((313), dtype=np.float64)

# Input image of size (H*W, 2)
in_data = tf.placeholder(tf.float64, [None, 2])

# Expand image by one axis so that we can broadcast bin locations over them
expand_in_data = tf.expand_dims(in_data, axis=1)

# Get distance of every pixel to all 313 AB bins
distance = tf.reduce_sum(tf.square(expand_in_data - points), axis=2)

# Get index of closest bin
index = tf.argmin(distance, axis=1)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

sess = tf.Session(config=config)

for img_name in tqdm(filename_list):
  img_name = img_name.strip()
  img = imread(img_name)
  img = resize(img, (128, 128), preserve_range=True)

  img_lab = color.rgb2lab(img)
  img_lab = img_lab.reshape((-1,3))
  img_ab_channels = img_lab[:,1:]

  closest_bin = sess.run(index, feed_dict={in_data: img_ab_channels})
  for i in closest_bin:
    probs[int(i)] += 1

  sys.stdout.flush()

sess.close()
probs = probs/np.sum(probs)
np.save('prior_probs_new.npy', probs)

# for img_f in tqdm(filename_lists):
#   img_f = img_f.strip()
#   img = imread(img_f)
#   img = resize(img, (128, 128), preserve_range=True)
#   if len(img.shape)!=3 or img.shape[2]!=3:
#     continue
#   img_lab = color.rgb2lab(img)
#   img_lab = img_lab.reshape((-1, 3))
#   img_ab = img_lab[:, 1:]
#   nd_index = sess.run(index, feed_dict={in_data: img_ab})
#   for i in nd_index:
#     i = int(i)
#     probs[i] += 1
#   sys.stdout.flush()
#   num += 1
# sess.close()
# probs = probs / np.sum(probs)
# print(probs)
# np.save('probs', probs)
