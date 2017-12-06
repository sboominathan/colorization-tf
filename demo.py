import tensorflow as tf
from utils import *
from net import Net
from skimage.io import imsave
from skimage.transform import resize
import cv2

def get_test_image(img_path):
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = img[None, :, :, None].astype(dtype=np.float32)
	return img/255.0 * 100 - 50

def get_colorized_image():
	img_path = "/home/ubuntu/imagenet/original/00000647.jpg"
	test_image = get_test_image(img_path)

	autocolor = Net(train=False)
	lab_distribution = autocolor.inference(test_image)

	saver = tf.train.Saver()
	with tf.Session() as sess:
	  saver.restore(sess, 'models/model.ckpt-22000')
	  lab_distribution = sess.run(lab_distribution)

	img_rgb = decode(test_image, lab_distribution, 2.63)
	imsave('color.jpg', img_rgb)

if __name__ == "__main__":
	get_colorized_image()