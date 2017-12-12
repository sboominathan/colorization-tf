import tensorflow as tf
from utils import *
from net import Net
from resnet import ResNet
from skimage.io import imsave
from skimage.transform import resize
from tqdm import tqdm
import glob
import cv2

def get_test_image(img_path):
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = img[None, :, :, None].astype(dtype=np.float32)
	return img/255.0 * 100 - 50

def get_colorized_image():
	img_path = "/home/ubuntu/vol/imagenet/train/00000075.jpg"
	# img_path = "/home/ubuntu/vol/colorization/data/test/00000163.jpg"
	# img_path = "gray.jpg"
	test_image = get_test_image(img_path)

	autocolor = ResNet(train=False)
	# autocolor = Net(train=False)
	lab_distribution = autocolor.inference(test_image)

	model_resnet = 'models/model_resnet.ckpt-21000'
	model_old = 'models/model.ckpt-30000'

	saver = tf.train.Saver()
	with tf.Session() as sess:
	  saver.restore(sess, model_resnet)
	  lab_distribution = sess.run(lab_distribution)

	img_rgb = decode(test_image, lab_distribution, 2.83)
	imsave('color.jpg', img_rgb)

def get_batch_colorized_image(batch_list):
	autocolor = Net(train=False)
	lab_distribution = autocolor.inference(batch_list)

	saver = tf.train.Saver()
	with tf.Session() as sess:
	  saver.restore(sess, 'models/model.ckpt-34000')
	  lab_distribution = sess.run(lab_distribution)

	rgb_images = []

	for i in range(lab_distribution.shape[0]):
		img_rgb = decode(batch_list[i:i+1,:,:,:], lab_distribution[i:i+1,:,:,:], 2.83)
		rgb_images.append(img_rgb)
	return rgb_images

def get_all_colorized_images():
	img_list = []
	batch_size = 100
	img_list = sorted(glob.glob("/home/ubuntu/vol/imagenet/train/*.jpg"))
	# img_list = sorted(glob.glob("/home/ubuntu/vol/colorization/data/test/*.jpg"))

	all_images = []
	num_batches = int(len(img_list)/batch_size)

	autocolor = ResNet(train=False)
	batch_list = tf.placeholder(tf.float32, (batch_size, 128, 128, 1))
	lab_distribution = autocolor.inference(batch_list)
	saver = tf.train.Saver()

	with tf.Session() as sess:
		saver.restore(sess, 'models/model_resnet.ckpt-20000')
		for i in tqdm(range(num_batches)):
			image_paths = img_list[i*batch_size:(i+1)*batch_size]
			images = [get_test_image(img_path) for img_path in image_paths]
			batch_image_arr = np.concatenate(images, axis=0)

			pred_distribution = sess.run(lab_distribution, feed_dict={batch_list: batch_image_arr})

			rgb_images = []
			for j in range(pred_distribution.shape[0]):
				img_rgb = decode(batch_image_arr[j:j+1,:,:,:], pred_distribution[j:j+1,:,:,:], 2.83)
				rgb_images.append(img_rgb)

			all_images.extend(rgb_images)

	for i in range(len(all_images)):
		image_name = str(i+1).zfill(8) + '_colorized.jpg'
		imsave(os.path.join('colorized_images/resnet_train', image_name), all_images[i])

if __name__ == "__main__":
	get_all_colorized_images()