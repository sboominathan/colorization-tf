from utils import *
from skimage.io import imsave
from skimage.transform import resize
from tqdm import tqdm
import glob
import cv2
import os

def get_rgb_difference(dir1, dir2):

	# get all images from directories 
	img_filelist_1 = sorted(glob.glob(dir1))
	img_filelist_2 = sorted(glob.glob(dir2))

	img_list_1 = [cv2.imread(img_file) for img_file in img_filelist_1]
	img_list_2 = [cv2.imread(img_file) for img_file in img_filelist_2]

	# calculate difference between each image

	for img1, img2 in tqdm(zip(img_list_1, img_list_2)):
		difference = 0.5*np.sum(np.square(img1-img2))
		total_difference += difference

	# calculate mean squared difference between each image and return average
	return difference/len(img_filelist_1)

if __name__=="__main__":
	dir1 = "/home/ubuntu/vol/colorization-tf/colorized_images/resnet_test/*.jpg"
	dir2 = "/home/ubuntu/vol/colorization/data/test/*.jpg"
	# dir2 = "/home/ubuntu/vol/imagenet/train/*.jpg"
	
	squared_error = get_rgb_difference(dir1, dir2)
	print("Squared Error: ", squared_error)