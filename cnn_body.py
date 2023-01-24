import os
import shutil
import config
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import torch.optim as optim
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# # pretrained model VGG19
# tf.keras.applications.vgg19.VGG19(
#     include_top=True,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation='softmax'
# )

def get_all_images(images_folder):
	images = []

	for entry in os.scandir(images_folder):
		if entry.is_dir():
			images.extend(get_all_images(entry.path))
		else:
			images.append(entry.path)

	return images

def get_tvt_split(source_dir, data_dir):

	plants_types = os.listdir(source_dir)

	for plant_type in plants_types:
		train_folder = f"{data_dir}/train_{plant_type}"
		val_folder = f"{data_dir}/val_{plant_type}"
		test_folder = f"{data_dir}/test_{plant_type}"

		os.makedirs(train_folder, exist_ok=True)
		os.makedirs(val_folder, exist_ok=True)
		os.makedirs(test_folder, exist_ok=True)

		plant_images = get_all_images(f"{source_dir}/{plant_type}")

		np.random.shuffle(plant_images)

		train_images, val_images, test_images = np.split(
			np.array(plant_images), 
	    [int(len(plant_images)*0.7), 
	    int(len(plant_images)*0.85)]
		)

		# View count of images after partition
		print(f"Data for '{plant_type}'")
		print("Total images: ", len(plant_images))
		print("Training: ", len(train_images))
		print("Validation: ", len(val_images))
		print("Testing: ", len(test_images))
		print("")

		# Copy-paste images
		for image_path in train_images:
			shutil.copy(image_path, train_folder)

		for image_path in val_images:
			shutil.copy(image_path, val_folder)

		for image_path in test_images:
			shutil.copy(image_path, test_folder)

get_tvt_split(config.CNN_FOLDER, config.DATA_FOLDER)
