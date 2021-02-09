# import all the necessary libraries to perform the performance evaluation of the trained neural network

import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from torch import nn
from model import TrafficSignNet
from data import get_test_loader
from torchvision.utils import make_grid
from train import valid_batch
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix

if __name__ == "__main__":
    # Evaluation settings
	parser = argparse.ArgumentParser(
		description='Traffic sign recognition feature visualisation script')
	parser.add_argument('--data', type=str, default='data', metavar='D',
						help="folder where data is located. test.p need to be found in the folder (default: data)")
	parser.add_argument('--model', type=str, default='model.pt', metavar='M',
						help="the model file to be evaluated. (default: model.pt)")

	args = parser.parse_args()
	
	# Load model checkpoint
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoint = torch.load(args.model, map_location=device)

	# Neural Network and Loss Function
	model = TrafficSignNet().to(device)
	model.load_state_dict(checkpoint)

	model_children = list(model.children());

	# load the data that are needed to be visualized
	test_loader = get_test_loader(args.data, device)

	# As instructed in the Programming assignment, Selecting 10 images from at least 5 different classes
	images = []
	images_classes = []
	dict = {}
	for x, y in test_loader:
		if(len(images)<10):
			dict[y]= dict[y] + 1 if (y in dict) else 1
			if(dict[y] <= 2):
				images.append(x)
				images_classes.append(y)
		else: 
			break

	# Get the feature output of layer 'conv3' for each individual image and storing it in a list
	features_output = []
	for image in images:
		result = image
		layer_index = 0
		for child in model_children:
			layer_index += 1
			if layer_index < 7:
				result = child (result)
		features_output.append(result) 

	# Saving the features output for each individual image in a picture with subplots
	for image_number in range(len(features_output)):
		plt.figure(figsize=(30, 30))
		layer_viz = result[0, :, :, :]
		layer_viz = layer_viz.data
		print(layer_viz.size())
		for i, filter in enumerate(layer_viz):
			if i == 64: # visualizing only first 64 channels in the feature output of conv3 layer
				break
			plt.subplot(8, 8, i + 1)
			plt.imshow(filter.cpu(), cmap='gray')
			plt.axis("off")
		print(f"Saving conv3 layer's output of image {image_number} (feature maps)...")
		plt.savefig(f"./outputs/image_{image_number}.png")
		plt.show()
		plt.close()