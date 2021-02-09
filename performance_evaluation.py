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


def performance_eval(model, test_loader):

	# Getting predictions for the entire test set

	with torch.no_grad():
		# test_preds = torch.tensor([], device='cuda:0') # if we are using CUDA, uncomment this line and comment the following line
		test_preds = torch.tensor([])
		for x, y in test_loader:
			test_preds = torch.cat(
				(test_preds, model(x))
				,dim=0
			)

	# Plotting the confusion matrix

	# test_preds = test_preds.cpu() # Since I am not using CUDA, I commented this line too
	cm = confusion_matrix(test_loader.dl.dataset.labels, test_preds.argmax(dim=1))

	class_labels = ('class: 1', 'class: 2', 'class: 3', 'class: 4', 'class: 5','class: 6','class: 7','class: 8','class: 9','class: 10','class: 11','class: 12','class: 13','class: 14','class: 15','class: 16','class: 17',
		'class: 18','class: 19','class: 20','class: 21','class: 22','class: 23','class: 24','class: 25','class: 26','class: 27','class: 28','class: 29','class: 30','class: 31','class: 32','class: 33','class: 34',
		'class: 35','class: 36','class: 37','class: 38','class: 39','class: 40','class: 41','class: 42','class: 43')
		
	plt.figure(figsize=(30,30))
	plot_confusion_matrix(cm, class_labels)
	plt.show()
	
	# Calculation of recall, precision and F1 score
	accuracy = np.sum(np.diag(cm))/np.sum(cm)
	recall = np.diag(cm) / np.sum(cm, axis = 1)
	precision = np.diag(cm) / np.sum(cm, axis = 0)

	recall_overall = np.mean(recall)
	precision_overall = np.mean(precision)

	F1_score_overall = 2 * (recall_overall * precision_overall)/(recall_overall + precision_overall)
	
	return cm, accuracy, recall_overall, precision_overall, F1_score_overall

if __name__ == "__main__":
	# Evaluation settings
	parser = argparse.ArgumentParser(
		description='Traffic sign recognition performance evaluation script')
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

	# Data Initialization and Loading
	test_loader = get_test_loader(args.data, device)
	confusion_mat, accuracy, recall, precision, F1_score = performance_eval(model, test_loader)
	
	# Displaying all the performance metrics calculated
	print(f"ACCURACY : {accuracy:.6f}")
	print(f"RECALL : {recall:.6f}")
	print(f"PRECISION : {precision:.6f}")
	print(f"F1_SCORE : {F1_score:.6f}")
	print(f"CONFUSION_MATRIX :\n {confusion_mat}")
	
	
	
