"""
Module containing reader to parse pascal_5i dataset from SBD and VOC2012
"""
import os
from PIL import Image
from scipy.io import loadmat
import numpy as np
import torch
import torchvision
import cv2
from tqdm import tqdm


class Pascal5iLoader(torchvision.datasets.vision.VisionDataset):
	"""
	pascal_5i dataset reader
	Parameters:
		- root:  root to data folder containing SBD and VOC2012 dataset. See README.md for details
		- fold:  folding index as in OSLSM (https://arxiv.org/pdf/1709.03410.pdf)
		- train: a bool flag to indicate whether L_{train} or L_{test} should be used
	"""

	def __init__(self, root, fold, preproc, preproc_lbl, train=True):
		super(Pascal5iLoader, self).__init__(root, None, None, None)
		assert fold >= 0 and fold <= 3
		self.train = train

		# Define base to SBD and VOC2012
		sbd_base = os.path.join(root, 'sbd')
		voc_base = os.path.join(root, 'VOCdevkit', 'VOC2012')

		# Define path to relevant txt files
		sbd_train_list_path = os.path.join(root, 'sbd', 'train.txt')
		sbd_val_list_path = os.path.join(root, 'sbd', 'val.txt')
		voc_train_list_path = os.path.join(
			voc_base, 'ImageSets', 'Segmentation', 'train.txt')
		voc_val_list_path = os.path.join(
			voc_base, 'ImageSets', 'Segmentation', 'val.txt')

		# Use np.loadtxt to load all train/val sets
		sbd_train_list = list(np.loadtxt(sbd_train_list_path, dtype="str"))
		sbd_val_list = list(np.loadtxt(sbd_val_list_path, dtype="str"))
		voc_train_list = list(np.loadtxt(voc_train_list_path, dtype="str"))
		voc_val_list = list(np.loadtxt(voc_val_list_path, dtype="str"))

		# Following PANet, we use images in SBD validation for training
		sbd_train_list = sbd_train_list + sbd_val_list

		# Remove overlapping images in SBD/VOC2012 from SBD train
		sbd_train_list = [i for i in sbd_train_list if i not in voc_val_list]

		# Generate self.images and self.targets
		if self.train:
			# If an image occur in both SBD and VOC2012, use VOC2012 annotation
			sbd_train_list = [
				i for i in sbd_train_list if i not in voc_train_list]

			# Generate image/mask full paths for SBD dataset
			sbd_train_img_list = [os.path.join(
				sbd_base, 'img', i + '.jpg') for i in sbd_train_list]
			sbd_train_target_list = [os.path.join(
				sbd_base, 'cls', i + '.mat') for i in sbd_train_list]

			# Generate image/mask full paths for VOC2012 segmentation training task
			voc_train_img_list = [os.path.join(
				voc_base, 'JPEGImages', i + '.jpg') for i in voc_train_list]
			voc_train_target_list = [os.path.join(
				voc_base, "SegmentationClass", 'pre_encoded', i + '.png') for i in voc_train_list]

			# FINAL: Merge these two datasets
			self.images = sbd_train_img_list + voc_train_img_list
			self.targets = sbd_train_target_list + voc_train_target_list
		else:
			# Generate image/mask full paths for VOC2012 semantation validation task
			# Following PANet, only VOC2012 validation set is used for validation
			self.images = [os.path.join(
				voc_base, 'JPEGImages', i + '.jpg') for i in voc_val_list]
			self.targets = [os.path.join(
				voc_base, "SegmentationClass", 'pre_encoded', i + '.png') for i in voc_val_list]

		# Split dataset based on folding. Refer to https://arxiv.org/pdf/1709.03410.pdf
		# Given fold number, define L_{test}
		self.val_label_set = [i for i in range(fold * 5 + 1, fold * 5 + 6)]
		self.train_label_set = [i for i in range(
			1, 21) if i not in self.val_label_set]
		if self.train:
			self.label_set = self.train_label_set
		else:
			self.label_set = self.val_label_set
		# self.label_offset = [0] + self.label_set 
		self.n_classes = 21

		assert len(self.images) == len(self.targets)
		self.to_tensor_func = torchvision.transforms.ToTensor()

		# Find subset of image. This is actually faster than hist
		folded_images = []
		folded_targets = []

		# Given a class, this dict returns list of images containing the class
		self.class_img_map = {}
		for label_id, _ in enumerate(self.label_set):
			self.class_img_map[label_id + 1] = []

		# Given an index of an image, this dict returns list of classes in the image
		self.img_class_map = {}

		if os.path.exists(f"dataset_{fold}_{train}.pt"):
			print('Using Saved Dataset')
			d = torch.load(f"dataset_{fold}_{train}.pt")
			self.img_class_map = d['icm']
			self.class_img_map = d['cim']
			folded_images = d['fi']
			folded_targets = d['ft']
		else:
			print('Creating Dataset')
			for i in tqdm(range(len(self.images))):
				mask = self.load_seg_mask(self.targets[i])
				appended_flag = False
				for label_id, x in enumerate(self.label_set):
					if x in mask:
						if not appended_flag:
							# contain at least one pixel in L_{train}
							folded_images.append(self.images[i])
							folded_targets.append(self.targets[i])
							appended_flag = True
						cur_img_id = len(folded_images) - 1
						cur_class_id = label_id + 1
						# This image must be the latest appended image
						self.class_img_map[cur_class_id].append(cur_img_id)
						if cur_img_id in self.img_class_map:
							self.img_class_map[cur_img_id].append(cur_class_id)
						else:
							self.img_class_map[cur_img_id] = [cur_class_id]
			torch.save({'icm':self.img_class_map, 
				'cim':self.class_img_map,
				'fi': folded_images,
				'ft': folded_targets,
				}, f"dataset_{fold}_{train}.pt")

		self.images = folded_images
		self.targets = folded_targets
		self.preproc = preproc
		self.preproc_lbl = preproc_lbl


	def __len__(self):
		return len(self.images)

	def load_seg_mask(self, file_path):
		"""
		Load seg_mask from file_path (supports .mat and .png).
		Target masks in SBD are stored as matlab .mat; while those in VOC2012 are .png
		Parameters:
			- file_path: path to the segmenation file
		Return: a numpy array of dtype long and element range(0, 21) containing segmentation mask
		"""
		if file_path.endswith('.mat'):
			mat = loadmat(file_path)
			target_np = mat['GTcls'][0]['Segmentation'][0]
		else:
			target_np = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
			target_np = self.encode_segmap(target_np)
		# target_np = np.array(target, dtype=np.long)

		# Annotation in VOC contains 255
		target_np[target_np > 20] = 0

		return target_np

	def set_bg_pixel(self, target_np):
		"""
		Following OSLSM, we mask pixels not in current label set as 0. e.g., when
		self.train = True, pixels whose labels are in L_{test} are masked as background
		Parameters:
			- target_np: segmentation mask (usually returned array from self.load_seg_mask)
		Return:
			- Offseted and masked segmentation mask
		"""
		if self.train:
			for x in self.val_label_set:
				target_np[target_np == x] = 0
			max_val_label = max(self.val_label_set)
			target_np[target_np >
					  max_val_label] = target_np[target_np > max_val_label] - 5
		else:
			label_mask_idx_map = []
			for x in self.val_label_set:
				label_mask_idx_map.append(target_np == x)
			target_np = np.zeros_like(target_np)
			for i in range(len(label_mask_idx_map)):
				target_np[label_mask_idx_map[i]] = i + 1
		return target_np

	def get_img_containing_class(self, class_id):
		"""
		Given a class label id (e.g., 2), return a list of all images in
		the dataset containing at least one pixel of the class.
		Parameters:
			- class_id: an integer representing class
		Return:
			- a list of all images in the dataset containing at least one pixel of the class
		"""
		return self.class_img_map[class_id]

	def get_class_in_an_image(self, img_idx):
		"""
		Given an image idx (e.g., 123), return the list of classes in
		the image.
		Parameters:
			- img_idx: an integer representing image
		Return:
			- list of classes in the image
		"""
		return self.img_class_map[img_idx]

	def __getitem__(self, idx):
		# For both SBD and VOC2012, images are stored as .jpg
		img = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
		img = self.preproc(img)

		target_np = self.load_seg_mask(self.targets[idx])
		target_np = self.set_bg_pixel(target_np)
		target_np = self.preproc_lbl(target_np).long()

		return img, target_np

	def get_pascal_labels(self):
		"""Load the mapping that associates pascal classes with label colors

		Returns:
			np.ndarray with dimensions (21, 3)
		"""
		return np.asarray(
			[
				[0, 0, 0],
				[128, 0, 0],
				[0, 128, 0],
				[128, 128, 0],
				[0, 0, 128],
				[128, 0, 128],
				[0, 128, 128],
				[128, 128, 128],
				[64, 0, 0],
				[192, 0, 0],
				[64, 128, 0],
				[192, 128, 0],
				[64, 0, 128],
				[192, 0, 128],
				[64, 128, 128],
				[192, 128, 128],
				[0, 64, 0],
				[128, 64, 0],
				[0, 192, 0],
				[128, 192, 0],
				[0, 64, 128],
			]
		)

	def encode_segmap(self, mask):
		"""Encode segmentation label images as pascal classes

		Args:
			mask (np.ndarray): raw segmentation label image of dimension
			  (M, N, 3), in which the Pascal classes are encoded as colours.

		Returns:
			(np.ndarray): class map with dimensions (M,N), where the value at
			a given location is the integer denoting the class index.
		"""
		mask = mask.astype(int)
		label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
		for ii, label in enumerate(self.get_pascal_labels()):
			label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
		label_mask = label_mask.astype(int)
		return label_mask

	def decode_segmap(self, label_mask):
		"""Decode segmentation class labels into a color image

		Args:
			label_mask (np.ndarray): an (M,N) array of integer values denoting
			  the class label at each spatial location.
			plot (bool, optional): whether to show the resulting color image
			  in a figure.

		Returns:
			(np.ndarray, optional): the resulting decoded color image.
		"""
		label_colours = self.get_pascal_labels()
		r = label_mask.clone()
		g = label_mask.clone()
		b = label_mask.clone()
		for ll in range(0, self.n_classes):
			r[label_mask == ll] = label_colours[ll, 0]
			g[label_mask == ll] = label_colours[ll, 1]
			b[label_mask == ll] = label_colours[ll, 2]
		rgb = torch.zeros((label_mask.shape[0], label_mask.shape[1], 3))
		rgb[:, :, 0] = r / 255.0
		rgb[:, :, 1] = g / 255.0
		rgb[:, :, 2] = b / 255.0
		return rgb