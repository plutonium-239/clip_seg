import torch
import clip
import cv2
import torch.nn as nn
from torchvision.models.segmentation import fcn
import torchvision.transforms as tf
from pspnet import PSPNet
import transform

class SegCLIP_PSP(nn.Module):
	def __init__(self, pspnet, clip_text, hidden_emb_depth=1024):
		super().__init__()
		self.clip_text = clip_text # only for the text part 
		self.embed_dim = hidden_emb_depth
		self.pspnet = pspnet
		fea_dim = 4096
		pspnet.cls = nn.Sequential(
			nn.Conv2d(fea_dim, 1024, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(1024),
		)


	def forward(self, image, text):
		
		

def load_segclip_psp(num_classes, zoom=8, img_size=320):
	model = PSPNet(classes=num_classes, zoom_factor=zoom)
	value_scale = 255
	mean = [0.485, 0.456, 0.406]
	mean = [item * value_scale for item in mean]
	std = [0.229, 0.224, 0.225]
	std = [item * value_scale for item in std]
	train_transform = transform.Compose([
		tf.ToTensor(),
		tf.RandomAffine(scale=(0.5, 2), interpolation=tf.InterpolationMode.BILINEAR),
		tf.RandomRotation(degrees=[-10, 10], interpolation=tf.InterpolationMode.BILINEAR, fill=mean),
		tf.Lambda(lambda x: tf.GaussianBlur(5, 0)(x) if random.random()<0.5 else x),
		tf.RandomHorizontalFlip(),
		tf.RandomCrop(img_size, fill=padding, pad_if_needed=True),
		tf.Normalize(mean=mean, std=std)
	])
	train_transform_label = transform.Compose([
		tf.ToTensor(),
		tf.RandomAffine(scale=(0.5, 2), interpolation=tf.InterpolationMode.NEAREST),
		tf.RandomRotation(degrees=[-10, 10], interpolation=tf.InterpolationMode.NEAREST, fill=mean),
		tf.RandomHorizontalFlip(),
		tf.RandomCrop(img_size, fill=padding, pad_if_needed=True),
		tf.Normalize(mean=mean, std=std)
	])
	return model, 
