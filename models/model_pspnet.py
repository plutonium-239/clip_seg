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
		tf.
		transform.RandScale([0.5, 2.0]),
		transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
		transform.RandomGaussianBlur(),
		transform.RandomHorizontalFlip(),
		transform.Crop([img_size, img_size], crop_type='rand', padding=mean, ignore_label=255),
		transform.ToTensor(),
		transform.Normalize(mean=mean, std=std)])
	return model, 
