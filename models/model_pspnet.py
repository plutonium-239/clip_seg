import torch
import clip
import cv2
import torch.nn as nn
import torch.nn.functional as F
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
		self.zoom_factor = self.pspnet.zoom_factor
		self.pspnet.cls = nn.Sequential(
			nn.Conv2d(fea_dim, 1024, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(1024),
		)

	def forward(self, image, text, label=None):
		image_size = image.size()
		# [batch, channels=3, H=320, W=320]
		assert (image_size[2]) % 8 == 0 and (image_size[3]) % 8 == 0
		h = int((image_size[2]) / 8 * self.zoom_factor)
		w = int((image_size[3]) / 8 * self.zoom_factor)
		
		with torch.no_grad():
			text = self.clip.encode_text(text)

		image = self.pspnet.layer0(image)
		image = self.pspnet.layer1(image)
		image = self.pspnet.layer2(image)
		aux = self.pspnet.layer3(image)
		image = self.pspnet.layer4(aux)
		if self.pspnet.use_ppm:
			image = self.pspnet.ppm(image)
		image = self.pspnet.cls(image)
		
		if self.zoom_factor != 1:
			image = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=True)

		ish = image.shape
		image = image.permute(0,2,3,1).reshape(-1, image.shape[1]) # linearize for faster dot product
		x = image @ text.t() # [1*320*320, 1024] dot [1024, num_classes] = [1*320*320, num_classes]
		x = x.reshape(ish[0], ish[2], ish[3], -1).permute(0,3,1,2)

		if self.training:
			if self.zoom_factor != 1:
				aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
			ish = aux.shape
			aux = aux.permute(0,2,3,1).reshape(-1, aux.shape[1]) # linearize for faster dot product
			xaux = aux @ text.t() # [1*320*320, 1024] dot [1024, num_classes] = [1*320*320, num_classes]
			xaux = xaux.reshape(ish[0], ish[2], ish[3], -1).permute(0,3,1,2)

			main_loss = self.pspnet.criterion(x, label)
			aux_loss = self.pspnet.criterion(xaux, label)
			return x, main_loss, aux_loss
		else:
			return x
		
		

def load_segclip_psp(num_classes, zoom=8, img_size=320, device=None):
	pspnet = PSPNet(zoom_factor=zoom)
	clip_text, _ = clip.load('initmodel/RN50.pt', device=device)
	del clip_text.visual
	clip_text = clip_text.float()
	
	model = SegCLIP_PSP(pspnet, clip_text)

	value_scale = 255
	mean = [0.485, 0.456, 0.406]
	mean = [item * value_scale for item in mean]
	std = [0.229, 0.224, 0.225]
	std = [item * value_scale for item in std]

	train_transform = transform.Compose([
		transform.RandScale([0.5, 2]),
		transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
		transform.RandomGaussianBlur(),
		transform.RandomHorizontalFlip(),
		transform.Crop([img_size, img_size], crop_type='rand', padding=mean, ignore_label=255),
		transform.ToTensor(),
		transform.Normalize(mean=mean, std=std)
	])
	
	return model, train_transform
