import torch
import clip
import cv2
import torch.nn as nn
from torchvision.models.segmentation import fcn
import torchvision.transforms as tf

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
		
	def forward(self, x):
		return x


class SegCLIP(nn.Module):
	def __init__(self, clip_encoder, num_classes, hidden_emb_depth):
		super().__init__()
		# self.clip_visual = clip_encoder.visual
		self.clip = clip_encoder # only for the text part 
		self.num_classes = num_classes
		self.embed_dim = hidden_emb_depth
		'''
		Layer (type:depth-idx)                   Output Shape
		=================================================================
		ModifiedResNet                           [1, 2048, 7, 7]
		├─Conv2d: 1-1                            [1, 32, 112, 112]
		├─BatchNorm2d: 1-2                       [1, 32, 112, 112]
		├─ReLU: 1-3                              [1, 32, 112, 112]
		├─Conv2d: 1-4                            [1, 32, 112, 112]
		├─BatchNorm2d: 1-5                       [1, 32, 112, 112]
		├─ReLU: 1-6                              [1, 32, 112, 112]
		├─Conv2d: 1-7                            [1, 64, 112, 112]
		├─BatchNorm2d: 1-8                       [1, 64, 112, 112]
		├─ReLU: 1-9                              [1, 64, 112, 112]
		├─AvgPool2d: 1-10                        [1, 64, 56, 56]
		├─Sequential: 1-11                       [1, 256, 56, 56]
		├─Sequential: 1-12                       [1, 512, 28, 28]
		├─Sequential: 1-13                       [1, 1024, 14, 14]
		├─Sequential: 1-14                       [1, 2048, 7, 7]

		Removing:
		├─AttentionPool2d: 1-15                  [1, 2048, 7, 7]

		Adding:
		├─FCNHead: 1-3                                [1, 1024, 28, 28]         [1, 21, 28, 28]
		│    └─Conv2d: 2-14                           [1, 1024, 28, 28]         [1, 256, 28, 28]
		│    └─BatchNorm2d: 2-15                      [1, 256, 28, 28]          [1, 256, 28, 28]
		│    └─ReLU: 2-16                             [1, 256, 28, 28]          [1, 256, 28, 28]
		│    └─Dropout: 2-17                          [1, 256, 28, 28]          [1, 256, 28, 28]
		│    └─Conv2d: 2-18                           [1, 256, 28, 28]          [1, 21, 28, 28]

		'''
		# to remove the Attention Pool layer (replace with identity)
		# self.clip_encoder.visual.attnpool = Identity()
		width = 224
		self.up1 = nn.Upsample((width//8,width//8))
		self.conv1 = nn.Conv2d(in_channels = 2048, out_channels = 512, kernel_size=3, padding=2, dilation=2, bias=False)
		self.bn1 = nn.BatchNorm2d(512)
		self.relu1 = nn.ReLU(inplace=True)

		self.up2 = nn.Upsample((width//4,width//4))
		self.conv2 = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size=3, padding=2, dilation=2, bias=False)
		self.bn2 = nn.BatchNorm2d(256)
		self.relu2 = nn.ReLU(inplace=True)

		self.up3 = nn.Upsample((width, width))
		self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size=3, padding=2, dilation=2, bias=False)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU(inplace=True)
		
		# self.fcnhead = fcn.FCNHead(in_channels = 64, channels = hidden_emb_depth)
		self.fcnheadconv = nn.Conv2d(in_channels = 64, out_channels = hidden_emb_depth, kernel_size=3, padding=2, dilation=2, bias=False)
		self.fcnheadbn = nn.BatchNorm2d(hidden_emb_depth)

	def forward(self, image, text):
		image = image.type(self.clip.visual.conv1.weight.dtype)
		# identity = image.clone()

		def stem(image):
			image = self.clip.visual.relu1(self.clip.visual.bn1(self.clip.visual.conv1(image)))
			image = self.clip.visual.relu2(self.clip.visual.bn2(self.clip.visual.conv2(image)))
			image = self.clip.visual.relu3(self.clip.visual.bn3(self.clip.visual.conv3(image)))
			image = self.clip.visual.avgpool(image)
			return image

		with torch.no_grad():
			text = self.clip.encode_text(text)
	
			image = stem(image)
			# [1, 64, 56, 56]
			image = self.clip.visual.layer1(image)
			res = image.clone()
			# [1, 256, 56, 56]
			image = self.clip.visual.layer2(image)
			res2 = image.clone()
			# [1, 512, 28, 28]
			image = self.clip.visual.layer3(image)
			# [1, 1024, 14, 14]
			image = self.clip.visual.layer4(image)
			# [1, 2048, 7, 7]

		# removing attnpool
		# image = self.clip_encoder.visual.attnpool(image)
		
		image = self.up1(image)
		image = self.conv1(image)
		# [1, 512, 28, 28]
		image += res2
		image = self.relu1(self.bn1(image))
		
		image = self.up2(image)
		image = self.conv2(image)
		# [1, 256, 56, 56]
		image += res
		image = self.relu2(self.bn2(image))
		
		image = self.up3(image)
		image = self.conv3(image)
		# [1, 64, 224, 224]
		# cannot do this since identity has only 3 channels
		# image += identity
		image = self.relu3(self.bn3(image))
		
		image = self.fcnheadconv(image)
		image = self.fcnheadbn(image)
		# [num_classes, 224, 224]

		'''
		image -> [1, 1024, 224, 224]
		text -> [num_classes, 1024]

		we want -> [1, num_classes, 224, 224]
		'''
		ish = image.shape
		image = image.permute(0,2,3,1).reshape(-1, image.shape[1]) # linearize for faster dot product
		x = image @ text.t() # [1*224*224, 1024] dot [1024, num_classes] = [1*224*224, num_classes]
		x = x.reshape(ish[0], ish[2], ish[3], -1).permute(0,3,1,2)

		return x
		




def load_custom_clip(model_name, num_classes, device=None):
	'''
	Function to load 'model_name' from clip and adapt it for segmentation
	Returns:
	model: The CLIP model with .visual and .transformer
	preproc: The preprocessing function reqd by model. Expects a numpy array of [H, W, C]
	'''
	assert model_name in clip.available_models(), f"model_name should be one of RNx from {clip.available_models()}, found {model_name}"
	model, preprocess = clip.load(model_name, device=device)
	model = model.float()
	# doing .float() is important otherwise it outputs nans (known issue on github page)
	'''
	this is 'preprocess' (torchvision.transforms) for RN50: 
	Compose(
		Resize(size=224, interpolation=bicubic)
		CenterCrop(size=(224, 224))
		<function _convert_image_to_rgb at 0x000002364D1EE430>
		ToTensor()
		Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
	)

	It expects a PIL image in the 3rd transform (convert to rgb)
	But since cv2(numpy) is ~1.5x faster than PIL we will use that
	We can call `cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)` and remove that from preprocess
	'''
	embed_dim =  model.text_projection.shape[1]
	segCLIP = SegCLIP(model, num_classes, embed_dim)
	preprocess.transforms.pop(2) # convert_to_rgb
	preprocess.transforms.pop(2) # ToTensor
	preprocess.transforms[0] = tf.Resize(size=224, interpolation=tf.InterpolationMode.BICUBIC)
	preproc = lambda x: preprocess( tf.ToTensor()(x) )
	preprocess_label = tf.Compose([])
	preprocess_label.transforms = preprocess.transforms.copy()
	# label cant be interpolated normally, use nearest neighbor interpolation to keep integer values
	preprocess_label.transforms[0] = tf.Resize(size=224, interpolation=tf.InterpolationMode.NEAREST)
	preprocess_label.transforms.pop(-1) # no normalization for label
	preproc_lbl = lambda x: preprocess_label(torch.tensor(x).unsqueeze(0)).squeeze(0)


	return segCLIP.to(device), preproc, preproc_lbl


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
	# 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
	assert (output.dim() in [1, 2, 3])
	assert output.shape == target.shape
	output = output.view(-1)
	target = target.view(-1)
	output[target == ignore_index] = ignore_index
	intersection = output[output == target]
	area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
	area_output = torch.histc(output, bins=K, min=0, max=K-1)
	area_target = torch.histc(target, bins=K, min=0, max=K-1)
	area_union = area_output + area_target - area_intersection
	return area_intersection, area_union, area_target