import model
import clip
import torch
import torch.nn as nn
# import torch.nn.functional as F
from pascal_voc_loader import pascalVOCLoader
from config import config
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

segclip, preproc, preproc_lbl = model.load_custom_clip('RN50', num_classes=5, device=device)

dataset = pascalVOCLoader(config['pascal_root'], preproc, preproc_lbl, split=config['mode'], img_size=224, is_transform=True)
trainloader = DataLoader(dataset, batch_size=config['batch_size'])


loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(segclip.parameters(), lr=config['lr'], weight_decay=config['wd'])

pbar = tqdm(enumerate(trainloader), total=len(trainloader))

pascal_labels = [
		'aeroplane',
		'bicycle',
		'bird',
		'boat',
		'bottle',
		'bus',
		'car',
		'cat',
		'chair',
		'cow',
		'dog',
		'horse',
		'motorbike',
		'person',
		'sheep',
		'sofa',
		'diningtable',
		'pottedplant',
		'train',
		'tvmonitor',
	]

template = 'a photo of a '
pascal_labels = [template+x for x in pascal_labels]
text_tokens = clip.tokenize(pascal_labels).to(device)

for i, (batch_img, batch_lbl, texts) in pbar:
	batch_img, batch_lbl = batch_img.to(device), batch_lbl.to(device)

	output = segclip(batch_img, text_tokens)
	print(output.shape, 
		batch_img.shape,
		# output.sum(dim=0).shape, 
		text_tokens.shape,
		batch_lbl.shape)
	# 1024 dimensional embedding for each pixel (or 1024 channels) and 
	# take dot product of all N classes and take softmax of that
	loss = loss_fn(output, batch_lbl)
	loss.backward()
	optimiser.step()
	pbar.set_postfix({'loss':loss.item()})
	
