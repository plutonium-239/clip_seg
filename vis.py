import model
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from pascal_voc_loader import pascalVOCLoader
from config import config
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import os
import subprocess # for uploading tensorboard
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
from model import intersectionAndUnionGPU

# previous_runs = os.listdir('images')
# if len(previous_runs) == 0:
# 	run_number = 1
# else:
# 	run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

logdir = 'run_%02d' % config['runid']

writer = SummaryWriter('images/'+logdir)

def norm_im(im):
	x_min, x_max = im.min(), im.max()
	ims = (im - x_min) / (x_max-x_min)
	return ims

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on', device, 'logging in', logdir)

segclip, preproc, preproc_lbl = model.load_custom_clip('RN50', num_classes=5, device=device)
segclip.load_state_dict(f"runs/run_{config['runid']}/model.pt")
segclip.to(device) # redundant

# dataset = pascalVOCLoader(config['pascal_root'], preproc, preproc_lbl, split='train', img_size=224, is_transform=True)
# trainloader = DataLoader(dataset, batch_size=config['batch_size'], pin_memory=True, num_workers=config['num_workers'])

valset = pascalVOCLoader(config['pascal_root'], preproc, preproc_lbl, split='val', img_size=224, is_transform=True)
valloader = DataLoader(valset, batch_size=config['batch_size'], pin_memory=True, num_workers=config['num_workers'])

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
pascal_labels.insert(0, '')
text_tokens = clip.tokenize(pascal_labels).to(device)
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
	# with record_function("model_inference"):
segclip.eval()

img, lbl, _ = next(iter(valloader))
img, lbl = img.to(device), lbl.to(device)

pred = segclip(img, text_tokens)
pred = torch.stack([valset.decode_segmap(x).permute(2,0,1) for x in pred]).to(device)
lbl = torch.stack([valset.decode_segmap(x).permute(2,0,1) for x in lbl]).to(device)

writer.add_images('img vs pred vs GT', torch.cat([norm_im(img), norm_im(pred), norm_im(lbl)], dim=2))

writer.close()