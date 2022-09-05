import clip
import torch
import torch.nn.functional as F
from pascal_5i_loader import Pascal5iLoader
from config import config
from torch.utils.data import DataLoader
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from models.model import intersectionAndUnionGPU

# previous_runs = os.listdir('images')
# if len(previous_runs) == 0:
# 	run_number = 1
# else:
# 	run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

config['runid'] = int(sys.argv[1])

logdir = f"run_{config['runid']}"

run = json.load(open('fewshotruns.json'))[f"{config['runid']}"]
config['fold'] = run['fold']
config['img_size'] = run['img_size']
config['model_name'] = run['model_name']

def norm_im(im):
	x_min, x_max = im.min(), im.max()
	ims = (im - x_min) / (x_max-x_min)
	return ims

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on', device, 'logging in', logdir)


if config['model_name'] == 'CLIP':
	from models import model_orig
	model, preproc, preproc_lbl = model_orig.load_custom_clip('RN50', device=device, img_size=config['img_size'])
elif config['model_name'] == 'PSPNet':
	from models import model_pspnet
	model, preproc = model_pspnet.load_segclip_psp(zoom=config['zoom'], img_size=config['img_size'], device=device)
	preproc_lbl = None
model.to(device) # redundant
pretrained_dict = torch.load(f"fewshotruns/run_{config['runid']}/model.pt", map_location=device)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
model.load_state_dict(pretrained_dict)

# dataset = pascalVOCLoader(config['pascal_root'], preproc, preproc_lbl, split='train', img_size=224, is_transform=True)
# trainloader = DataLoader(dataset, batch_size=config['batch_size'], pin_memory=True, num_workers=config['num_workers'])

valset = Pascal5iLoader(config['pascal_root'], fold=config['fold'], preproc=preproc, preproc_lbl=preproc_lbl, train=False)
valloader = DataLoader(valset, batch_size=config['batch_size'], pin_memory=True, num_workers=config['num_workers'])

pascal_classes = [
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
pascal_labels = [template+x for x in pascal_classes]
pascal_labels.insert(0, '')
# pascal_labels_train = [pascal_labels[x] for x in dataset.label_set]
# pascal_labels_val = [pascal_labels[x] for x in valset.label_set]
pascal_labels_val = pascal_labels
# pascal_labels_train.insert(0, '')
pascal_labels_val.insert(0, '')
# text_tokens_train = clip.tokenize(pascal_labels_train).to(device)
text_tokens_val = clip.tokenize(pascal_labels_val).to(device)
# print(text_tokens_train.shape)
print(text_tokens_val.shape)
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
	# with record_function("model_inference"):
model.eval()

vc_miou_i, vc_miou_u, tc_miou_i, tc_miou_u  = 0, 0, 0, 0

for i,(img,lbl) in tqdm(enumerate(valloader), total=len(valloader)): 
	img, lbl = img.to(device), lbl.to(device)

	preds = model(img, text_tokens_val)
	pred_mask = F.softmax(preds, dim=1).argmax(dim=1)
	
	inter,union,_ = intersectionAndUnionGPU(pred_mask, lbl, preds.shape[1])
	# print(inter, union)
	# print((inter/union))
	# train_classes_miou = (inter/union)[valset.train_label_set].sum().item()
	# val_classes_miou = (inter/union)[valset.val_label_set].sum().item()

	tc_miou_i += inter[valset.train_label_set].sum().item()
	tc_miou_u += union[valset.train_label_set].sum().item()
	vc_miou_i += inter[valset.val_label_set].sum().item()
	vc_miou_u += union[valset.val_label_set].sum().item()
	

# vc_miou /= len(valloader)
# tc_miou /= len(valloader)
vc_miou = vc_miou_i/vc_miou_u
tc_miou = tc_miou_i/tc_miou_u

print(vc_miou)
print(tc_miou)
