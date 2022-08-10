from models import model_orig
from models.model import intersectionAndUnionGPU
import clip
import torch
import torch.nn.functional as F
from pascal_5i_loader import Pascal5iLoader
from config import config
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

# previous_runs = os.listdir('images')
# if len(previous_runs) == 0:
# 	run_number = 1
# else:
# 	run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

config['runid'] = int(sys.argv[1])

logdir = f"run_{config['runid']}"

writer = SummaryWriter('fsimages/'+logdir)

def norm_im(im):
	x_min, x_max = im.min(), im.max()
	ims = (im - x_min) / (x_max-x_min)
	return ims

def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
	fig = plt.figure()
	# Plot the heatmap
	plt.imshow(data, **kwargs)
	# Create colorbar
	plt.colorbar(**cbar_kw)
	plt.ylabel(cbarlabel, rotation=-90, va="bottom")
	# Show all ticks and label them with the respective list entries.
	plt.xticks(np.arange(data.shape[1]), labels=col_labels)
	plt.yticks(np.arange(data.shape[0]), labels=row_labels)
	# Let the horizontal axes labeling appear on top.
	plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
	# Turn spines off and create white grid.
	# plt.spines[:].set_visible(False)
	plt.box(False)
	# plt.xticks(np.arange(data.shape[1]+1)-.5, minor=True)
	# plt.yticks(np.arange(data.shape[0]+1)-.5, minor=True)
	plt.grid(which="minor", color="w", linestyle='-', linewidth=3)
	plt.tick_params(which="minor", bottom=False, left=False)
	return fig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on', device, 'logging in', logdir)

segclip, preproc, preproc_lbl = model_orig.load_custom_clip('RN50', device=device, img_size=224)
segclip.load_state_dict(torch.load(f"fewshotruns/run_{config['runid']}/model.pt", map_location=device))
segclip.to(device) # redundant

# dataset = pascalVOCLoader(config['pascal_root'], preproc, preproc_lbl, split='train', img_size=224, is_transform=True)
# trainloader = DataLoader(dataset, batch_size=config['batch_size'], pin_memory=True, num_workers=config['num_workers'])
config['fold'] = json.load(open('fewshotruns.json'))[f"{config['runid']}"]

dataset = Pascal5iLoader(config['pascal_root'], fold=config['fold'], preproc=preproc, preproc_lbl=preproc_lbl)
trainloader = DataLoader(dataset, batch_size=config['batch_size'], pin_memory=True, num_workers=config['num_workers'])

valset = Pascal5iLoader(config['pascal_root'], fold=config['fold'], preproc=preproc, preproc_lbl=preproc_lbl, train=False)
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
pascal_labels_train = [pascal_labels[x] for x in dataset.label_set]
# pascal_labels_val = [pascal_labels[x] for x in valset.label_set]
pascal_labels_val = pascal_labels
pascal_labels_train.insert(0, '')
# pascal_labels_val.insert(0, '')
text_tokens_train = clip.tokenize(pascal_labels_train).to(device)
text_tokens_val = clip.tokenize(pascal_labels_val).to(device)
print(text_tokens_train.shape)
print(text_tokens_val.shape)
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
	# with record_function("model_inference"):
segclip.eval()

confusion_matrix = torch.zeros(21,21, device=device)
val_confusion_matrix = torch.zeros(21,21)

for i,(img,lbl) in tqdm(enumerate(trainloader), total=len(trainloader)): 
	img, lbl = img.to(device), lbl.to(device)

	pred = segclip(img, text_tokens_train)
	pred = F.softmax(pred, dim=1).argmax(dim=1)

	if i < 20:
		pred = torch.stack([valset.decode_segmap(x).permute(2,0,1) for x in pred]).to(device)
		lbl = torch.stack([valset.decode_segmap(x).permute(2,0,1) for x in lbl]).to(device)

		writer.add_images('TRAIN img vs pred vs GT', torch.cat([norm_im(img), norm_im(pred), norm_im(lbl)], dim=2), global_step=i)
		
	lbl, pred = lbl.long(), pred.long().to(device)
	for label in torch.unique(lbl):
		pred_classes, counts = torch.unique(pred[lbl==label], return_counts=True)
		for j in range(len(pred_classes)):
			confusion_matrix[label][pred_classes[j]] += counts[j]

torch.save(confusion_matrix, 'fsimages/'+logdir+'/conf.pt')
writer.add_figure('conf', heatmap(confusion_matrix, pascal_labels, pascal_labels))

for i,(img,lbl) in tqdm(enumerate(valloader), total=len(trainloader)): 
	img, lbl = img.to(device), lbl.to(device)

	pred = segclip(img, text_tokens_val)
	pred = F.softmax(pred, dim=1).argmax(dim=1)
	
	if i == 20:
		pred = torch.stack([valset.decode_segmap(x).permute(2,0,1) for x in pred]).to(device)
		lbl = torch.stack([valset.decode_segmap(x).permute(2,0,1) for x in lbl]).to(device)

		writer.add_images('VAL img vs pred vs GT', torch.cat([norm_im(img), norm_im(pred), norm_im(lbl)], dim=2), global_step=i)

	for label in torch.unique(lbl):
		pred_classes, counts = torch.unique(pred[lbl==label], return_counts=True)
		for j in range(len(pred_classes)):
			val_confusion_matrix[label][pred_classes[j]] += counts[j]

torch.save(val_confusion_matrix, 'fsimages/'+logdir+'/val_conf.pt')
writer.add_figure('val_conf', heatmap(val_confusion_matrix, pascal_labels, pascal_labels))

writer.close()
