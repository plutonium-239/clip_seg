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

writer = SummaryWriter('fsimagesnobg/'+logdir)

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
	plt.setp(plt.xticks()[1], rotation=-90, ha="right", rotation_mode="anchor")
	# plt.xticks(np.arange(data.shape[1]+1)-.5, minor=True)
	# plt.yticks(np.arange(data.shape[0]+1)-.5, minor=True)
	plt.grid(which="minor", color="w", linestyle='-', linewidth=3)
	plt.tick_params(which="minor", bottom=False, left=False)
	return fig

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

dataset = Pascal5iLoader(config['pascal_root'], fold=config['fold'], preproc=preproc, preproc_lbl=preproc_lbl)
trainloader = DataLoader(dataset, batch_size=config['batch_size'], pin_memory=True, num_workers=config['num_workers'])

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
pascal_labels_train = [pascal_labels[x] for x in dataset.label_set]
# pascal_labels_val = [pascal_labels[x] for x in valset.label_set]
pascal_labels_val = pascal_labels[1:]
# pascal_labels_train.insert(0, '')
# pascal_labels_val.insert(0, '')
text_tokens_train = clip.tokenize(pascal_labels_train).to(device)[1:]
text_tokens_val = clip.tokenize(pascal_labels_val).to(device)[1:]
print(text_tokens_train.shape)
print(text_tokens_val.shape)
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
	# with record_function("model_inference"):
model.eval()

# approx ones only do +1 instead of +(num_of_pixels) for a pair of (gt, pred) labels
confusion_matrix = torch.zeros(20,20, device=device)
confusion_matrix_approx = torch.zeros(20,20, device=device)
val_confusion_matrix = torch.zeros(20,20, device=device)
val_confusion_matrix_approx = torch.zeros(20,20, device=device)

for i,(img,lbl) in tqdm(enumerate(trainloader), total=len(trainloader)): 
	img, lbl = img.to(device), lbl.to(device)

	pred = model(img, text_tokens_train)
	pred = F.softmax(pred, dim=1).argmax(dim=1)

	if i < 20:
		pred = torch.stack([valset.decode_segmap(x).permute(2,0,1) for x in pred]).to(device)
		lbl = torch.stack([valset.decode_segmap(x).permute(2,0,1) for x in lbl]).to(device)

		writer.add_images('TRAIN img vs pred vs GT', torch.cat([norm_im(img), norm_im(pred), norm_im(lbl)], dim=2), global_step=i)
		

	lbl, pred = lbl.long(), pred.long().to(device)
	for label in torch.unique(lbl):
		if label == 255:
			continue
		pred_classes, counts = torch.unique(pred[lbl==label], return_counts=True)
		for j in range(len(pred_classes)):
			confusion_matrix[label][pred_classes[j]] += counts[j]
			confusion_matrix_approx[label][pred_classes[j]] += 1

torch.save(confusion_matrix, 'fsimagesnobg/'+logdir+'/conf.pt')
writer.add_figure('conf', heatmap(confusion_matrix.cpu()[1:,1:], pascal_classes, pascal_classes))
torch.save(confusion_matrix_approx, 'fsimagesnobg/'+logdir+'/conf_approx.pt')
writer.add_figure('conf_approx', heatmap(confusion_matrix_approx.cpu()[1:,1:], pascal_classes, pascal_classes))


vc_miou, tc_miou = 0

for i,(img,lbl) in tqdm(enumerate(valloader), total=len(valloader)): 
	img, lbl = img.to(device), lbl.to(device)

	pred = model(img, text_tokens_val)
	psh = pred.shape[1]
	pred = F.softmax(pred, dim=1).argmax(dim=1)
	inter,union,_ = intersectionAndUnionGPU(pred, lbl, psh)
	train_classes_miou = (inter[valset.train_label_set].sum() / union[valset.train_label_set].sum()).item()
	val_classes_miou = (inter[valset.val_label_set].sum() / union[valset.val_label_set].sum()).item()

	vc_miou += train_classes_miou  
	tc_miou += val_classes_miou
	
	if i == 20:
		pred = torch.stack([valset.decode_segmap(x).permute(2,0,1) for x in pred]).to(device)
		lbl = torch.stack([valset.decode_segmap(x).permute(2,0,1) for x in lbl]).to(device)

		writer.add_images('VAL img vs pred vs GT', torch.cat([norm_im(img), norm_im(pred), norm_im(lbl)], dim=2), global_step=i)

	lbl, pred = lbl.long(), pred.long().to(device)
	for label in torch.unique(lbl):
		if label == 255:
			continue
		pred_classes, counts = torch.unique(pred[lbl==label], return_counts=True)
		for j in range(len(pred_classes)):
			val_confusion_matrix[label][pred_classes[j]] += counts[j]
			val_confusion_matrix_approx[label][pred_classes[j]] += 1

vc_miou /= len(valloader)
tc_miou /= len(valloader)

torch.save(val_confusion_matrix, 'fsimagesnobg/'+logdir+'/val_conf.pt')
writer.add_figure('val_conf', heatmap(val_confusion_matrix.cpu()[1:,1:], pascal_classes, pascal_classes))
torch.save(val_confusion_matrix_approx, 'fsimagesnobg/'+logdir+'/val_conf_approx.pt')
writer.add_figure('val_conf_approx', heatmap(val_confusion_matrix_approx.cpu()[1:,1:], pascal_classes, pascal_classes))

writer.close()
