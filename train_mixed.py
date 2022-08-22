import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pascal_voc_loader import pascalVOCLoader
from pascal_5i_loader import Pascal5iLoader
from config import config
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import os,sys
import json
from models.model import intersectionAndUnionGPU
import torch.cuda.amp as amp

previous_runs = os.listdir('fewshotruns')
if len(previous_runs) == 0:
	run_number = 1
else:
	run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

logdir = 'run_%02d' % run_number


writer = SummaryWriter('fewshotruns/'+logdir)
layout = {
	"Loss and mIOU for train and val": {
		"loss": ["Multiline", ["loss/train", "loss/val"]],
		"miou": ["Multiline", ["miou/train", "miou/val"]],
	},
}
writer.add_custom_scalars(layout)

def norm_im(im):
	x_min, x_max = im.min(), im.max()
	ims = (im - x_min) / (x_max-x_min)
	return ims


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on', device, 'logging in', logdir)

img_size=224
if len(sys.argv)>1:
	config['fold'] = int(sys.argv[1])
if len(sys.argv)>2:
	img_size = int(sys.argv[2])
if len(sys.argv)>3:
	config['model_name'] = sys.argv[3]
config['img_size'] = img_size
if len(sys.argv)>4:
	if sys.argv[4] == 'jit':
		jit = True
	else:
		jit = False

if config['model_name'] == 'CLIP':
	from models import model_orig
	model, preproc, preproc_lbl = model_orig.load_custom_clip('RN50', device=device, img_size=img_size)
elif config['model_name'] == 'PSPNet':
	from models import model_pspnet
	model, preproc = model_pspnet.load_segclip_psp(zoom=config['zoom'], img_size=img_size, device=device, jit=jit)
	preproc_lbl = None
model.to(device) # redundant

	
runs = json.load(open('fewshotruns.json'))
runs[run_number] = {'fold': config['fold'], 'img_size': img_size}
json.dump(runs, open('fewshotruns.json','w'), indent=4)

# dataset = pascalVOCLoader(config['pascal_root'], preproc, preproc_lbl, split='train', img_size=224, is_transform=True)
dataset = Pascal5iLoader(config['pascal_root'], fold=config['fold'], preproc=preproc, preproc_lbl=preproc_lbl)
trainloader = DataLoader(dataset, batch_size=config['batch_size'], pin_memory=True, num_workers=config['num_workers'])

# valset = pascalVOCLoader(config['pascal_root'], preproc, preproc_lbl, split='val', img_size=224, is_transform=True)
valset = Pascal5iLoader(config['pascal_root'], fold=config['fold'], preproc=preproc, preproc_lbl=preproc_lbl, train=False)
valloader = DataLoader(valset, batch_size=config['batch_size'], pin_memory=True, num_workers=config['num_workers'])

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

scaler = amp.GradScaler()

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
final_loss = 0
final_miou_t = 0
final_miou_v = 0
start = time.time()

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
# 	with record_function("one epoch total"):
for epoch in tqdm(range(config['num_epochs'])):
	# tqdm.write(f'Epoch {epoch} started.')
	epoch_loss_t = 0
	epoch_miou_t = 0
	pbar = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
	model.train()
	for i, (batch_img, batch_lbl) in pbar:
		# times.append(t_diff)
		batch_img, batch_lbl = batch_img.to(device), batch_lbl.to(device)
		# if i==0:
		#   writer.add_graph(model, (batch_img, text_tokens))
		with amp.autocast():
			if config['model_name'] == 'CLIP':
				output = model(batch_img, text_tokens_train)
				# print(output.min(), output.max())
				loss = loss_fn(output, batch_lbl)
			elif config['model_name']=='PSPNet':
				output, main_loss, aux_loss = model(batch_img, text_tokens_train, label=batch_lbl)
				loss = main_loss + config['aux_weight']*aux_loss

		optimiser.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimiser)
		scaler.update()
		
		batch_pred = F.softmax(output, dim=1).argmax(dim=1)
		# batch_pred[batch_pred > config['fold']*5] += 5
		inter,union,_ = intersectionAndUnionGPU(batch_pred, batch_lbl, output.shape[1])
		batch_miou = (inter.sum()/union.sum()).item()
		# tqdm.write(str(i)+str(u))
		epoch_miou_t += batch_miou
		# tqdm.write(str(batch_miou))
		# tqdm.write(str((i/u).mean()))		
		pbar.set_description_str(f'train loss: {loss.item():.4f}, iou: {batch_miou:.4f}')
		epoch_loss_t += loss.item()
		# if i==len(trainloader)-1:
		# 	pred = torch.stack([dataset.decode_segmap(x).permute(2,0,1) for x in batch_pred]).to(device)
		# 	lbl = torch.stack([dataset.decode_segmap(x).permute(2,0,1) for x in batch_lbl]).to(device)
		# 	writer.add_images('img + GT', (norm_im(batch_img)*255).int() | lbl.int(), epoch)
		# 	writer.add_images('img + pred', (norm_im(batch_img)*255).int() | pred.int(), epoch)
		# 	writer.add_images('img', norm_im(batch_img), epoch)
		# 	writer.add_images('GT', lbl, epoch)
		# 	writer.add_images('pred', pred, epoch)
	epoch_loss_t /= len(trainloader)
	epoch_miou_t /= len(trainloader)

	model.eval()
	epoch_miou_v = 0
	epoch_loss_v = 0
	pbar2 = tqdm(enumerate(valloader), total=len(valloader), leave=False)
	for i, (batch_img, batch_lbl) in pbar2:
		# times.append(t_diff)
		batch_img, batch_lbl = batch_img.to(device), batch_lbl.to(device)
		# if i==0:
		#   writer.add_graph(model, (batch_img, text_tokens))
		if config['model_name'] == 'CLIP':
			output = model(batch_img, text_tokens_val)
		elif config['model_name']=='PSPNet':
			output = model(batch_img, text_tokens_val)
		# print(output.min(), output.max())
		batch_pred = F.softmax(output, dim=1).argmax(dim=1)
		# batch_pred[batch_pred > 0] += config['fold']*5
		inter,union,_ = intersectionAndUnionGPU(batch_pred, batch_lbl, output.shape[1])
		batch_miou = (inter.sum()/union.sum()).item()
		# tqdm.write(str(i)+str(u))
		epoch_miou_v += batch_miou
		# tqdm.write(str(batch_miou))
		# tqdm.write(str((i/u).mean()))		
		pbar2.set_description_str(f'val loss: {loss.item():.4f}, iou: {batch_miou:.4f}')
		epoch_loss_v += loss.item()
		if i==0 and epoch%5==4:
			pred = torch.stack([dataset.decode_segmap(x).permute(2,0,1) for x in batch_pred]).to(device)
			lbl = torch.stack([dataset.decode_segmap(x).permute(2,0,1) for x in batch_lbl]).to(device)
			writer.add_images('img vs pred vs GT', torch.cat([norm_im(batch_img), norm_im(pred), norm_im(lbl)], dim=2), epoch)
			# writer.add_images('img + GT', (norm_im(batch_img)*255).int() | lbl.int(), epoch)
			# writer.add_images('img + pred', (norm_im(batch_img)*255).int() | pred.int(), epoch)
			# writer.add_images('img', norm_im(batch_img), epoch)
			# writer.add_images('GT', lbl, epoch)
			# writer.add_images('pred', pred, epoch)
	epoch_loss_v /= len(valloader)
	epoch_miou_v /= len(valloader)
	tqdm.write(f'(train/val) Epoch {epoch} loss: {epoch_loss_t:.4f}/{epoch_loss_v:.4f}, mean mIOU: {epoch_miou_t:.4f}/{epoch_miou_v:.4f}')
	
	writer.add_scalar('loss/train', epoch_loss_t, epoch)
	writer.add_scalar('loss/val', epoch_loss_v, epoch)

	writer.add_scalar('miou/train', epoch_miou_t, epoch)
	writer.add_scalar('miou/val', epoch_miou_v, epoch)
	
	final_miou_t += epoch_miou_t
	final_miou_v += epoch_miou_v
	final_loss = epoch_loss_v
# f = open('profile.txt','w')
# f.write(prof.key_averages().table(sort_by="cpu_time_total"))

end = time.time()
t_diff = end - start
final_miou_t /= config['num_epochs']
final_miou_v /= config['num_epochs']
# writer.add_scalar(f'Epoch time', t_diff, epoch)
print('End')
print()
writer.add_hparams(config, {'mean train mIOU':final_miou_t, 'mean val mIOU':final_miou_v, 'final loss': final_loss, 'total time': t_diff}, run_name='.')

writer.close()
torch.save(model.state_dict(), f'fewshotruns/{logdir}/model.pt')
# subprocess.run(['tensorboard', 'dev', 'upload', '--logdir', 'runs/'])
