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

previous_runs = os.listdir('runs')
if len(previous_runs) == 0:
	run_number = 1
else:
	run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

logdir = 'run_%02d' % run_number

writer = SummaryWriter('runs/'+logdir)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on', device, 'logging in', logdir)

segclip, preproc, preproc_lbl = model.load_custom_clip('RN50', num_classes=5, device=device)
segclip.to(device) # redundant

dataset = pascalVOCLoader(config['pascal_root'], preproc, preproc_lbl, split=config['mode'], img_size=224, is_transform=True)
trainloader = DataLoader(dataset, batch_size=config['batch_size'], pin_memory=True, num_workers=config['num_workers'])


loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(segclip.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

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
final_loss = 0
final_miou = 0
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
	# with record_function("model_inference"):
start = time.time()

for epoch in tqdm(range(config['num_epochs'])):
	# tqdm.write(f'Epoch {epoch} started.')
	epoch_loss = 0
	epoch_miou = 0
	pbar = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
	t_sum = 0
	for i, (batch_img, batch_lbl, texts) in pbar:
		# times.append(t_diff)
		batch_img, batch_lbl = batch_img.to(device), batch_lbl.to(device)
		# if i==0:
		#   writer.add_graph(segclip, (batch_img, text_tokens))
		output = segclip(batch_img, text_tokens)
		# print(output.min(), output.max())
		loss = loss_fn(output, batch_lbl)
		loss.backward()
		optimiser.step()
		i,u,_ = intersectionAndUnionGPU(F.softmax(output, dim=1).argmax(dim=1), batch_lbl, output.shape[1])
		iou = i/u
		epoch_miou += iou.mean().item()
		pbar.set_description_str(f'loss: {loss.item()}, iou: {iou.mean().item()}')
		epoch_loss += loss.item()
	epoch_loss /= len(trainloader)
	epoch_miou /= len(trainloader)	
	tqdm.write(f'Epoch {epoch} loss: {epoch_loss}, mean mIOU: {epoch_miou}')
	# plot = plt.plot(times)
	# if epoch%10==9:
	writer.add_scalar('training loss', epoch_loss, epoch)
	writer.add_scalar('epoch mIOU', epoch_miou, epoch)
	final_miou += epoch_miou
	final_loss = epoch_loss
end = time.time()
t_diff = end - start
final_miou /= config['num_epochs']
# writer.add_scalar(f'Epoch time', t_diff, epoch)
print('End')
print()
# f = open('profile.txt','w')
# f.write(prof.key_averages().table(sort_by="cpu_time_total"))
writer.add_hparams(config, {'mean mIOU':final_miou, 'final loss': final_loss, 'total time': t_diff})

writer.close()
torch.save(segclip.state_dict(), f'runs/{logdir}/model.pt')
# subprocess.run(['tensorboard', 'dev', 'upload', '--logdir', 'runs/'])