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

for i, (batch_img, batch_lbl, texts) in pbar:
	batch_img, batch_lbl = batch_img.to(device), batch_lbl.to(device)
	text_tokens = clip.tokenize(texts).to(device)
	output = segclip(batch_img, text_tokens)
	loss = loss_fn(output, batch_lbl)
	loss.backward()
	optimiser.step()
	pbar.set_postfix({'loss':loss.item()})
	
