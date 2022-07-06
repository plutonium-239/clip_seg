import model
import clip
import torch
import torch.nn.functional as F
from pascal_voc_loader import pascalVOCLoader
from config import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

segclip, preproc, preproc_lbl = model.load_custom_clip('RN50', num_classes=5, device=device)

dataloader = pascalVOCLoader(config['pascal_root'], preproc, preproc_lbl, split=config['mode'], img_size=224, is_transform=True)
