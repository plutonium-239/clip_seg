from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import argparse
import plotly.express as px
import torch

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

parser = argparse.ArgumentParser()

parser.add_argument('--runid', type=int, help='which run to use')
parser.add_argument('--log', default=0, type=int, help='log to tensorboard? 0/1')
parser.add_argument('--plotly', default=1, type=int, help='use plotly? 0/1')

args, _ = parser.parse_known_args()

logdir = f"run_{args.runid}"

if args.log:
	writer = SummaryWriter('fsimages/'+logdir)

conf = torch.load('fsimages/'+logdir+'/conf.pt', map_location='cpu')
conf_approx = torch.load('fsimages/'+logdir+'/conf_approx.pt', map_location='cpu')[1:, 1:]
val_conf = torch.load('fsimages/'+logdir+'/val_conf.pt', map_location='cpu')
val_conf_approx = torch.load('fsimages/'+logdir+'/val_conf_approx.pt', map_location='cpu')[1:, 1:]

newconf = torch.zeros_like(conf)
newconf[6:21, 6:21] = conf[1:16,1:16]
newconf[0, 6:] = conf[0, 1:16]
newconf[6:,0] = conf[1:16, 0]
newconf[0,0] = conf[0,0]

conf = torch.nn.functional.normalize(newconf, dim=1)
val_conf = torch.nn.functional.normalize(val_conf, dim=1)

pascal_classes = ['background'] + pascal_classes

if args.plotly:
	# px.imshow(conf, x=pascal_classes, y=pascal_classes).show()
	px.imshow(conf, x=pascal_classes, y=pascal_classes, text_auto=True).show()
	px.imshow(val_conf, x=pascal_classes, y=pascal_classes, text_auto=True).show()
else:
	heatmap(conf, pascal_classes, pascal_classes)
	plt.show()
	heatmap(conf, pascal_classes, pascal_classes, cmap='plasma')
	plt.show()
	heatmap(conf, pascal_classes, pascal_classes, cmap='PuRd')
	plt.show()
	heatmap(conf, pascal_classes, pascal_classes, cmap='YlGnBu')
	plt.show()
	heatmap(conf, pascal_classes, pascal_classes, cmap='cool')
	plt.show()