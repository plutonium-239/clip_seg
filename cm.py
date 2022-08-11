from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import sys

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

log = 