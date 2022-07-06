import model
import clip
import torch
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

segclip, preproc, preproc_lbl = model.load_custom_clip('RN50', num_classes=5, device=device)

img = torch.randn(4, 3, 224, 224, device=device) # batch_size=4
# since we are taking a tensor of the reqd shape directly we dont need to call preproc

classes = ['dog','samoyed','cat','chair','ice cream'] # the 5 classes
text_tokens = clip.tokenize(classes).to(device)

# this will give us the output feature block
output = segclip(img, text_tokens)
print(f'Output of model:\t {output.shape}') # [batch_size, num_classes, 224, 224]

# we can take a softmax and find which class has highest prob
probs = F.softmax(output, dim=1)
print(f'Probabilities shape:\t {probs.shape}') # [batch_size, num_classes, 224, 224]

mask = probs.argmax(dim=1)
print(f'Final Mask Shape:\t {mask.shape}') # [batch_size, 224, 224]