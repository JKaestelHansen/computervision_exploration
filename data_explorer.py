# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import tifffile as tiff
from natsort import natsorted

"""
Interestingly by eye it looks like cells divide around the same time when there were just 2

cells can really really elongate but less so the more cells there are
"""

data_path = '/Users/bzx569/Documents/PostDoc/Work/cell_tracking_segmentation/cell_data/'
exp_path = 'BF-C2DL-MuSC/01'
GT_path = 'BF-C2DL-MuSC/01_GT/TRA'
ST_path = 'BF-C2DL-MuSC/01_ST'
ERR_SEG_path = 'BF-C2DL-MuSC/01_ERR_SEG'

exp_files = os.listdir(os.path.join(data_path, exp_path))
exp_files = natsorted(exp_files)

GT_files = os.listdir(os.path.join(data_path, GT_path))
GT_files = natsorted(GT_files)

ERR_SEG_files = os.listdir(os.path.join(data_path, ERR_SEG_path))
ERR_SEG_files = natsorted(ERR_SEG_files)

print(exp_files)
print(GT_files)
print(ERR_SEG_files)

i = 1300
fig, ax = plt.subplots(1,3,figsize=(10, 8))
img = tiff.imread(os.path.join(data_path, exp_path, exp_files[i]))
ax[0].imshow(img, cmap='gray')
ax[0].set_title(f'Frame {i}')

img = tiff.imread(os.path.join(data_path, GT_path, GT_files[i]))
print(np.unique(img))
ax[1].imshow(img, cmap='gray')
ax[1].set_title(f'Frame {i}')

img = tiff.imread(os.path.join(data_path, ERR_SEG_path, ERR_SEG_files[i]))
print(np.unique(img))
ax[2].imshow(img, cmap='gray')
ax[2].set_title(f'Frame {i} ERR_SEG')
plt.show()


i = 1299
fig, ax = plt.subplots(1,3,figsize=(10, 8))
img = tiff.imread(os.path.join(data_path, exp_path, exp_files[i]))
ax[0].imshow(img, cmap='gray')
ax[0].set_title(f'Frame {i}')

img = tiff.imread(os.path.join(data_path, GT_path, GT_files[i]))
print(np.unique(img))
ax[1].imshow(img, cmap='gray')
ax[1].set_title(f'Frame {i}')

img = tiff.imread(os.path.join(data_path, ERR_SEG_path, ERR_SEG_files[i]))
print(np.unique(img))
ax[2].imshow(img, cmap='gray')
ax[2].set_title(f'Frame {i} ERR_SEG')
plt.show()


# make cmap from hsv but 0 is transparent
my_cmap = plt.cm.hsv
my_cmap.set_under('black', alpha=0)  # Set color for values below vmin to transparent


i = np.random.randint(0, len(exp_files))
fig, ax = plt.subplots(figsize=(10, 8))
img = tiff.imread(os.path.join(data_path, exp_path, exp_files[i]))
ax.imshow(img, cmap='gray')
ax.set_title(f'Frame {i}')

img = tiff.imread(os.path.join(data_path, ERR_SEG_path, ERR_SEG_files[i]))
print(np.unique(img))
ax.imshow(img, cmap=my_cmap, alpha=0.1, vmin=0.1)
ax.set_title(f'Frame {i} ERR_SEG')
plt.show()


# %%

torch.set_default_device('mps')
print('torch.mps.is_available()', torch.mps.is_available())
print(torch.Generator(device='mps').device)