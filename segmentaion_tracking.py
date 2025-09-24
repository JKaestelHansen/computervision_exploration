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

from ultralytics import YOLO

# load data
data_path = '/Users/bzx569/Documents/PostDoc/Work/cell_tracking_segmentation/cell_data/'
exp_path = 'BF-C2DL-MuSC/01'
GT_path = 'BF-C2DL-MuSC/01_GT/TRA'
ST_path = 'BF-C2DL-MuSC/01_ST'
ERR_SEG_path = 'BF-C2DL-MuSC/01_ERR_SEG'

yolo_path = os.path.join(data_path, exp_path+'_yolo_dataset', 'images')

exp_files = os.listdir(os.path.join(data_path, exp_path))
exp_files = natsorted(exp_files)

GT_files = os.listdir(os.path.join(data_path, GT_path))
GT_files = natsorted(GT_files)

ERR_SEG_files = os.listdir(os.path.join(data_path, ERR_SEG_path))
ERR_SEG_files = natsorted(ERR_SEG_files)

# Load a model
model = YOLO("yolo11n-seg.pt")  # load an official model

# Train the model
retrain = True
model_path = '/Users/bzx569/Documents/PostDoc/Work/cell_tracking_segmentation/yolo11n-seg_MuSC.pt'
if retrain:
    results = model.train(data=os.path.join(data_path, 'BF-C2DL-MuSC/cell_dataset.yaml'), epochs=1)
    model.save(model_path)
else:
    model = YOLO(model_path)  # load a custom model
# %%

results = model.predict(source=yolo_path, 
                        save=True, save_txt=True, save_conf=True, 
                        conf=0.25, iou=0.5, device='mps', stream=False,
                        augment=False, verbose=True)  # predict on an image

# %%
i = np.random.randint(0, len(results))
print(results[i])
orig = results[i].orig_img
mask = results[i].masks.data.cpu().numpy()

print(orig.shape)
print(mask.shape)

# make cmap from hsv but 0 is transparent
my_cmap = plt.cm.hsv
my_cmap.set_under('black', alpha=0)  # Set color for values below vmin to transparent

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(orig[:,:,2], cmap='gray')
ax.set_title(f'Frame {i}')

ax.imshow(np.max(mask, axis=0), cmap=my_cmap, alpha=0.1, vmin=0.1)
ax.set_title(f'Frame {i} ERR_SEG')
plt.show()

# %%
