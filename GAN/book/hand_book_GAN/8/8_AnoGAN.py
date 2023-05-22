import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F 
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision
import glob
import os
from PIL import Image
import numpy as np
import options
import itertools
import cv2


#%%p335
# データはfruit 360 を使用
# https://www.kaggle.com/datasets/moltean/fruits

#%%
def add_damage(image_path):

    folder = os.path.dirname(image_path)
    save_folder = folder + "_damaged"
    save_base_path = os.path.basename(image_path).replace('.jpg', "_damaged.jpg")
    save_path = os.paath.join(save_folder, save_base_path)

    os.makdirs(save_folder, exist_ok=True)

    image = cv2.imread(image_path)
    center_x = np.random.randint(20, 76)
    center_y = np.random.randint(20, 76)
    color_r = np.random.randint(0, 255)
    color_g = np.random.randint(0, 255)
    color_b = np.random.randint(0, 255)

    center = (center_x, center_y)
    color = (color_r, color_g, color_b)
    cv2.circle(image, center = center, radius = 10, color = color, thickness = -1)
    cv2.imwrite(save_path, image)

#%%p339
IMAGE_SIZE = 96
EMBED_SIZE = 128
BATCH_SIZE = 16
LR = 0.0004
EPOCHS = 1000
DEVICE = 'mps'
kwargs = {'num_workers': 1, 'pin_memory': True}



