from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2


# 转换成边界掩码
def mask_to_boundary(mask, kernel_size=(3, 3)):
    mask = (mask > 0).astype(np.uint8) if mask.max() > 1 else mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    boundary = mask - eroded_mask
    return torch.from_numpy(boundary).float()


class SemiSegDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(os.path.join('splits', name, 'val.txt'), 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]

        img = Image.open(os.path.join(self.root, 'img', id)).convert('RGB')

        base_name = os.path.splitext(id)[0]
        mask_path = os.path.join(self.root, 'mask', f"{base_name}.tiff")
        if not os.path.exists(mask_path):
            for ext in ['.tif', '.png']:
                alt = os.path.join(self.root, 'mask', f"{base_name}{ext}")
                if os.path.exists(alt):
                    mask_path = alt
                    break

        mask = Image.open(mask_path)

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        img, mask = resize(img, mask, (0.8, 1.2))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            boundary_mask = mask_to_boundary(np.array(mask))
            img, mask = normalize(img, mask)
            return img, mask, boundary_mask

        # train_u
        img_w  = deepcopy(img)
        img_s1 = deepcopy(img)
        img_s2 = deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = torch.from_numpy(np.array(mask)).long()

        return normalize(img_w), normalize(img_s1), normalize(img_s2), \
               ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
