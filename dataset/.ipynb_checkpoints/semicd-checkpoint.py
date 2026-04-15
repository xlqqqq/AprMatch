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

### ADDED ###
# Import OpenCV for morphological operations
import cv2

### ADDED ###
def mask_to_boundary(mask, kernel_size=(3, 3)):
    """
    Convert a segmentation mask to a boundary mask.

    Args:
        mask (np.array): Input segmentation mask (H, W), with integer class labels.
        kernel_size (tuple): The size of the kernel for the erosion operation.

    Returns:
        torch.Tensor: A float tensor representing the boundary mask (H, W),
                      where 1 indicates a boundary pixel and 0 indicates a non-boundary pixel.
    """
    # Ensure the mask is a binary numpy array (0 for background, 1 for foreground)
    # This assumes class 0 is the background. Adjust if necessary.
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    # Define the erosion kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Erode the mask to shrink the foreground region
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # The boundary is the difference between the original mask and the eroded mask
    boundary = mask - eroded_mask

    # Convert to a float tensor for the loss function
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
            # Assumes val.txt is in a standard splits directory
            with open(os.path.join('splits', name, 'val.txt'), 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]

        # MODIFIED: 支持TIFF格式
        img = Image.open(os.path.join(self.root, 'img', id)).convert('RGB')

        base_name = os.path.splitext(id)[0]
        # MODIFIED: 掩码也支持TIFF格式
        mask_id = f"{base_name}.tiff"  # 改为.tiff格式
        mask_path = os.path.join(self.root, 'mask', mask_id)

        # 如果.tiff不存在，尝试其他格式
        if not os.path.exists(mask_path):
            for ext in ['.tif', '.png']:
                alt_mask_path = os.path.join(self.root, 'mask', f"{base_name}{ext}")
                if os.path.exists(alt_mask_path):
                    mask_path = alt_mask_path
                    break

        # Load mask as a PIL Image
        mask = Image.open(mask_path)

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        # Data augmentation
        img, mask = resize(img, mask, (0.8, 1.2))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        ### MODIFIED ###
        # The logic for labeled and unlabeled data is now distinct.
        if self.mode == 'train_l':
            # For labeled data, we need the image, segmentation mask, and the new boundary mask.

            # Convert PIL mask to numpy array to generate the boundary
            numpy_mask = np.array(mask)
            boundary_mask = mask_to_boundary(numpy_mask)

            # Normalize image and segmentation mask (converts them to tensors)
            img, mask = normalize(img, mask)

            return img, mask, boundary_mask

        # The rest of the function handles the 'train_u' case
        img_w = deepcopy(img)   # Weakly augmented image
        img_s1 = deepcopy(img)  # First strongly augmented image
        img_s2 = deepcopy(img)  # Second strongly augmented image

        # Apply strong augmentation to img_s1
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        # Apply strong augmentation to img_s2
        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        # Create ignore_mask for unsupervised loss from the original mask
        ignore_mask = torch.from_numpy(np.array(mask)).long()

        return normalize(img_w), normalize(img_s1), normalize(img_s2), \
               ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)