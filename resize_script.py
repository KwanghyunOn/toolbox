import os
import glob
from tqdm import tqdm

import torch
import torchvision

from resize import resize


root = "/home/kwanghyunon/data/DIV2K/"
src_dir = "DIV2K_valid_LR_bicubic/X4"
save_dir = "DIV2K_valid_LR_bicubic/X4_upsampled"
os.makedirs(os.path.join(root, save_dir), exist_ok=True)
scale = 4.0

img_paths = sorted(glob.glob(os.path.join(root, src_dir, "*.png")))
for img_path in tqdm(img_paths):
    name = os.path.basename(img_path)
    img = torchvision.io.read_image(img_path)
    img = img / 255.0
    img_resized = resize(img, scale_factors=scale).clamp(min=0.0, max=1.0)
    torchvision.utils.save_image(img_resized, os.path.join(root, save_dir, name))
