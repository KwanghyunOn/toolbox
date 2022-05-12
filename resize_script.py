import os
import glob
import blobfile as bf
from tqdm import tqdm

import torch
import torchvision

from resize import resize


def main():
    root = "/home/kwanghyunon/data/FFHQ/"
    src_dir = "thumbnails128x128"
    save_dir = "bicubic32x32"
    os.makedirs(os.path.join(root, save_dir), exist_ok=True)
    scale = 0.25

    img_paths = _list_image_files_recursively(os.path.join(root, src_dir))
    for img_path in tqdm(img_paths):
        abs_dir, filename = os.path.split(img_path)
        rel_dir = os.path.relpath(abs_dir, os.path.join(root, src_dir))
        abs_save_dir = os.path.join(root, save_dir, rel_dir)
        os.makedirs(abs_save_dir, exist_ok=True)

        img = torchvision.io.read_image(img_path)
        img = img / 255.0
        img_resized = resize(img, scale_factors=scale).clamp(min=0.0, max=1.0)
        torchvision.utils.save_image(img_resized, os.path.join(abs_save_dir, filename))


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


if __name__ == "__main__":
    main()