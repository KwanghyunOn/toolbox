import torch
from torchvision import transforms as T
import einops

from . import paired_transforms as PT


class SRTransform(PT.Compose):
    def __init__(self, train, patch_size=None):
        transforms = []
        if train:
            if patch_size is not None:
                transforms.append(PT.RandomCrop(size=patch_size))
            transforms.extend([
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
                PT.RandomHorizontalFlip(),
            ]
        else:
            if patch_size is not None:
                transforms.append(PT.CentorCrop(size=patch_size))
            transforms.extend([
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ])
        super().__init__(transforms)
