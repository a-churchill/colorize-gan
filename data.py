import os
import cv2
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from parameters import BATCH_SIZE, DATA_DIRECTORY, IMAGE_SIZE, WORKERS


def get_dataloader(dataset: ImageFolder) -> DataLoader:
    """Gets a dataloader for the given data set"""
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True,
    )


def rgb_to_lab_tensor(pic: Image.Image) -> torch.Tensor:
    """Converts an RGB tensor to LAB, then returns the tensor. Implementation based on
    `to_tensor`
    [here](https://pytorch.org/vision/stable/_modules/torchvision/transforms/functional.html#to_tensor)
    """

    img = np.array(pic, np.uint8, copy=True)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    tensor = torch.from_numpy(img)
    tensor = tensor.view(pic.size[1], pic.size[0], 3)

    # move channel dimension to the front
    tensor = tensor.permute((2, 0, 1)).contiguous()

    # convert to float
    tensor = tensor.to(dtype=torch.get_default_dtype()).div(255)

    return tensor


def get_dataset(train: bool) -> ImageFolder:
    """Gets the dataset, and sets up the transforms"""
    return ImageFolder(
        os.path.join(DATA_DIRECTORY, "train" if train else "test"),
        transform=transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.Lambda(rgb_to_lab_tensor),
                # transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
