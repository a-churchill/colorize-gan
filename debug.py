import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as utils
from torch.utils.data import DataLoader

from gan import ColorizeGAN
from parameters import DEVICE


def plot_losses(losses, legend=None):
    """Plots the losses over each epoch"""
    if legend is None:
        legend = ["real_loss", "fake_loss", "gan_loss", "l1_loss", "saturation_loss"]
    plt.figure(figsize=(8, 8))
    plt.title("Losses")
    plt.plot(losses)
    plt.legend(legend)
    plt.show()


def preview_images(image_batch: torch.Tensor, title="Preview Images", grid_size=5):
    """Previews a grid of images in the batch passed

    Args:
        image_batch (torch.Tensor): a batch of images to preview
    """
    image_batch_np: np.ndarray = image_batch.cpu()[: grid_size ** 2].detach().permute(
        (0, 2, 3, 1)
    ).mul(255).numpy().astype(np.uint8)

    converted_images = torch.from_numpy(
        np.array(
            [
                cv2.cvtColor(image_batch_np[i], cv2.COLOR_LAB2RGB)
                for i in range(image_batch_np.shape[0])
            ]
        )
    ).permute((0, 3, 1, 2))

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(
        np.transpose(
            utils.make_grid(converted_images, nrow=grid_size, padding=2), (1, 2, 0),
        )
    )
    return fig


def run_model(model: ColorizeGAN, dataloader: DataLoader):
    """Runs the model on the given dataloader.

    Args:
        model (ColorizeGAN): trained model
        dataloader (DataLoader): data loader to load test data (should be different from training 
        data)
    """
    model.eval()
    batch, _ = next(iter(dataloader))
    batch = batch.to(DEVICE)
    return preview_images(
        torch.cat((batch[:32], model(batch)[:32]), dim=0), "Results", 8
    )

