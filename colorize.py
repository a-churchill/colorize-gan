import torch

from data import get_dataloader, get_dataset
from debug import preview_images
from gan import ColorizeGAN
from parameters import DEVICE
from train import train


def main():
    """
    Entrypoint for colorize
    """
    dataloader = get_dataloader(get_dataset())

    model = ColorizeGAN()

    train(model, dataloader)

    batch, _ = next(iter(dataloader))
    batch = batch.to(DEVICE)
    preview_images(torch.cat((batch[:32], model(batch)[:32]), dim=0), "Results", 8)


if __name__ == "__main__":
    # solves an error being thrown on Windows without this line
    torch.multiprocessing.freeze_support()
    main()
