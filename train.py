from torch.utils.data import DataLoader
from tqdm import tqdm
from debug import plot_losses
from gan import ColorizeGAN

from parameters import DEVICE, NUM_EPOCHS


def train(model: ColorizeGAN, dataloader: DataLoader):
    """Trains `model` with the given data."""
    model = model.to(DEVICE)
    model.train()

    losses = []
    for epoch in range(NUM_EPOCHS):
        print("\nEPOCH", epoch + 1)
        for data, _ in tqdm(dataloader):
            data = data.to(DEVICE)
            model.forward(data)
            model.backward()

        # debug
        losses.append(model.report_losses())
        # preview_images(model(data), f"Epoch {epoch + 1}")

    plot_losses(losses)
