import os
from typing import Union
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from debug import plot_losses, run_model
from gan import ColorizeGAN

from parameters import DEVICE, NUM_EPOCHS

FOLDER = "gan_1"


def train(
    model: ColorizeGAN,
    dataloader: DataLoader,
    test_dataloader: DataLoader,
    writer: Union[SummaryWriter, None] = None,
):
    """Trains `model` with the given data."""
    model = model.to(DEVICE)
    model.train()
    os.makedirs(os.path.join(os.curdir, "checkpoints", FOLDER), exist_ok=True)

    losses = []
    for epoch in range(NUM_EPOCHS):
        print("\nEPOCH", epoch + 1)
        model.train()
        for data, _ in tqdm(dataloader):
            data = data.to(DEVICE)
            model.forward(data)
            model.backward()

        # debug
        losses.append(model.report_losses())
        writer.add_scalars(
            "Loss",
            {
                "fake": model.fake_loss,
                "real": model.real_loss,
                "gan": model.gan_loss,
                "saturation": model.saturation_loss,
                "l1": model.l1_loss,
            },
            epoch,
        )

        writer.add_figure("Sample", run_model(model, test_dataloader), epoch)
        writer.flush()

        path = os.path.join(os.curdir, "checkpoints", FOLDER, f"model_{epoch + 1}")
        torch.save(
            (
                model.state_dict(),
                model.optimizer_d.state_dict(),
                model.optimizer_g.state_dict(),
            ),
            path,
        )

    plot_losses(losses)
