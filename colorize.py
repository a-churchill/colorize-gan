import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from data import get_dataloader, get_dataset
from debug import run_model
from gan import ColorizeGAN
from train import train


def main(model_state_path=None):
    """
    Entrypoint for colorize
    """
    dataloader = get_dataloader(get_dataset(train=True))
    test_dataloader = get_dataloader(get_dataset(train=False))
    model = ColorizeGAN()
    writer = SummaryWriter("runs/colorize_gan_4")

    if model_state_path is None:
        train(model, dataloader, test_dataloader, writer)
    else:
        model_state, optimizer_d_state, optimizer_g_state = torch.load(model_state_path)
        model.load_state_dict(model_state)
        model.optimizer_d.load_state_dict(optimizer_d_state)
        model.optimizer_g.load_state_dict(optimizer_g_state)

    run_model(model, test_dataloader)
    plt.show()


if __name__ == "__main__":
    # solves an error being thrown on Windows without this line
    torch.multiprocessing.freeze_support()

    # main(os.path.join(os.curdir, "checkpoints", f"model_{9}"))
    main()
