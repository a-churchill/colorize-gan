import torch
import torch.nn as nn
import torch.nn.functional as F

from discriminator import Discriminator
from generator import Generator
from parameters import ADAM_BETA, DEVICE, LAMBDA_L1, LEARNING_RATE


def init_weights(model):
    """Custom weight initialization for generator and discriminator. Code from
    [Pytorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
    """
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def prediction_loss(prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
    """Computes loss for a set of predictions (single-element tensors)."""
    labels = torch.tensor([1.0 if is_real else 0.0], device=DEVICE).expand_as(
        prediction
    )
    return F.mse_loss(prediction, labels)


def saturation(batch: torch.Tensor) -> torch.Tensor:
    """Computes the saturation of each pixel of each image, as explained
    [here](https://knowledge.ulprospector.com/10780/pc-the-cielab-lab-system-the-method-to-quantify-colors-of-coatings/)

    Args:
        batch (torch.Tensor): a batch of images (CxWxH) in the LAB color space

    Returns:
        torch.Tensor: a batch of 1-channel images with saturation values
    """
    a_channel = batch[:, 1, :, :]
    b_channel = batch[:, 2, :, :]
    return torch.sqrt(torch.square(a_channel) + torch.square(b_channel))


def set_requires_grad(net: nn.Module, requires_grad=False):
    """Saves computation by avoiding computing gradient unless necessary. Credit to
    [this repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for
    implementation inspiration.

    Args:
        nets (List[nn.Module]): nets to update
        requires_grad (bool, optional): whether `nets` require gradient. Defaults to False.
    """
    for param in net.parameters():
        param.requires_grad = requires_grad


class ColorizeGAN(nn.Module):
    """GAN for colorizing images."""

    def __init__(self) -> None:
        super().__init__()

        self.generator = Generator()
        self.generator.apply(init_weights)
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(), lr=LEARNING_RATE, betas=(ADAM_BETA, 0.999)
        )

        self.discriminator = Discriminator()
        self.discriminator.apply(init_weights)
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=LEARNING_RATE, betas=(ADAM_BETA, 0.999)
        )

        # grayscale images
        self.input = None

        # original colorized images
        self.real_output = None

        # generated colorized images
        self.fake_output = None

        # losses, for reporting
        self.fake_loss = None
        self.real_loss = None
        self.l1_loss = None
        self.gan_loss = None
        self.saturation_loss = None

    def forward(self, original_images: torch.Tensor) -> torch.Tensor:
        """Generates colorized images from original images (which are already colorized)"""
        self.input = original_images[:, 0:1, :, :]  # use grayscale channel only
        self.real_output = original_images
        self.fake_output = self.generator(self.input)

        return self.fake_output

    def backward(self):
        """Updates generator and discriminator weights"""
        # make sure forward already ran
        if self.fake_output is None:
            raise Exception(
                "fake_output not generated, make sure to run forward() before backward()!"
            )

        # update discriminator
        set_requires_grad(self.discriminator, True)
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()

        # update generator
        set_requires_grad(self.discriminator, False)
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # reset input until next forward call
        self.input = None
        self.real_output = None
        self.fake_output = None

    def backward_d(self):
        """Computes loss and gradients for discriminator network"""
        # include input because this is a conditional GAN
        input_d_fake = torch.cat((self.input, self.fake_output), 1)
        input_d_real = torch.cat((self.input, self.real_output), 1)

        # run discriminator (with detached input, to make sure generator is not affected)
        output_d_fake = self.discriminator(input_d_fake.detach())

        # compute loss for fake batch
        self.fake_loss = prediction_loss(output_d_fake, False)

        # run discriminator (now with real images)
        output_d_real = self.discriminator(input_d_real)

        # compute loss for real batch
        self.real_loss = prediction_loss(output_d_real, True)

        # combine loss, compute gradients
        loss = (self.fake_loss + self.real_loss) * 0.5
        loss.backward()

    def backward_g(self):
        """Computes loss and gradients for generator network"""
        # run loss on discriminator results, pretending they are real
        output_d_fake = self.discriminator(torch.cat((self.input, self.fake_output), 1))
        self.gan_loss = prediction_loss(output_d_fake, True)

        # run loss on output directly
        self.l1_loss = F.l1_loss(self.fake_output, self.real_output) * LAMBDA_L1

        self.saturation_loss = (
            F.l1_loss(saturation(self.fake_output), saturation(self.real_output))
            * LAMBDA_L1
        )

        # combine losses, compute gradients
        loss = self.gan_loss + self.l1_loss + self.saturation_loss
        loss.backward()

    def report_losses(self):
        """Logs the current losses"""
        print("Current loss values:\n")
        print(f"\treal: {self.real_loss}")
        print(f"\tfake: {self.fake_loss}")
        print(f"\tgan: {self.gan_loss}")
        print(f"\tl1: {self.l1_loss}")
        print(f"\tsaturation: {self.saturation_loss}")
        return [
            self.real_loss.item(),
            self.fake_loss.item(),
            self.gan_loss.item(),
            self.l1_loss.item(),
            self.saturation_loss.item(),
        ]
