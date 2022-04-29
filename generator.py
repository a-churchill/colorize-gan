from typing import Union

import torch
import torch.nn as nn
from typing_extensions import Self

from parameters import CONV_CHANNELS


class UNetLayer(nn.Module):
    """Creates a layer of the UNet (see
    [this paper](https://arxiv.org/pdf/1505.04597.pdf), page 2 for visual). Credit to
    [this repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for
    implementation inspiration.
    """

    def __init__(
        self,
        output_channels: int,
        internal_channels: int,
        input_channels: Union[int, None] = None,
        submodule: Union[Self, None] = None,
        outermost: bool = False,
    ) -> None:
        """Creates a U-Net layer, including a skip connection and a submodule.

        Args:
            output_channels (int): number of channels in the output from this layer
            internal_channels (int): number of channels to use internally in this layer
            input_channels (int): number of channels in the input to this layer.
            Defaults to `outer_channels` if `None`.
            submodule (Union[Self, None], optional): submodule used below this layer.
            Must have input set to `internal_channels`, output set to
            `internal_channels` (which means it will output `internal_channels * 2` due
            to the skip connection).
            Defaults to `None`. If `None`, will treat this as innermost layer.
            outermost (bool, optional): whether this is the outermost layer. Defaults to
            `False`.
        """
        super(UNetLayer, self).__init__()
        self.outermost = outermost
        innermost = submodule is None

        include_down_norm = not innermost and not outermost
        # accounts for skip connections
        up_conv_channels = internal_channels if innermost else internal_channels * 2

        input_channels = (
            input_channels if input_channels is not None else output_channels
        )

        self.model = nn.Sequential(
            # down
            nn.LeakyReLU(0.2, True) if not outermost else nn.Identity(),
            nn.Conv2d(
                input_channels, internal_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(internal_channels) if include_down_norm else nn.Identity(),
            submodule if submodule is not None else nn.Identity(),
            # up
            nn.ReLU(True),
            nn.ConvTranspose2d(
                up_conv_channels, output_channels, kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(output_channels) if not outermost else nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        """Performs forward propagation."""
        if self.outermost:
            return self.model(x)

        # add skip connection
        return torch.cat([x, self.model(x)], 1)


class Generator(nn.Module):
    """Generator to colorize images."""

    def __init__(self):
        super(Generator, self).__init__()
        # create innermost first
        innermost = UNetLayer(CONV_CHANNELS * 8, CONV_CHANNELS * 8)

        # gradually work from CONV_LAYERS * 8 to CONV_LAYERS
        unet_block = UNetLayer(
            CONV_CHANNELS * 8, CONV_CHANNELS * 8, submodule=innermost
        )
        unet_block = UNetLayer(
            CONV_CHANNELS * 4, CONV_CHANNELS * 8, submodule=unet_block
        )
        unet_block = UNetLayer(
            CONV_CHANNELS * 2, CONV_CHANNELS * 4, submodule=unet_block
        )
        unet_block = UNetLayer(CONV_CHANNELS, CONV_CHANNELS * 2, submodule=unet_block)

        # model has 1 input channel (grayscale), 3 output channels
        self.model = UNetLayer(
            3, CONV_CHANNELS, input_channels=1, submodule=unet_block, outermost=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward propagation of a batch of data through the network.

        Args:
            x (torch.Tensor): batched tensor of grayscale landscape images

        Returns:
            torch.Tensor: batched tensor of colorized landscape images
        """
        return self.model(x)
