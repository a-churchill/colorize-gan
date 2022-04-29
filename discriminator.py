from typing import List

import torch
import torch.nn as nn

from parameters import CONV_CHANNELS


class Discriminator(nn.Module):
    """Discriminator to determine if an image is appropriately colorized."""

    def __init__(self) -> None:
        super(Discriminator, self).__init__()

        layers: List[nn.Module] = [
            nn.Conv2d(3 + 1, CONV_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]

        for layer in range(1, 3):
            filter_multiplier_prev = 2 ** (layer - 1)
            filter_multiplier = 2 ** layer
            layers.extend(
                [
                    nn.Conv2d(
                        CONV_CHANNELS * filter_multiplier_prev,
                        CONV_CHANNELS * filter_multiplier,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(CONV_CHANNELS * filter_multiplier,),
                    nn.LeakyReLU(0.2, True),
                ]
            )

        layers.append(
            nn.Conv2d(
                CONV_CHANNELS * filter_multiplier,
                1,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Performs forward propagation of a batch of data through the network."""
        return self.model(x)
