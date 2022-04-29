import sys

import torch


if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

# beta value for Adam optimizer
ADAM_BETA = 0.5

# Batch size during training
BATCH_SIZE = 128

# convolutional layers per level of our generator and discriminator
CONV_CHANNELS = 64

# location of data on disk
DATA_DIRECTORY = "data"

# GPU device to run training on
DEVICE = torch.device("cuda:0")

# image size to use within the network
IMAGE_SIZE = 128

# weight for l1 loss when computing loss for generator model
LAMBDA_L1 = 100.0

# learning rate for optimizers
LEARNING_RATE = 0.0002

# number of training epochs
NUM_EPOCHS = 20

# number of workers to use for the dataloader
WORKERS = 4
