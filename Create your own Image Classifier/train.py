import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import model_functions
# Command Line arguments
parser = argparse.ArgumentParser(description='Train classifier for flower identification')
parser.add_argument('data_dir', nargs='?', default="./flowers/")
parser.add_argument('--path', dest="path", default="checkpoints", help="path to save model checkpoint")

# Model-specific settings
parser.add_argument('--arch', dest="arch", default="densenet121", help="model architecture to use for training")
parser.add_argument('--learning_rate', dest="learning_rate", default=0.001, type=float)
parser.add_argument('--hidden_units', nargs='+', dest="hidden_units", type=int, default=[500, 200], help="one or more argument values for each layer: --hidden_units 500 200")
parser.add_argument('--epochs', dest="epochs", default=1, type=int)
parser.add_argument('--output_size', dest="output_size", default=102, type=int)
parser.add_argument('--input_size', dest="input_size", default=1024, type=int)
parser.add_argument('--drop_out', dest="drop_out", default=0.95, type=float, help="dropout probability for the model")

# Enable GPU?
parser.add_argument('--gpu', dest="gpu", action='store_true', help="enable GPU mode, otherwise use CPU")

args = parser.parse_args()

# Load the model and perform training
model_functions.load_checkpoint(args)
