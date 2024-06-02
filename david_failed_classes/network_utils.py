import torch
import torch.nn as nn
from collections import OrderedDict

def build_mlp(input_size, output_size, n_layers, hidden_size, activation=nn.ReLU()):
    """ Build multi-layer perception with n_layers hidden layers of size hidden_size """
    layers = [nn.Linear(input_size,hidden_size),activation]
    for i in range(n_layers-1):
        layers.append(nn.Linear(hidden_size,hidden_size))
        layers.append(activation)
    layers.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers).to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x
