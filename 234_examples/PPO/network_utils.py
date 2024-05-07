import torch
import torch.nn as nn
from collections import OrderedDict

def build_mlp(input_size, output_size, n_layers, size):
    """
    Args:
        input_size: int, the dimension of inputs to be given to the network
        output_size: int, the dimension of the output
        n_layers: int, the number of hidden layers of the network
        size: int, the size of each hidden layer
    Returns:
        An instance of (a subclass of) nn.Module representing the network.

    TODO:
    Build a feed-forward network (multi-layer perceptron, or mlp) that maps
    input_size-dimensional vectors to output_size-dimensional vectors.
    It should have 'n_layers' layers, each of 'size' units and followed
    by a ReLU nonlinearity. Additionally, the final layer should be linear (no ReLU).

    That is, the network architecture should be the following:
    [LINEAR LAYER]_1 -> [RELU] -> [LINEAR LAYER]_2 -> ... -> [LINEAR LAYER]_n -> [RELU] -> [LINEAR LAYER]

    "nn.Linear" and "nn.Sequential" may be helpful.
    """
    #######################################################
    #########   YOUR CODE HERE - 7-15 lines.   ############
    layers = [('0',nn.Linear(input_size,size)),('1',nn.ReLU())]
    for i in range(n_layers-1):
        layers.append((f"{2*i+2}",nn.Linear(size,size)))
        layers.append((f"{2*i+3}",nn.ReLU()))
    layers.append((f"{2*n_layers}",nn.Linear(size, output_size)))
    model = nn.Sequential(OrderedDict(layers))
    return model
    #######################################################
    #########          END YOUR CODE.          ############


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
