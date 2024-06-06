"""
set all utility functions! 
- import mpl from util in order to get the right colors
- import np from util in order to get the right print options
- use np2torch to convert np arrays to torch tensors
"""
import numpy as np
from glob import glob
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
#import matplotlib.ticker as mtick
import os
from scipy.optimize import curve_fit
import torch
import torch.nn as nn
import logging
#from torch.masked import masked_tensor

# blue = bids, red = asks
#           |   Blue  |   Red  |  Orange |  Purple | Yellow  |   Green |   Teal  | Grey
hexcolors = ['648FFF', 'DC267F', 'FE6100', '785EF0', 'FFB000', '009E73', '3DDBD9', '808080']
mpl.rcParams['axes.prop_cycle'] = cycler('color', [mpl.colors.to_rgba('#' + c) for c in hexcolors])

FIGSIZE = (10,6)
SAVEDIR = os.path.join(os.getcwd(), "plots")
os.makedirs(SAVEDIR, exist_ok=True)
SAVEEXT = ".png"
np.set_printoptions(precision=4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def np2torch(x, requires_grad=False, cast_double_to_float=True):
    #mask = 0
    #if isinstance(x, np.ma.masked_array):
    #    mask = torch.from_numpy(x.mask).to(device)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = torch.Tensor(x).to(device)
    if cast_double_to_float and x.dtype == torch.float64:
        x = x.float()  # cast double to float
    if requires_grad:
        x = x.float()
        x.requires_grad = True
    return x

def build_mlp(input_size, output_size, n_layers, hidden_size, activation=nn.ReLU()):
    """ Build multi-layer perception with n_layers hidden layers of size hidden_size """
    layers = [nn.Linear(input_size,hidden_size),activation]
    for i in range(n_layers-1):
        layers.append(nn.Linear(hidden_size,hidden_size))
        layers.append(activation)
    layers.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers).to(device)

def get_logger(filename):
    """ Return a logger instance to a file """
    if os.path.exists(filename):
        os.remove(filename)
    logger = logging.getLogger("LOG")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(filename=filename, 
                        filemode = 'a',  # add to existing config
                        format='%(asctime)s:%(levelname)s:%(message)s', 
                        datefmt='%m-%d %H:%M',
                        level=logging.DEBUG)
    # LOL was doing everything twice oops
    #handler = logging.FileHandler(filename)
    #handler.setLevel(logging.DEBUG)
    #handler.setFormatter(logging.Formatter('%(asctime)s:%(name)-12s:%(levelname)-8s:%(message)s'))
    #logger.addHandler(handler)
    return logger

def normalize(x):
    """ Normalize np.ndarray or torch.Tensor """
    return (x - x.mean()) / (x.std() + 1e-10)

def list_to_masked(arrays: list[list]):
    """
    Returns masked (2d) array from list of arrays of potentially different lengths
    """
    lens = [len(arr) for arr in arrays]
    all_arr = np.ma.empty( (len(arrays), max(lens)) )
    all_arr.mask = True
    for idx, arr in enumerate(arrays):
        all_arr[idx, :lens[idx]] = np.array(arr)
    return all_arr

def plot_WIM(wealth: np.ndarray, inventory: np.ndarray, midprices: np.ndarray, dt: float, title='', savename=''):
    """ plot data from a batch of trajectories
    Inputs: (nbatch x nt) np.ndarrays """
    times = np.arange(0, wealth.shape[-1]*dt, dt)
    fig, axs = plt.subplots(2,2, figsize=(12,10))
    value =  wealth + inventory*midprices
    c = 0
    for i, y, name in zip(((0,0),(0,1),(1,0),(1,1)),(wealth, inventory, midprices,value),('Wealth', 'Inventory', 'Midprice', 'Value')):
        ax = axs[i]
        ax.set(ylabel=name)
        ys = y.mean(axis = 0)
        yerrs = y.std(axis = 0)/np.sqrt(y.shape[-1])
        ax.fill_between(times, ys - yerrs, ys + yerrs, alpha=0.25, color=f"C{c}")
        ax.plot(times, ys, color=f"C{c}")
    axs[0,1].set(xlabel="Time")
    axs[1,1].set(xlabel='Time')
    if title: fig.suptitle(title)
    if savename: plt.savefig(savename)
    plt.close()
    #plt.show()

def plot_WIM_2(paths, dt: float, title='', savename=''):
    """ plot data from a batch of trajectories
    Inputs: (nbatch x nt) np.ndarrays """
    wealths = [p['wea'] for p in paths]
    inventories = [p['inv'] for p in paths]
    midprices = [p['mid'] for p in paths]
    wealth = list_to_masked(wealths)
    inventory = list_to_masked(inventories)
    midprice = list_to_masked(midprices)
    print(wealth.shape, inventory.shape, midprice.shape)
    times = np.arange(0, wealth.shape[-1]*dt, dt)
    fig, axs = plt.subplots(2,2, figsize=(12,10), sharex=True)
    value = wealth + inventory*midprice
    c = 0
    for i, y, name in zip(((0,0),(0,1),(1,0),(1,1)),(wealth, inventory, midprice, value),('Wealth', 'Inventory', 'Midprice','Value')):
        ax = axs[i]
        ax.set(ylabel=name)
        ys = y.mean(axis = 0)
        yerrs = y.std(axis = 0)/np.sqrt(y.shape[-1])
        ax.fill_between(times, ys - yerrs, ys + yerrs, alpha=0.25, color=f"C{c}")
        ax.plot(times, ys, color=f"C{c}")
        c += 1
    axs[0,1].set(xlabel="Time")
    axs[1,1].set(xlabel="Time")
    if title: fig.suptitle(title)
    if savename: fig.savefig(savename)
    plt.close()

def export_plot(y, ylabel, title, filename):
    """ plot epochs. """
    fig, ax = plt.subplots(figsize=(10,8))
    times = np.arange(0, y.shape[0])
    ys = y.mean(axis = -1)
    yerrs = y.std(axis = -1)/np.sqrt(y.shape[-1])
    ax.fill_between(times, ys - yerrs, ys + yerrs, alpha=0.25)
    ax.plot(times, ys)
    ax.set(xlabel='Training Episode',ylabel=ylabel)
    ax.set_title(title)
    plt.savefig(filename)
    plt.close()

""" Used for exponential value functions """
class Exponential(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return -abs(self.a) * torch.exp(-abs(self.b) * x)

    def string(self):
        return f'y = -{abs(self.a.item())} * exp(-{abs(self.b.item())} x)'


def uFormat(number, uncertainty, round = 0, sig_figs = 4, FormatDecimals = False):
    """
    V 3.0
    Returns "num_rounded(with_sgnfcnt_dgts_ofuncrtnty)", formatted to 10^round
    According to section 5.3 of "https://pdg.lbl.gov/2011/reviews/rpp2011-rev-rpp-intro.pdf"

    Arguments:
    - float number:      the value
    - float uncertainty: the absolute uncertainty (stddev) in the value
       - if zero, will format number to optional number of sig_figs (see sig_figs)
    - int round:  optionally, shift the resultant number to a higher/lower digit expression
       - i.e. if number is in Hz and you want a string in GHz, specify round = 9
               likewise for going from MHz to Hz, specify round = -6
    - int sig_figs: when uncertainty = 0, format number to degree of sig figs instead
       - if zero, will simply return number as string
    - bool FormatDecimals:  for a number 0.00X < 1e-2, option to express in "X.XXe-D" format
             for conciseness. doesnt work in math mode because '-' is taken as minus sign
    """
    num = str(number); err = str(uncertainty)
    
    sigFigsMode = not uncertainty    # UNCERTAINTY ZERO: IN SIG FIGS MODE
    if sigFigsMode and not sig_figs: # nothing to format
        return num
    
    negative = False  # add back negative later
    if num[0] == '-':
        num = num[1:]
        negative = True
    if err[0] == '-':
        err = err[1:]
    
    # ni = NUM DIGITS to the RIGHT of DECIMAL
    # 0.00001234=1.234e-4 has ni = 8, 4 digs after decimal and 4 sig figs
    # 1234 w/ ni=5 corresponds to 0.01234
    ni = ei = 0  
    if 'e' in num:
        ff = num.split('e')
        num = ff[0]
        ni = -int(ff[1])
    if 'e' in err:
        ff = err.split('e')
        err = ff[0]
        ei = -int(ff[1])

    if not num[0].isdigit():
        print(f"uFormat: {num} isn't a number")
        return num
    if not err[0].isdigit():
        err = '?'

    # comb through error, get three most significant figs
    foundSig = False; decimal = False
    topThree = ""; numFound = 0
    jErr = ""
    for ch in err:
        if decimal:
            ei += 1
        if not foundSig and ch == '0': # dont care ab leading zeroes
            continue  
        if ch == '.':
            decimal = True
            continue
        jErr += ch
        if numFound >= 3:  # get place only to three sigfigs
            ei -= 1
            continue
        foundSig = True
        topThree += ch
        numFound += 1
    
    foundSig = False; decimal = False
    jNum = ""
    for ch in num:
        if decimal:
            ni += 1
        if not foundSig and ch == '0': # dont care ab leading zeroes
            continue  
        if ch == '.':
            decimal = True
            continue
        jNum += ch
        foundSig = True
    
    # round error correctly according to PDG
    if len(topThree) == 3:
        nTop = int(topThree)
        if nTop < 355: # 123 -> (12.)
            Err = int(topThree[:2])
            if int(topThree[2]) >= 5:
                Err += 1
            ei -= 1
        elif nTop > 949: # 950 -> (10..)
            Err = 10
            ei -= 2
        else:  # 355 -> (4..)
            Err = int(topThree[0])
            if int(topThree[1]) >= 5:
                Err += 1
            ei -= 2
        Err = str(Err)
    else:
        Err = topThree

    n = len(jNum); m = len(Err)
    nBefore = ni - n  #; print(num, jNum, n, ni, nBefore)
    eBefore = ei - m  #; print(err, Err, m, ei, eBefore)
    if nBefore > eBefore:  # uncertainty is a magnitude larger than number, still format number
        if not sigFigsMode:
            print(f'Uncrtnty: {uncertainty} IS MAGNITUDE(S) > THAN Numba: {number}')
        Err = '?'
    if sigFigsMode or nBefore > eBefore:
        ei = nBefore + sig_figs

    # round number to error
    d = ni - ei 
    if ni == ei: 
        Num = jNum[:n-d]
    elif d > 0:  # error has smaller digits than number = round number
        Num = int(jNum[:n-d])
        if int(jNum[n-d]) >= 5:
            Num += 1
        Num = str(Num)
    else:  # error << num
        Num = jNum
        if ei < m + ni:
            Err = Err[n+d-1]
        else:
            Err = '0'
    if ni >= ei: ni = ei  # indicate number has been rounded
    
    n = len(Num)
    # if were at <= e-3 == 0.009, save formatting space by removing decimal zeroes
    extraDigs = 0
    if not round and FormatDecimals and (ni-n) >= 2:
        round -= ni - n + 1
        extraDigs = ni - n + 1
    
    # shift digits up/down by round argument
    ni += round
    end = ''
    if ni >= n:   # place decimal before any digits
        Num = '0.' + "0"*(ni-n) + Num
    elif ni > 0:  # place decimal in-between digits
        Num = Num[:n-ni] + '.' + Num[n-ni:]
    elif ni < 0:  # add non-significant zeroes after number
        end = 'e'+str(-ni)
    if extraDigs:  # format removed decimal zeroes
        end = 'e'+str(-extraDigs)
    
    if negative: Num = '-' + Num  # add back negative
    if not sigFigsMode:
        end = '(' + Err + ')' + end
    return Num + end