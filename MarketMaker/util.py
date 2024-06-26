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
import torch.masked as masked
#from torch.masked import masked_tensor

# CONFIGURE MODULES
# blue = bids, red = asks
#           |   Blue  |   Red  |  Orange |  Purple | Yellow  |   Green |   Teal  | Grey
FIGSIZE = (10,9)
textsize = 20  # size of text for overleaf formatting

def reset_matplotlib():
    mpl.rcParams.update(mpl.rcParamsDefault)
    hexcolors = ['648FFF', 'DC267F', 'FE6100', '785EF0', 'FFB000', '009E73', '3DDBD9', '808080']
    mpl.rcParams['axes.prop_cycle'] = cycler('color', [mpl.colors.to_rgba('#' + c) for c in hexcolors])
    #mpl.rcParams['figure.figsize'] = FIGSIZE

SAVEDIR = os.path.join(os.getcwd(), "plots")
os.makedirs(SAVEDIR, exist_ok=True)
SAVEEXT = ".png"
np.set_printoptions(precision=4)


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_dtype(torch.float32)
elif torch.backends.mps.is_available():
    pass
    #device = "mps"

def np2torch(x, requires_grad=False, cast_double_to_float=True):
    ''' for some reason, numpy convention is to have mask=True 
    when the mask is on, while torch has mask=False when the mask 
    is on, so make sure to invert the mask between the two.'''
    if isinstance(x, np.ma.MaskedArray):
        mask = torch.from_numpy(~x.mask)
        x = torch.from_numpy(x.data)
        x = torch.masked.as_masked_tensor(x, mask)
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        x = torch.Tensor(x)
    if requires_grad:
        x = x.float()
        x.requires_grad = True
    elif cast_double_to_float and x.dtype == torch.float64:
        x = x.float()
    return x.to(device)

def torch2np(x: torch.Tensor | torch.masked.MaskedTensor):
    if isinstance(x, torch.masked.MaskedTensor):
        mask = ~x._masked_mask.detach().cpu().numpy()
        data = x._masked_data.detach().cpu().numpy()
        return np.ma.MaskedArray(data, mask=mask)
    return x.detach().cpu().numpy()



class MaskedSequential(nn.Sequential):
    """ Version of nn.Sequential that can handle masked tensors """
    def __init__(self, *args):
        super().__init__(*args)
    
    def forward(self, x):
        isMasked = isinstance(x, masked.MaskedTensor)
        if isMasked:
            mask = x._masked_mask[...,:1]
            x    = x._masked_data
            req_grad = x.requires_grad
        for module in self:
            x = module(x)
        if isMasked:
            # account for different output, input dims
            newshape = (1,)*(len(x.shape)-1)+(x.shape[-1],)
            mask = mask.repeat(*newshape)
            return masked.masked_tensor(x, mask, requires_grad=req_grad).to(device)
        return x

def build_mlp(input_size, output_size, n_layers, hidden_size, activation=nn.ReLU()):
    """ Build multi-layer perception with n_layers hidden layers of size hidden_size. 
    network input can be torch.Tensor or torch.masked.MaskedTensor """
    layers = [nn.Linear(input_size,hidden_size),activation]
    for i in range(n_layers-1):
        layers.append(nn.Linear(hidden_size,hidden_size))
        layers.append(activation)
    layers.append(nn.Linear(hidden_size, output_size))
    return MaskedSequential(*layers).to(device)

def get_logger(filename):
    """ Return a logger instance to a file """
    logger = logging.getLogger("LOG")
    # log only info things, no debug
    logging.basicConfig(filename=filename, 
                        filemode = 'a',  # add to existing config
                        format='%(asctime)s:%(levelname)s:%(message)s', 
                        datefmt='%m-%d %H:%M',
                        level=logging.INFO)
    return logger

def normalize(x):
    """ Normalize np.ndarray or torch.Tensor """
    return (x - x.mean()) / (x.std() + 1e-10)

def get_lengths(arr: np.ma.MaskedArray) -> np.ndarray:
    """ print length of the true elements in arr """
    lengths = np.empty(arr.shape[0], dtype=int)
    for i in range(arr.shape[0]):  # get last non-masked value in each row
        trues = np.where(~arr.mask[i])[0]
        if len(trues.shape) > 1:
            trues = trues[:,0]  # we dont care about obs_dim
        if not len(trues): 
            lengths[i] = 0
            continue
        lengths[i] = trues[-1]
    return lengths

def get_finals(arr: np.ndarray | np.ma.MaskedArray) -> np.ndarray:
    """ Get final returns from batched returns of shape (nb x nt) """
    if not isinstance(arr, np.ma.MaskedArray):
        return arr[:,-1]
    finals = np.empty(arr.shape[0])
    for i in range(arr.shape[0]):  # get last non-masked value in each row
        trues = np.where(~arr.mask[i])[0]
        if not len(trues):
            finals[i] = 0
            continue
        finals[i] = arr[i, trues[-1]]
    return finals

def arrs_to_masked(arrays: list[list]):
    """
    Returns masked (2d) array from list of arrays of potentially different lengths
    """
    lens = [len(arr) for arr in arrays]
    all_arr = np.ma.empty( (len(arrays), max(lens)) )
    all_arr[:] = np.ma.masked
    for idx, arr in enumerate(arrays):
        all_arr[idx, :lens[idx]] = np.array(arr)
    # mask any np.inf or np.nan values
    all_arr = np.ma.masked_invalid(all_arr)
    return all_arr

def arrs_to_masked_3d(arrays: list[list[list]]):
    """
    Returns masked (3d) array from list of arrays of potentially different lengths
    """
    lens = [len(arr) for arr in arrays]
    try:
        sublens = [[len(arr) for arr in arrays] for arrays in arrays]
    except TypeError:   # a 2D array was passed in instead of a 3d one?
        return arrs_to_masked(arrays)
    all_arr = np.ma.empty( (len(arrays), max(lens), max(sublens)) )
    all_arr[:] = np.ma.masked
    for idx, arrs in enumerate(arrays):
        for jdx, arr in enumerate(arrs):
            all_arr[idx, :lens[idx], :sublens[idx][jdx]] = np.array(arr)
    return all_arr

def plot_trajectory(x, y, ax, color):
    ''' plot trajectory of form (nbatch x nt) '''
    ys = np.mean(y, axis = 0)
    yerrs = np.std(y, axis = 0)/np.sqrt(y.shape[-1])
    ax.fill_between(x, ys - yerrs, ys + yerrs, alpha=0.25, color=color)
    ax.plot(x, ys, color=color)

def plot_WIM(paths, dt: float, title='', savename='', isfinal=True):
    """ plot data from a batch of trajectories
    Inputs: (nbatch x nt) np.ndarrays """
    isfinal = True
    if isfinal:
        FIGSIZE = (10,9)
        mpl.rcParams['font.size'] = textsize
    else:
        FIGSIZE = (12,10/3)
        reset_matplotlib()
    if isinstance(paths, dict):  # when not using book_quit
        wealth = paths['wea']
        inventory = paths['inv']
        states = paths['book']
        mids = states[...,0]; high_bids = states[...,1]; low_asks = states[...,2]
    else:   # when using book_quit with variable-length trajectories, convert to masked array for plotting
        wealth    = arrs_to_masked([p['wea'] for p in paths])
        inventory = arrs_to_masked([p['inv'] for p in paths])
        states    = [p['book'] for p in paths]  # lists of (length x 3)
        mids      = arrs_to_masked([[i[0] for i in traj] for traj in states])
        high_bids = arrs_to_masked([[i[1] for i in traj] for traj in states])
        low_asks  = arrs_to_masked([[i[2] for i in traj] for traj in states])
    # states is nbatch x nt x (midprice, highest_bid, lowest_ask)
    times = np.arange(0, (wealth.shape[-1]+1)*dt, dt)
    times = times[:wealth.shape[-1]]
    fig, axs = plt.subplots(3,1, figsize=(FIGSIZE[0],3*FIGSIZE[1]), sharex=False)
    # plot states
    ax = axs[0]
    ax.set_ylabel('Order Book')
    ax.tick_params(axis='y',labelcolor='C3')
    plot_trajectory(times, mids, ax, 'C3')
    plot_trajectory(times, high_bids, ax, 'C0')
    plot_trajectory(times, low_asks, ax, 'C1')
    # plot wealth and inventory on same x axis
    ax = axs[1]
    ax.set_ylabel('Wealth')
    ax.tick_params(axis='y',labelcolor='C5')
    plot_trajectory(times, wealth, ax, 'C5')
    inv_ax = ax.twinx()
    inv_ax.set_ylabel('Inventory')
    inv_ax.tick_params(axis='y',labelcolor='C2')
    plot_trajectory(times, inventory, inv_ax, 'C2')
    # plot total value
    ax = axs[2]
    ax.set_ylabel('Total Value')
    value = wealth + inventory*mids
    plot_trajectory(times, value, ax, 'C6')
    for ax in axs:
        ax.set_xlabel('Time (s)')
        ax.grid(which='both', axis='both')
    if title: fig.suptitle(title, fontsize=0.8*textsize)
    fig.tight_layout()
    if savename: 
        fig.savefig(savename)
    else: 
        plt.show()
    plt.close()

def export_plot(y, ylabel, title, filename):
    """ plot values / final returns """
    mpl.rcParams['font.size'] = textsize
    fig, ax = plt.subplots(figsize=FIGSIZE)
    times = np.arange(0, y.shape[0])
    ys = y.mean(axis = -1)
    yerrs = y.std(axis = -1)/np.sqrt(y.shape[-1])
    ax.fill_between(times, ys - yerrs, ys + yerrs, alpha=0.25)
    ax.plot(times, ys)
    ax.set(xlabel='Training Episode',ylabel=ylabel, yscale='log')
    ax.set_title(title, fontsize=0.8*textsize)
    ax.grid(which='both', axis='both')
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

""" Used for exponential value functions """
class Exponential(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(()))
        self.b = nn.Parameter(torch.randn(()))

    def forward(self, x):
        return -abs(self.a) * torch.exp(-abs(self.b) * x)

    def __repr__(self):
        return f'y = -{abs(self.a.item())} * exp(-{abs(self.b.item())} x)'

def uFormat(number, uncertainty=0, figs = 4, shift = 0, FormatDecimals = False):
    """
    Returns "num_rounded(with_sgnfcnt_dgts_ofuncrtnty)", formatted to 10^shift
    According to section 5.3 of "https://pdg.lbl.gov/2011/reviews/rpp2011-rev-rpp-intro.pdf"

    Arguments:
    - float number:      the value
    - float uncertainty: the absolute uncertainty (stddev) in the value
       - if zero, will format number to optional number of sig_figs (see figs)
    - int shift:  optionally, shift the resultant number to a higher/lower digit expression
       - i.e. if number is in Hz and you want a string in GHz, specify shift = 9
               likewise for going from MHz to Hz, specify shift = -6
    - int figs: when uncertainty = 0, format number to degree of sig figs instead
       - if zero, will simply return number as string
    - bool FormatDecimals:  for a number 0.00X < 1e-2, option to express in "X.XXe-D" format
             for conciseness. doesnt work in math mode because '-' is taken as minus sign
    """
    num = str(number); err = str(uncertainty)
    
    sigFigsMode = not uncertainty    # UNCERTAINTY ZERO: IN SIG FIGS MODE
    if sigFigsMode and not figs: # nothing to format
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
    if len(jNum) == 0:  # our number is literally zero!
        return '0'
    
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
        ei = nBefore + figs

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
    if not shift and FormatDecimals and (ni-n) >= 2:
        shift -= ni - n + 1
        extraDigs = ni - n + 1
    
    # shift digits up/down by round argument
    ni += shift
    end = ''

    # there are digits to the right of decimal and we dont 
    # care about exact sig figs (to not format floats to 0.02000)
    if ni > 0 and sigFigsMode:
        while Num[-1] == '0':
            if len(Num) == 1: break
            Num = Num[:-1]
            ni -= 1
            n -= 1
    
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