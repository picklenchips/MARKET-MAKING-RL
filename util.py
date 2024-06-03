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
import matplotlib.ticker as mtick
import sys, time, os
from scipy.optimize import curve_fit
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
from scipy import stats

# blue = bids, red = asks
#           |   Blue  |   Red  |  Orange |  Purple | Yellow  |   Green |   Teal  | Grey
hexcolors = ['648FFF', 'DC267F', 'FE6100', '785EF0', 'FFB000', '009E73', '3DDBD9', '808080']
mpl.rcParams['axes.prop_cycle'] = cycler('color', [mpl.colors.to_rgba('#' + c) for c in hexcolors])

FIGSIZE = (10,6)
SAVEDIR = os.getcwd()+"/"
SAVEEXT = ".png"
np.set_printoptions(precision=4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def np2torch(x):
    x = torch.from_numpy(x).to(device)
    if x.dtype is torch.float64: x = x.float()  # cast double to float
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
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return logger

def normalize(x):
    """ Normalize np.ndarray or torch.Tensor """
    return (x - x.mean()) / (x.std() + 1e-10)

def plot_WIM(self, wealth: np.ndarray, inventory: np.ndarray, midprices: np.ndarray, dt: float, title=''):
    """ plot data from a batch of trajectories
    Inputs: (nbatch x nt) np.ndarrays """
    times = np.arange(0, wealth.shape[-1]*dt, dt)
    fig, axs = plt.subplots(3,1, figsize=(10,8))
    for i, y, name in zip((0,1,2),(wealth, inventory, midprices),('Wealth', 'Inventory', 'Midprice')):
        ax = axs[i]
        ax.set(ylabel=name)
        ys = np.mean(y, axis=0)
        yerrs = stats.sem(y, axis=0)
        ax.fill_between(times, ys - yerrs, ys + yerrs, alpha=0.25, color=f"C{i}")
        ax.plot(times, ys, color=f"C{i}")
    axs[2].set(xlabel="Time")
    i += 1
    y = wealth + inventory*midprices
    ys = np.mean(y, axis=0); yerrs = stats.sem(y, axis=0)
    axs[0].fill_between(times, ys - yerrs, ys + yerrs, alpha=0.25, color=f"C{i}")
    axs[0].plot(times, ys, label='Total Value', color=f"C{i}")
    axs[0].legend()
    if title: axs[0].set_title(title)
    plt.show()

def export_plot(y, ylabel, title, filename):
    """ yuh. """
    fig, ax = plt.subplots(figsize=(10,8))
    plt.plot(range(len(ys)), ys)
    times = np.arange(0, len(ys))
    ys = np.mean(y, axis=0)
    yerrs = stats.sem(y, axis=0)
    ax.fill_between(times, ys - yerrs, ys + yerrs, alpha=0.25)
    ax.plot(times, ys)
    ax.set(xlabel='Training Episode',ylabel=ylabel)
    ax.set_title(title)
    fig.savefig(filename)
    plt.show()

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

# --- robust number formatting to sig_figs
def uFormat(number, uncertainty, round = 0, sig_figs = 4, FormatDecimals = False):
    """
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
    if isinstance(number,str):
        if number == "None": return ""
    if isinstance(uncertainty,str):
        if uncertainty == "None": uncertainty = 0
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
        if not len(ff[1]):  # handles num="None"
            return ''
        ni = -int(ff[1])
    if 'e' in err:
        ff = err.split('e')
        err = ff[0]
        ei = -int(ff[1])

    if not num[0].isdigit():
        print("uFormat: {num} isn't a number")
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
        else: # 355 -> (4..)
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

# wrapper printing time a function runs in, use @timeIt before function def
def timeIt(func):
    """@ timeIt: Wrapper to print run time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.clock_gettime_ns(0)
        res = func(*args, **kwargs)
        end_time = time.clock_gettime_ns(0)
        diff = (end_time - start_time) * 10**(-9)
        print(func.__name__, 'ran in %.6fs' % diff)
        return res
    return wrapper

# search for first instance of X_val in array X
def binarySearch(X_val, X, decreasing=False):
    """
    For sorted X, returns i such that X[:i] < X_val, X[i:] >= X_val
     - if decreasing, returns i such that    X[:i] > X_val, X[i:] <= X_val
    """
    l = 0; r = len(X) - 1
    #print(f"searching for {X_val}, negative={negative}")
    m = (l + r) // 2
    while r > l:  # common binary search
        #print(f"{l}:{r} is {X[l:r+1]}, middle {X[m]}")
        if X[m] == X_val:  # repeat elements of X_val in array
            break
        if decreasing: # left is always larger than right
            if X[m] > X_val:
                l = m + 1
            else:
                r = m - 1
        else:        # right is always larger than left
            if X[m] < X_val:
                l = m + 1
            else:
                r = m - 1
        m = (l + r) // 2
    if r < l:
        return l
    if m + 1 < len(X):  # make sure we are always on right side of X_val
        if X[m] < X_val and not decreasing:
            return m + 1
        if X[m] > X_val and decreasing:
            return m + 1
    if X[m] == X_val:  # repeat elements of X_val in array
        if decreasing:
            while m > 0 and X[m - 1] == X_val:
                m -= 1
        elif not decreasing:
            while m + 1 < len(X) and X[m + 1] == X_val:
                m += 1
    return m

# linear interpolate 1D with sorted X
def linearInterpolate(x,X,Y):
    """example: 2D linear interpolate by adding interpolations from both
    - """
    i = binarySearch(x,X)
    if i == 0: i += 1  # lowest ting, interpolate backwards
    m = (Y[i]-Y[i-1])/(X[i]-X[i-1])
    b = Y[i] - m*X[i]
    return m*x + b