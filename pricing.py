import numpy as np
from util import mpl, plt, FIGSIZE, uFormat

import stochastic.processes as processes
import stochastic.processes.noise as noise
GaussianNoise = noise.GaussianNoise
from stochastic.processes.continuous.brownian_motion import BrownianMotion
from scipy.optimize import curve_fit

max_t = 1
times = np.arange(0,10,0.001)
n_times = times.shape[0]
fig, ax = plt.subplots(figsize=FIGSIZE)
print(n_times)

lamb = 1
poiss = processes.PoissonProcess(rate=lamb)
poiss.sample(n_times, max_t)

def line(x, m, b): return m*x + b
i = 0
slopes=[]
n_runs = 1000
for drift in [0.5677]*n_runs:
    model = BrownianMotion(drift=drift, scale=8.685, t=max_t)
    new = 533+model._sample_brownian_motion(n_times)[:n_times]
    coefs, covar = curve_fit(line, times, new)
    #ax.plot(times, new, c=f"C{i}", label=f"{drift} -> {uFormat(coefs[0],0)}")
    #ax.plot(times, line(times, *coefs), c=f"C{i}", linestyle='dashed')
    slopes.append(coefs[0])
    i += 1
print(np.mean(slopes))
#plt.legend()
#plt.show()
fig, ax = plt.subplots(figsize=FIGSIZE)
for scale in [0.5,1,2,3]:
    model = BrownianMotion(drift=0, scale=scale, t=max_t)
    new = model._sample_brownian_motion(n_times)[:n_times]
    coefs, covar = curve_fit(line, times, new)
    ax.plot(times, new, c=f"C{i}", label=f"{scale} -> {uFormat(coefs[0],0)}")
    ax.plot(times, line(times, *coefs), c=f"C{i}", linestyle='dashed')
    i += 1
plt.legend()
plt.show()