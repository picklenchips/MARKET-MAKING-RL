import numpy as np
from util import mpl, plt, FIGSIZE, uFormat

import stochastic.processes.noise as noise
GaussianNoise = noise.GaussianNoise
from stochastic.processes.continuous.brownian_motion import BrownianMotion
from scipy.optimize import curve_fit


n_times = 100000
max_t = 1
times = np.linspace(0, max_t, n_times+1)
fig, ax = plt.subplots(figsize=FIGSIZE)

def line(x, m, b): return m*x + b
i = 0
for drift in [-1,-.5,.5,1]:
    model = BrownianMotion(drift=drift, scale=1, t=max_t)
    new = model._sample_brownian_motion(n_times)
    coefs, covar = curve_fit(line, times, new)
    ax.plot(times, new, c=f"C{i}", label=f"{drift} -> {uFormat(coefs[0],0)}")
    ax.plot(times, line(times, *coefs), c=f"C{i}", linestyle='dashed')
    i += 1
plt.legend()
plt.show()
fig, ax = plt.subplots(figsize=FIGSIZE)
for scale in [0.5,1,2,3]:
    model = BrownianMotion(drift=0, scale=scale, t=max_t)
    new = model._sample_brownian_motion(n_times)
    coefs, covar = curve_fit(line, times, new)
    ax.plot(times, new, c=f"C{i}", label=f"{scale} -> {uFormat(coefs[0],0)}")
    ax.plot(times, line(times, *coefs), c=f"C{i}", linestyle='dashed')
    i += 1
plt.legend()
plt.show()