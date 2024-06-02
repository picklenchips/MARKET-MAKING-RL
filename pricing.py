import numpy as np
import os
from util import mpl, plt, FIGSIZE, SAVEDIR, uFormat

import stochastic.processes as processes
import stochastic.processes.noise as noise
GaussianNoise = noise.GaussianNoise
from stochastic.processes.continuous.brownian_motion import BrownianMotion
from scipy.optimize import curve_fit

N_RUNS = 1000
DRIFT_VALUE = 0.5677
SCALE_VALUE = 8.685
MAX_T = 1
BASELINE = 533

times = np.arange(0,10,0.001)
N_TIMES = times.shape[0]

def line(x, m, b): return m*x + b
fig, ax = plt.subplots(figsize=FIGSIZE)

i = 0
slopes=[]
for run in range(N_RUNS):
    model = BrownianMotion(drift=DRIFT_VALUE, scale=SCALE_VALUE, t=MAX_T)
    new = BASELINE + model._sample_brownian_motion(1)[1]
    print(new)
#     coefs, covar = curve_fit(line, times, new)
#     slopes.append(coefs[0])
#     # ax.plot(times, new, c=f"C{i}", label=f"{DRIFT_VALUE} -> {uFormat(coefs[0],0)}")
#     # ax.plot(times, line(times, *coefs), c=f"C{i}", linestyle='dashed')
#     i += 1
# print(np.mean(slopes))
# # plt.legend()
# # plt.savefig(os.path.join(SAVEDIR, "drifting.png"))
# # plt.clf()

# for scale in [0.5,1,2,3]:
#     model = BrownianMotion(drift=0, scale=scale, t=MAX_T)
#     new = model._sample_brownian_motion(N_TIMES)[:N_TIMES]
#     coefs, covar = curve_fit(line, times, new)
#     ax.plot(times, new, c=f"C{i}", label=f"{scale} -> {uFormat(coefs[0],0)}")
#     ax.plot(times, line(times, *coefs), c=f"C{i}", linestyle='dashed')
#     i += 1

# plt.legend()
# plt.savefig(os.path.join(SAVEDIR, "pricing.png"))
# plt.close(fig)